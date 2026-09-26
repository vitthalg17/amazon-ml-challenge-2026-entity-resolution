"""Blocking / candidate generation.

For every Source 2/3 record, retrieve its top-K Source 1 records within the same country
through several TF-IDF channels (IDF fitted on Source 1 of that country), plus one exact key:
  name_word - core-name words + word bigrams + an order-free whole-name key
  name_char - char 4-grams of the core name (typos, OCR-style digit swaps, transliteration)
  addr      - word uni+bi-grams of the normalized address (renamed / gibberish names)
  combo     - name words + address words in one space (weak name + weak address)
  exact     - identical order-free core name (all S1 with that name, if the group is small)
The union is the candidate set. The TF-IDF cosine of every channel is then computed for
every candidate pair and reused as model features.
"""
import argparse
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import SEED, WORK_DIR, pq_path
import stage1
from prep import load_norm

N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))
CHUNK = int(os.environ.get("ER_BLOCK_CHUNK", 300_000))
MAX_DF_FRAC = float(os.environ.get("ER_MAX_DF_FRAC", 0.01))  # retrieval-side feature df cap
MAX_DF_MIN = 2000
EXACT_MAX_GROUP = int(os.environ.get("ER_EXACT_MAX_GROUP", 50))
# names shared by more S1 entities than that ("Pediatric National Medicine", "Bordeaux Club"):
# every S1 with the name is scored by address similarity and the best EXACT_ADDR_K are kept
EXACT_BIG_MAX = int(os.environ.get("ER_EXACT_BIG_MAX", 3000))
EXACT_ADDR_K = int(os.environ.get("ER_EXACT_ADDR_K", 5))
_u = os.environ.get("ER_UNSEEN_PMIN")
UNSEEN_PMIN = float(_u) if _u else None  # stage-1 floor for countries absent from training

_COMMON = dict(sublinear_tf=True, dtype=np.float32, lowercase=False)
CHANNEL_SPECS = {
    "name_word": dict(k=20, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                                         min_df=1, **_COMMON)),
    # char 4-grams have long posting lists: a tighter df cap halves top-K time at -0.08pp recall
    # (measured on a 2% train sample)
    "name_char": dict(k=10, max_df_frac=float(os.environ.get("ER_NAME_CHAR_MAX_DF_FRAC", 0.002)),
                      vec=lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4),
                                                  min_df=1, **_COMMON)),
    "addr": dict(k=20, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                                    min_df=2, **_COMMON)),
    "combo": dict(k=20, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 1),
                                                     min_df=1, **_COMMON)),
}
CHANNELS = tuple(CHANNEL_SPECS)
RANK_COLS = [f"rank_{c}" for c in CHANNELS] + ["rank_exact", "rank_exact_addr"]


def name_key_expr() -> pl.Expr:
    """Order-free core-name key ('galaxy properties' == 'properties galaxy')."""
    core = pl.when(pl.col("name_core") != "").then(pl.col("name_core")).otherwise(pl.col("name_norm"))
    return core.str.split(" ").list.sort().list.join("_")


def channel_texts(df: pl.DataFrame) -> dict[str, list[str]]:
    core = pl.when(pl.col("name_core") != "").then(pl.col("name_core")).otherwise(pl.col("name_norm"))
    t = df.select(
        name=core,
        name_word=pl.concat_str(core, pl.lit(" K_") + name_key_expr()),
        combo=pl.concat_str(core.str.replace_all(r"(\S+)", "n_$1"), pl.col("addr_norm"),
                            separator=" "),
        addr=pl.col("addr_norm"),
    )
    return {"name_word": t["name_word"].to_list(), "name_char": t["name"].to_list(),
            "addr": t["addr"].to_list(), "combo": t["combo"].to_list()}


COSINE_STEP = int(os.environ.get("ER_COSINE_STEP", 200_000))


def row_cosine(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
               step: int = COSINE_STEP) -> np.ndarray:
    """cos(A[ia[k]], B[ib[k]]) for all k (rows are already L2-normalised). Rows are gathered
    `step` pairs at a time; this runs for 4 channels in parallel, so keep `step` modest."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), step):
        e = s + step
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


def topk_pairs(Q: sp.csr_matrix, BT: sp.csr_matrix, k: int, thresh: float, col: str,
               offset: int, n_threads: int = N_THREADS) -> pl.DataFrame:
    C = sp_matmul_topn(Q, BT, top_n=k, threshold=thresh, sort=True, n_threads=n_threads)
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int32), counts)
    rank = np.arange(len(rows)) - np.repeat(C.indptr[:-1], counts)
    return pl.DataFrame({"oi": rows + offset, "si": C.indices.astype(np.int32),
                         col: rank.astype(np.int8)})


def exact_pairs(s1_small: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """keys: (oi, key) of the queries; s1_small: (key, si) of S1 names shared by <= EXACT_MAX_GROUP
    entities. Every S1 with the identical order-free core name is a candidate."""
    return (keys.join(s1_small, on="key").select(pl.col("oi").cast(pl.Int32),
                                                 pl.col("si").cast(pl.Int32))
            .with_columns(rank_exact=pl.lit(0, dtype=pl.Int8)))


def exact_addr_pairs(keys: pl.DataFrame, big: pl.DataFrame, Q: sp.csr_matrix, A: sp.csr_matrix,
                     offset: int, max_pairs: int = 1_000_000) -> pl.DataFrame:
    """For S2/S3 records whose name key is shared by many S1 entities (big: key, si), score every
    S1 with that name by address cosine and keep the best EXACT_ADDR_K. Q holds this chunk's
    address vectors (row = oi - offset), A all S1 address vectors.
    Records are processed in batches whose joined pairs stay under max_pairs, and each batch is
    cut to its top EXACT_ADDR_K per record straight away: common Indian names ("X Traders") have
    hundreds of S1 entities each, and joining a whole chunk at once needed several GB."""
    empty = pl.DataFrame(schema={"oi": pl.Int32, "si": pl.Int32, "rank_exact_addr": pl.Int8})
    if keys.height == 0:
        return empty
    k = keys.join(big.group_by("key").agg(n=pl.len()), on="key")
    k = k.with_columns(batch=((pl.col("n").cum_sum() - 1) // max_pairs))
    out = []
    for (_,), kb in k.group_by("batch", maintain_order=True):
        part = kb.select("oi", "key").join(big, on="key").select("oi", "si")
        oi, si = part["oi"].to_numpy(), part["si"].to_numpy()
        part = part.with_columns(c=pl.Series(row_cosine(Q, A, oi - offset, si))).filter(pl.col("c") > 0)
        out.append(part.with_columns(r=pl.col("c").rank("ordinal", descending=True).over("oi") - 1)
                   .filter(pl.col("r") < EXACT_ADDR_K))  # before the Int8 cast: groups > 127
    if not out:
        return empty
    return pl.concat(out).select(pl.col("oi").cast(pl.Int32), pl.col("si").cast(pl.Int32),
                                 rank_exact_addr=pl.col("r").cast(pl.Int8))


def block_country(s1: pl.DataFrame, qpath, n: int, k: dict[str, int] | None = None,
                  pruner=None, thresh: float = 0.05, cross_fit: bool = False,
                  prune_pmin: float | None = None, spill_dir=None) -> None:
    """Candidates for one country. The S2/S3 queries are streamed from `qpath` (a parquet file
    with the country's n query records) CHUNK rows at a time, so only the S1 index and one chunk
    are ever in memory. When `pruner` (stage-1 model) is given, each chunk is pruned before it
    is kept. Each chunk's result is written to spill_dir.
    cross_fit (train only): records the main pruner was trained on are pruned by the alt one."""
    t0 = time.time()
    t1 = channel_texts(s1)
    fitted = {}
    for ch, spec in CHANNEL_SPECS.items():
        vec = spec["vec"]()
        A = vec.fit_transform(t1[ch]).tocsr()
        vec.stop_words_ = None  # every term cut by min_df (millions of bigrams); never used again
        BT = A.T.tocsr()
        # Retrieval cost is the sum of posting-list lengths of each query's features, so very
        # common features (" sh", "rd", "mh") are dropped from the *query* side for retrieval
        # only. They carry little IDF weight; full vectors are kept for the cosines.
        max_df = max(MAX_DF_MIN, int(spec.get("max_df_frac", MAX_DF_FRAC) * s1.height))
        keep = sp.diags((np.diff(BT.indptr) <= max_df).astype(np.float32))
        fitted[ch] = (vec, A, BT, keep)
        t1[ch] = None
    del t1
    s1_keys = s1.select(key=name_key_expr()).with_row_index("si").with_columns(
        pl.col("si").cast(pl.Int32)).with_columns(n=pl.len().over("key"))
    small = s1_keys.filter(pl.col("n") <= EXACT_MAX_GROUP).select("key", "si")
    big = s1_keys.filter(pl.col("n").is_between(EXACT_MAX_GROUP + 1, EXACT_BIG_MAX)).select("key", "si")
    big_keys = big.select("key").unique()
    s1_ids = s1["entity_id"]
    del s1_keys
    print(f"    vectorized in {time.time() - t0:.0f}s", flush=True)

    n_raw, n_kept = 0, 0
    with ThreadPoolExecutor(len(fitted)) as pool:  # per-channel cosines run in parallel
        for s in range(0, n, CHUNK):
            q = pl.scan_parquet(qpath).slice(s, CHUNK).collect()
            keys = q.select(key=name_key_expr()).with_row_index("oi", offset=s).with_columns(
                pl.col("oi").cast(pl.Int32))
            parts = [exact_pairs(small, keys)]
            to = channel_texts(q)
            qs = {}
            for ch, (vec, A, BT, keep) in fitted.items():
                Q = qs[ch] = vec.transform(to[ch]).tocsr()
                Qr = Q @ keep
                Qr.eliminate_zeros()
                parts.append(topk_pairs(Qr, BT, (k or {}).get(ch, CHANNEL_SPECS[ch]["k"]),
                                        thresh, f"rank_{ch}", s))
            parts.append(exact_addr_pairs(keys.join(big_keys, on="key", how="semi"),
                                          big, qs["addr"], fitted["addr"][1], s))
            cand = (pl.concat(parts, how="diagonal")
                    .group_by("oi", "si")
                    .agg(*[pl.col(c).min() for c in RANK_COLS]))
            oi, si = cand["oi"].to_numpy(), cand["si"].to_numpy()
            cos = dict(zip(fitted, pool.map(
                lambda ch: row_cosine(qs[ch], fitted[ch][1], oi - s, si), list(fitted))))
            cand = cand.with_columns(
                **{f"cos_{ch}": pl.Series(v) for ch, v in cos.items()},
                **{c: pl.col(c).fill_null(99) for c in RANK_COLS},
            )
            n_raw += cand.height
            if pruner is not None:
                in_main = None
                if cross_fit and pruner["half"]:
                    in_main = (q["entity_id"].hash(SEED) % 100 < pruner["half"]).to_numpy()[oi - s]
                cand = stage1.prune(cand, pruner, CHANNELS, RANK_COLS, key="oi",
                                    in_main_sample=in_main, pmin=prune_pmin)
            cand = cand.with_columns(
                other_id=q["entity_id"].gather(cand["oi"] - s),
                s1_id=s1_ids.gather(cand["si"]),
            ).drop("oi", "si")
            n_kept += cand.height
            cand.write_parquet(spill_dir / f"{s:012d}.parquet")
            del cand, to, parts, q, keys
    print(f"    {n_raw:,} raw -> {n_kept:,} kept candidate pairs for {n:,} records "
          f"({n_kept / max(n, 1):.2f}/record) in {time.time() - t0:.0f}s", flush=True)


def candidates_path(split: str, pct: int = 100, raw: bool = False):
    tag = ("" if pct >= 100 else f"_p{pct}") + ("_raw" if raw else "")
    return WORK_DIR / f"{split}_candidates{tag}.parquet"


BLOCK_COLS = ["entity_id", "name_core", "name_norm", "addr_norm"]  # all blocking needs


def _run_signature(split: str, pct: int, raw: bool) -> str:
    """What a partial blocking run was made with: resumable only if all of this is unchanged."""
    import hashlib
    h = hashlib.sha1(f"{split}|{pct}|{raw}".encode())  # CHUNK does not change results
    h.update(Path(__file__).read_bytes())
    for f in [stage1.MODEL_PATH, stage1.ALT_PATH] + [
            norm_path_for(split, src, pct) for src in (1, 2, 3)]:
        if f.exists():
            h.update(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()


def norm_path_for(split: str, source: int, pct: int):
    from prep import _source_file
    return _source_file(split, source, pct)


def run(split: str, pct: int = 100, k: dict[str, int] | None = None, raw: bool = False):
    """raw=True skips stage-1 pruning (used to produce stage-1 training data).
    Resumable: each finished country leaves a _DONE marker in the spill folder; a re-run with the
    same inputs and code skips those countries."""
    pruner = None if raw else stage1.load_model()
    if not raw and pruner is None:
        raise SystemExit("stage-1 model missing: run blocking --raw on a train sample, then stage1.py")
    countries = sorted(load_norm(split, 1, 100, columns=["country"], lazy=True)
                       .select(pl.col("country").unique()).collect()["country"].to_list())
    # countries absent from training (France): same pruning rule by default. Keeping the top 10
    # regardless of probability (ER_UNSEEN_PMIN=0) gave France ~10 candidates per record vs
    # ~1.5 in training, which shifts the stage-2 context features (ncand, gaps, ranks)
    seen = set(pl.scan_parquet(pq_path("train", 1)).select(pl.col("country").unique())
               .collect()["country"].to_list())
    out_path = candidates_path(split, pct, raw)
    spill = out_path.with_suffix(".parts")
    sig = _run_signature(split, pct, raw)
    sig_file = spill / "_SIGNATURE"
    if spill.exists() and not (sig_file.exists() and sig_file.read_text() == sig):
        shutil.rmtree(spill)  # partial results from other inputs / code
    spill.mkdir(exist_ok=True)
    sig_file.write_text(sig)
    for country in countries:
        cdir = spill / country
        if (cdir / "_DONE").exists():
            print(f"[{country}] already done (resumed)", flush=True)
            continue
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir()
        # one country at a time: S1 in memory, the S2/S3 queries streamed from a temp file
        a = (load_norm(split, 1, 100, columns=BLOCK_COLS + ["country"], lazy=True)
             .filter(pl.col("country") == country).drop("country").collect())
        qpath = spill / f"_queries_{country}.parquet"
        pl.concat([load_norm(split, s, pct, columns=BLOCK_COLS + ["country"], lazy=True)
                   .filter(pl.col("country") == country).drop("country") for s in (2, 3)]
                  ).sink_parquet(qpath, row_group_size=CHUNK)
        n = pl.scan_parquet(qpath).select(pl.len()).collect().item()
        print(f"[{country}] S1={a.height:,} S2/S3={n:,}", flush=True)
        if a.height and n:
            block_country(a, qpath, n, k, pruner, cross_fit=split == "train",
                          prune_pmin=UNSEEN_PMIN if country not in seen else None, spill_dir=cdir)
        qpath.unlink()
        (cdir / "_DONE").touch()
        del a
    # S2/S3 records whose country never appears in S1 cannot match anything.
    parts = [pl.scan_parquet(d / "*.parquet").with_columns(country=pl.lit(d.name))
             for d in sorted(p for p in spill.iterdir() if p.is_dir()) if any(d.glob("*.parquet"))]
    pl.concat(parts, how="diagonal").sink_parquet(out_path)
    shutil.rmtree(spill)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--k", type=int, nargs=len(CHANNELS), default=None, metavar=CHANNELS)
    ap.add_argument("--raw", action="store_true", help="skip stage-1 pruning")
    a = ap.parse_args()
    run(a.split, a.pct, dict(zip(CHANNELS, a.k)) if a.k else None, a.raw)
