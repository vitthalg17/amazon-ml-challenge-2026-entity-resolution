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
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import WORK_DIR
import stage1
from prep import load_norm

N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))
CHUNK = int(os.environ.get("ER_BLOCK_CHUNK", 300_000))
MAX_DF_FRAC = float(os.environ.get("ER_MAX_DF_FRAC", 0.01))  # retrieval-side feature df cap
MAX_DF_MIN = 2000
EXACT_MAX_GROUP = int(os.environ.get("ER_EXACT_MAX_GROUP", 50))

_COMMON = dict(sublinear_tf=True, dtype=np.float32, lowercase=False)
CHANNEL_SPECS = {
    "name_word": dict(k=10, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                                         min_df=1, **_COMMON)),
    "name_char": dict(k=10, vec=lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4),
                                                         min_df=1, **_COMMON)),
    "addr": dict(k=10, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                                    min_df=2, **_COMMON)),
    "combo": dict(k=10, vec=lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 1),
                                                     min_df=1, **_COMMON)),
}
CHANNELS = tuple(CHANNEL_SPECS)
RANK_COLS = [f"rank_{c}" for c in CHANNELS] + ["rank_exact"]


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


def row_cosine(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
               step: int = 2_000_000) -> np.ndarray:
    """cos(A[ia[k]], B[ib[k]]) for all k (rows are already L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), step):
        e = s + step
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


def topk_pairs(Q: sp.csr_matrix, BT: sp.csr_matrix, k: int, thresh: float, col: str,
               offset: int) -> pl.DataFrame:
    C = sp_matmul_topn(Q, BT, top_n=k, threshold=thresh, sort=True, n_threads=N_THREADS)
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int32), counts)
    rank = np.arange(len(rows)) - np.repeat(C.indptr[:-1], counts)
    return pl.DataFrame({"oi": rows + offset, "si": C.indices.astype(np.int32),
                         col: rank.astype(np.int8)})


def exact_pairs(s1: pl.DataFrame, oth: pl.DataFrame) -> pl.DataFrame:
    a = s1.select(key=name_key_expr()).with_row_index("si")
    a = a.filter(pl.len().over("key") <= EXACT_MAX_GROUP)
    b = oth.select(key=name_key_expr()).with_row_index("oi")
    return (b.join(a, on="key").select(pl.col("oi").cast(pl.Int32), pl.col("si").cast(pl.Int32))
            .with_columns(rank_exact=pl.lit(0, dtype=pl.Int8)))


def block_country(s1: pl.DataFrame, oth: pl.DataFrame, k: dict[str, int] | None = None,
                  pruner=None, thresh: float = 0.05) -> pl.DataFrame:
    """Candidates for one country. Processes S2/S3 records in chunks of CHUNK so memory stays
    bounded; when `pruner` (stage-1 model) is given, each chunk is pruned before it is kept."""
    t0 = time.time()
    max_df = max(MAX_DF_MIN, int(MAX_DF_FRAC * s1.height))
    t1, to = channel_texts(s1), channel_texts(oth)
    mats = {}
    for ch, spec in CHANNEL_SPECS.items():
        vec = spec["vec"]()
        A = vec.fit_transform(t1[ch]).tocsr()
        Q = vec.transform(to[ch]).tocsr()
        BT = A.T.tocsr()
        # Retrieval cost is the sum of posting-list lengths of each query's features, so very
        # common features (" sh", "rd", "mh") are dropped from the *query* side for retrieval
        # only. They carry little IDF weight; full vectors are kept for the cosines.
        Qr = Q @ sp.diags((np.diff(BT.indptr) <= max_df).astype(np.float32))
        Qr.eliminate_zeros()
        mats[ch] = (Q, A, BT, Qr)
    del t1, to
    print(f"    vectorized in {time.time() - t0:.0f}s", flush=True)

    exact = exact_pairs(s1, oth)
    out, n_raw = [], 0
    for s in range(0, oth.height, CHUNK):
        e = min(s + CHUNK, oth.height)
        parts = [exact.filter(pl.col("oi").is_between(s, e - 1))]
        for ch, (Q, A, BT, Qr) in mats.items():
            parts.append(topk_pairs(Qr[s:e], BT, (k or {}).get(ch, CHANNEL_SPECS[ch]["k"]),
                                    thresh, f"rank_{ch}", s))
        cand = (pl.concat(parts, how="diagonal")
                .group_by("oi", "si")
                .agg(*[pl.col(c).min() for c in RANK_COLS]))
        oi, si = cand["oi"].to_numpy(), cand["si"].to_numpy()
        cand = cand.with_columns(
            **{f"cos_{ch}": pl.Series(row_cosine(Q, A, oi, si)) for ch, (Q, A, _, _) in mats.items()},
            **{c: pl.col(c).fill_null(99) for c in RANK_COLS},
        )
        n_raw += cand.height
        if pruner is not None:
            cand = stage1.prune(cand, pruner, CHANNELS, RANK_COLS, key="oi")
        out.append(cand)
    cand = pl.concat(out)
    cand = cand.with_columns(
        other_id=oth["entity_id"].gather(cand["oi"]),
        s1_id=s1["entity_id"].gather(cand["si"]),
    ).drop("oi", "si")
    print(f"    {n_raw:,} raw -> {cand.height:,} kept candidate pairs for {oth.height:,} records "
          f"({cand.height / max(oth.height, 1):.2f}/record) in {time.time() - t0:.0f}s", flush=True)
    return cand


def candidates_path(split: str, pct: int = 100, raw: bool = False):
    tag = ("" if pct >= 100 else f"_p{pct}") + ("_raw" if raw else "")
    return WORK_DIR / f"{split}_candidates{tag}.parquet"


def run(split: str, pct: int = 100, k: dict[str, int] | None = None, raw: bool = False):
    """raw=True skips stage-1 pruning (used to produce stage-1 training data)."""
    pruner = None if raw else stage1.load_model()
    if not raw and pruner is None:
        raise SystemExit("stage-1 model missing: run blocking --raw on a train sample, then stage1.py")
    s1 = load_norm(split, 1, 100)
    oth = pl.concat([load_norm(split, s, pct) for s in (2, 3)])
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        a = s1.filter(pl.col("country") == country)
        b = oth.filter(pl.col("country") == country)
        print(f"[{country}] S1={a.height:,} S2/S3={b.height:,}", flush=True)
        if a.height and b.height:
            out.append(block_country(a, b, k, pruner).with_columns(country=pl.lit(country)))
    # S2/S3 records whose country never appears in S1 cannot match anything.
    cand = pl.concat(out)
    cand.write_parquet(candidates_path(split, pct, raw))
    return cand


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--k", type=int, nargs=len(CHANNELS), default=None, metavar=CHANNELS)
    ap.add_argument("--raw", action="store_true", help="skip stage-1 pruning")
    a = ap.parse_args()
    run(a.split, a.pct, dict(zip(CHANNELS, a.k)) if a.k else None, a.raw)
