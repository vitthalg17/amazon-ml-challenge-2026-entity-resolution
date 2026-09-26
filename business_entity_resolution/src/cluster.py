"""Stage 3: cluster-consistency re-scoring.

Stage 2 judges each (S2/S3 record, S1 entity) pair on its own. But an S1 entity usually has
several S2/S3 copies, and they are strong evidence about a new candidate: a distractor tends to
disagree with the entity's other copies (different house number, legal form, street), a noisy
true copy agrees with most of them.

For every candidate pair (o, s) the "siblings" are the other records whose best S1 is s with
stage-2 probability >= MEMBER_P. Features compare o with its siblings (name, address, street,
house number, legal form) and summarise the competition for o (its stage-2 probability, the gap
to its best candidate, ...). A small LightGBM re-scores the pairs:
  train  fitted on the same fit rows as stage 2, out-of-fold by S1 entity; the other rows get
         the fold-model average (honest: never fitted)
  test   fold-model average
Stage-2 probabilities used as inputs are themselves out-of-fold / never-fitted (match.score), so
no label leaks into the sibling features.
Memory: everything runs per S1-hash part (one short-lived process per part for the sibling
features, part-by-part matrices for fitting and scoring); only slim hashed tables are global.
Output: <work>/{train,test}_scores3.parquet.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

import features
from config import SEED, WORK_DIR
from match import N_FOLDS, N_THREADS, fit_rows, scores_path
from prep import load_norm

MEMBER_P = float(os.environ.get("ER_MEMBER_P", 0.5))
PAIRS_PER_PART = int(os.environ.get("ER_SIB_PART", 400_000))
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=N_THREADS, deterministic=True,
              force_row_wise=True)
N_ROUNDS = int(os.environ.get("ER_ROUNDS3", 2000))
ATTR_COLS = ["entity_id", "name_core", "addr_norm", "street", "house", "business_name"]


def members_path(split):
    return WORK_DIR / f"{split}_members.parquet"


def pctx_path(split):
    return WORK_DIR / f"{split}_pctx.parquet"


def sib_dir(split):
    return WORK_DIR / f"{split}_sib"


# ---------------------------------------------------------------- sibling features
def sib_part(split: str, pct: int, i: int, n_parts: int):
    """Sibling features for the pairs of the S1 entities hashed to part i."""
    def in_part(col):
        return pl.col(col).hash(SEED) % n_parts == i

    out_path = sib_dir(split) / f"part_{i:03d}.parquet"
    if out_path.exists():
        return  # finished in an earlier (interrupted) run
    pairs = (pl.scan_parquet(scores_path(split, 2)).filter(in_part("s1_id"))
             .select("s1_id", "other_id").collect(engine="streaming"))
    mem = (pl.scan_parquet(members_path(split)).filter(in_part("s1_id"))
           .select("s1_id", pl.col("other_id").alias("o2"), pl.col("p").alias("p2"))
           .collect(engine="streaming"))
    j = pairs.join(mem, on="s1_id").filter(pl.col("other_id") != pl.col("o2"))
    ids = pl.concat([j["other_id"], j["o2"]]).unique().to_frame("entity_id").lazy()
    attrs = pl.concat([load_norm(split, s, pct, columns=ATTR_COLS, lazy=True)
                       .join(ids, on="entity_id", how="semi").collect(engine="streaming")
                       for s in (2, 3)])
    attrs = attrs.with_columns(
        legal=features._legal_tokens("business_name").list.sort().list.join(" ")).drop("business_name")
    j = (j.join(attrs, left_on="other_id", right_on="entity_id")
         .join(attrs, left_on="o2", right_on="entity_id", suffix="_2"))

    def cp(a, b):
        return process.cpdist(j[a].to_list(), j[b].to_list(), scorer=fuzz.token_set_ratio,
                              workers=N_THREADS, dtype=np.float32)

    j = j.with_columns(name_sim=pl.Series(cp("name_core", "name_core_2")),
                       addr_sim=pl.Series(cp("addr_norm", "addr_norm_2")),
                       street_sim=pl.Series(cp("street", "street_2")))
    both_h = (pl.col("house") != "") & (pl.col("house_2") != "")
    both_s = (pl.col("street") != "") & (pl.col("street_2") != "")
    both_a = (pl.col("addr_norm") != "") & (pl.col("addr_norm_2") != "")
    j = j.with_columns(
        addr_sim=pl.when(both_a).then("addr_sim"),
        street_sim=pl.when(both_s).then("street_sim"),
        house_eq=pl.when(both_h).then((pl.col("house") == pl.col("house_2")).cast(pl.Float32)),
        legal_eq=(pl.col("legal") == pl.col("legal_2")).cast(pl.Float32),
    )
    agg = j.group_by("s1_id", "other_id").agg(
        sib_n=pl.len().cast(pl.Int16),
        sib_psum=pl.col("p2").sum(),
        sib_name_max=pl.col("name_sim").max(), sib_name_mean=pl.col("name_sim").mean(),
        sib_addr_max=pl.col("addr_sim").max(), sib_addr_mean=pl.col("addr_sim").mean(),
        sib_street_max=pl.col("street_sim").max(),
        sib_house_agree=pl.col("house_eq").mean(), sib_house_any=pl.col("house_eq").max(),
        sib_house_n=pl.col("house_eq").count().cast(pl.Int16),
        sib_legal_agree=pl.col("legal_eq").mean(),
    )
    j = attrs = mem = None  # free before the joins below
    out = pairs.join(agg, on=["s1_id", "other_id"], how="left").with_columns(
        pl.col("sib_n").fill_null(0), pl.col("sib_psum").fill_null(0.0))
    ctx = pl.scan_parquet(pctx_path(split)).filter(pl.col("_part") == i).drop("_part").collect()
    out = (out.with_columns(_o=pl.col("other_id").hash(), _s=pl.col("s1_id").hash())
           .join(ctx, on=["_o", "_s"], how="left").drop("_o", "_s"))
    if split == "train":
        gt = (pl.scan_parquet(WORK_DIR / "train_gt_pairs.parquet").filter(in_part("s1"))
              .with_columns(label=pl.lit(1, pl.Int8)).collect())
        out = (out.join(gt, left_on=["s1_id", "other_id"], right_on=["s1", "other"], how="left")
               .with_columns(pl.col("label").fill_null(0)))
    tmp = out_path.with_suffix(".tmp")
    out.select("s1_id", "other_id", *FEATS3, *(["label"] if split == "train" else [])
               ).write_parquet(tmp)
    os.replace(tmp, out_path)
    print(f"    sib part {i + 1}/{n_parts}: {pairs.height:,} pairs", flush=True)


def _signature(split: str, n_parts: int) -> str:
    """Inputs of the sibling parts: resumable only while all of these are unchanged."""
    h = hashlib.sha1(Path(__file__).read_bytes())
    f = scores_path(split, 2)
    h.update(f"{f.stat().st_size}:{f.stat().st_mtime_ns}:{MEMBER_P}:{n_parts}".encode())
    return h.hexdigest()


def _p_context(split: str, n_parts: int):
    """The stage-2 score in context of the record's other candidates (needs all of them, so it
    is computed globally, on hashed keys only) -> pctx parquet, sorted by S1 part."""
    p = pl.col("p").clip(1e-6, 1 - 1e-6)
    (pl.scan_parquet(scores_path(split, 2))
     .select(_o=pl.col("other_id").hash(), _s=pl.col("s1_id").hash(),
             _part=(pl.col("s1_id").hash(SEED) % n_parts).cast(pl.Int32), p=pl.col("p"))
     .collect(engine="streaming")
     .with_columns(
         p_logit=(p / (1 - p)).log(),
         p_gap_o=pl.col("p").max().over("_o") - pl.col("p"),
         p_rank_o=pl.col("p").rank("ordinal", descending=True).over("_o").cast(pl.Int16),
         p_second_o=pl.col("p").sort(descending=True).slice(1, 1).first().over("_o"),
         ncand_o=pl.len().over("_o").cast(pl.Int16),
         p_sum_s=pl.col("p").sum().over("_s"),
         ncand_s=pl.len().over("_s").cast(pl.Int16))
     .sort("_part")
     .write_parquet(pctx_path(split), row_group_size=100_000))


def build_siblings(split: str, pct: int):
    t = time.time()
    n_pairs = pl.scan_parquet(scores_path(split, 2)).select(pl.len()).collect().item()
    n_parts = max(1, -(-n_pairs // PAIRS_PER_PART))
    d = sib_dir(split)
    sig = _signature(split, n_parts)
    if not (d / "_SIG").exists() or (d / "_SIG").read_text() != sig:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        # members: each record's best S1 if p >= MEMBER_P (filter first: only those rows load)
        (pl.scan_parquet(scores_path(split, 2)).filter(pl.col("p") >= MEMBER_P)
         .collect(engine="streaming")
         .sort(["p", "s1_id"], descending=[True, False])
         .unique("other_id", keep="first", maintain_order=True)
         .write_parquet(members_path(split)))
        _p_context(split, n_parts)
        (d / "_SIG").write_text(sig)
    for i in range(n_parts):  # own process per part (memory is returned to the OS)
        if (d / f"part_{i:03d}.parquet").exists():
            continue
        rc = subprocess.call([sys.executable, "-u", __file__, "sib-part", "--split", split,
                              "--pct", str(pct), "--part", str(i), "--nparts", str(n_parts)])
        if rc:
            raise SystemExit(f"sib-part {i} failed ({rc})")
    print(f"{split}: sibling features in {n_parts} parts, {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- stage-3 model
FEATS3 = ["p", "p_logit", "p_gap_o", "p_rank_o", "p_second_o", "ncand_o", "p_sum_s", "ncand_s",
          "sib_n", "sib_psum", "sib_name_max", "sib_name_mean", "sib_addr_max", "sib_addr_mean",
          "sib_street_max", "sib_house_agree", "sib_house_any", "sib_house_n", "sib_legal_agree"]


def _parts(split):
    return sorted(sib_dir(split).glob("part_*.parquet"))


def train_and_score(fit_pct: int):
    t = time.time()
    fit_expr = fit_rows(fit_pct) if fit_pct < 100 else pl.lit(True)
    parts = _parts("train")
    counts = [pl.scan_parquet(f).filter(fit_expr).select(pl.len()).collect().item() for f in parts]
    n = sum(counts)
    X = np.empty((n, len(FEATS3)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    fold = np.empty(n, dtype=np.int64)
    es_group = np.empty(n, dtype=np.int64)
    off = 0
    for f, c in zip(parts, counts):
        df = pl.scan_parquet(f).filter(fit_expr).select("s1_id", "label", *FEATS3).collect()
        X[off:off + c] = df.select(FEATS3).cast(pl.Float32).to_numpy()
        y[off:off + c] = df["label"].to_numpy()
        fold[off:off + c] = (df["s1_id"].hash(SEED) % N_FOLDS).to_numpy()
        es_group[off:off + c] = (df["s1_id"].hash(SEED + 1) % 10).to_numpy()
        off += c
        del df
    print(f"stage-3: fitting on {n:,} pairs", flush=True)
    models = []
    for f in range(N_FOLDS):
        va = fold == f
        es = ~va & (es_group == f)
        trn = ~va & ~es
        m = lgb.train(PARAMS, lgb.Dataset(X[trn], y[trn], feature_name=FEATS3), N_ROUNDS,
                      valid_sets=[lgb.Dataset(X[es], y[es])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
        m.save_model(str(WORK_DIR / f"stage3_fold{f}.txt"))
        print(f"stage-3 fold {f}: best_iter={m.best_iteration}", flush=True)
    imp = sorted(zip(models[-1].feature_importance("gain"), FEATS3), reverse=True)
    print("stage-3 top features:", [(c, round(g / 1e3)) for g, c in imp[:12]])
    del X, y, fold, es_group
    _score("train", models, fit_expr)
    _score("test", models, None)
    print(f"stage-3 done in {time.time() - t:.0f}s", flush=True)


def _score(split: str, models, fit_expr):
    """Part by part: fitted rows get their out-of-fold model (the fold model that never saw
    their S1 entity), all other rows the fold-model average."""
    out_dir = WORK_DIR / f"{split}_scores3.parts"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    for f in _parts(split):
        df = pl.read_parquet(f, columns=["s1_id", "other_id", *FEATS3])
        X = df.select(FEATS3).cast(pl.Float32).to_numpy()
        pr = np.mean([m.predict(X, num_threads=N_THREADS) for m in models], axis=0)
        if fit_expr is not None:
            fit = df.select(fit_expr).to_series().to_numpy()
            fold = (df["s1_id"].hash(SEED) % N_FOLDS).to_numpy()
            for k, m in enumerate(models):
                sel = fit & (fold == k)
                if sel.any():
                    pr[sel] = m.predict(X[sel], num_threads=N_THREADS)
        (df.select("s1_id", "other_id").with_columns(p=pl.Series(pr, dtype=pl.Float32))
         .write_parquet(out_dir / f.name))
        del df, X
    pl.scan_parquet(out_dir / "*.parquet").sink_parquet(scores_path(split, 3))
    shutil.rmtree(out_dir)


def run(train_pct: int, test_pct: int, fit_pct: int):
    build_siblings("train", train_pct)
    build_siblings("test", test_pct)
    train_and_score(fit_pct)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sib-part", "run"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--test-pct", type=int, default=None)
    ap.add_argument("--fit-pct", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "sib-part":
        sib_part(a.split, a.pct, a.part, a.nparts)
    else:
        run(a.pct, a.test_pct or a.pct, a.fit_pct)
