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
Output: <work>/{train,test}_scores3.parquet.
"""
import argparse
import os
import subprocess
import sys
import time

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


def sib_dir(split):
    return WORK_DIR / f"{split}_sib"


# ---------------------------------------------------------------- sibling features
def sib_part(split: str, pct: int, i: int, n_parts: int):
    """Sibling features for the pairs of the S1 entities hashed to part i."""
    def in_part(col):
        return pl.col(col).hash(SEED) % n_parts == i

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
    out = pairs.join(agg, on=["s1_id", "other_id"], how="left").with_columns(
        pl.col("sib_n").fill_null(0), pl.col("sib_psum").fill_null(0.0))
    out.write_parquet(sib_dir(split) / f"part_{i:03d}.parquet")
    print(f"    sib part {i + 1}/{n_parts}: {pairs.height:,} pairs, {j.height:,} sibling rows",
          flush=True)


def build_siblings(split: str, pct: int):
    t = time.time()
    sc = pl.read_parquet(scores_path(split, 2))
    (sc.sort("p", descending=True).unique("other_id", keep="first")
     .filter(pl.col("p") >= MEMBER_P).write_parquet(members_path(split)))
    n_parts = max(1, -(-sc.height // PAIRS_PER_PART))
    del sc
    d = sib_dir(split)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        f.unlink()
    for i in range(n_parts):  # own process per part (memory is returned to the OS)
        rc = subprocess.call([sys.executable, "-u", __file__, "sib-part", "--split", split,
                              "--pct", str(pct), "--part", str(i), "--nparts", str(n_parts)])
        if rc:
            raise SystemExit(f"sib-part {i} failed ({rc})")
    print(f"{split}: sibling features in {n_parts} parts, {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- stage-3 model
def frame(split: str) -> pl.DataFrame:
    """Stage-2 score, its context among the record's candidates, and the sibling features."""
    sc = pl.read_parquet(scores_path(split, 2))
    sib = pl.concat([pl.read_parquet(f) for f in sorted(sib_dir(split).glob("part_*.parquet"))])
    p = pl.col("p").clip(1e-6, 1 - 1e-6)
    df = sc.with_columns(
        p_logit=(p / (1 - p)).log(),
        p_gap_o=pl.col("p").max().over("other_id") - pl.col("p"),
        p_rank_o=pl.col("p").rank("ordinal", descending=True).over("other_id").cast(pl.Int16),
        p_second_o=pl.col("p").sort(descending=True).slice(1, 1).first().over("other_id"),
        ncand_o=pl.len().over("other_id").cast(pl.Int16),
        p_sum_s=pl.col("p").sum().over("s1_id"),
        ncand_s=pl.len().over("s1_id").cast(pl.Int16),
    )
    return df.join(sib, on=["s1_id", "other_id"], how="left")


FEATS3 = ["p", "p_logit", "p_gap_o", "p_rank_o", "p_second_o", "ncand_o", "p_sum_s", "ncand_s",
          "sib_n", "sib_psum", "sib_name_max", "sib_name_mean", "sib_addr_max", "sib_addr_mean",
          "sib_street_max", "sib_house_agree", "sib_house_any", "sib_house_n", "sib_legal_agree"]


def train_and_score(fit_pct: int):
    t = time.time()
    tr = frame("train")
    gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet").with_columns(label=pl.lit(1, pl.Int8))
    tr = (tr.join(gt, left_on=["s1_id", "other_id"], right_on=["s1", "other"], how="left")
          .with_columns(pl.col("label").fill_null(0)))
    fit = tr.select(fit_rows(fit_pct) if fit_pct < 100 else pl.lit(True)).to_series().to_numpy()
    X = tr.select(FEATS3).cast(pl.Float32).to_numpy()
    y = tr["label"].to_numpy()
    fold = (tr["s1_id"].hash(SEED) % N_FOLDS).to_numpy()
    es_group = (tr["s1_id"].hash(SEED + 1) % 10).to_numpy()
    p3 = np.zeros(len(y), dtype=np.float32)
    held = ~fit
    models = []
    for f in range(N_FOLDS):
        va = fit & (fold == f)
        es = fit & ~va & (es_group == f)
        trn = fit & ~va & ~es
        m = lgb.train(PARAMS, lgb.Dataset(X[trn], y[trn], feature_name=FEATS3), N_ROUNDS,
                      valid_sets=[lgb.Dataset(X[es], y[es])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p3[va] = m.predict(X[va], num_threads=N_THREADS)
        models.append(m)
        m.save_model(str(WORK_DIR / f"stage3_fold{f}.txt"))
        print(f"stage-3 fold {f}: best_iter={m.best_iteration}", flush=True)
    if held.any():
        p3[held] = np.mean([m.predict(X[held], num_threads=N_THREADS) for m in models], axis=0)
    imp = sorted(zip(models[-1].feature_importance("gain"), FEATS3), reverse=True)
    print("stage-3 top features:", [(c, round(g / 1e3)) for g, c in imp[:12]])
    tr.select("s1_id", "other_id").with_columns(p=pl.Series(p3, dtype=pl.Float32)).write_parquet(
        scores_path("train", 3))
    del tr, X

    te = frame("test")
    Xt = te.select(FEATS3).cast(pl.Float32).to_numpy()
    pt = np.mean([m.predict(Xt, num_threads=N_THREADS) for m in models], axis=0)
    te.select("s1_id", "other_id").with_columns(p=pl.Series(pt, dtype=pl.Float32)).write_parquet(
        scores_path("test", 3))
    print(f"stage-3 done in {time.time() - t:.0f}s", flush=True)


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
