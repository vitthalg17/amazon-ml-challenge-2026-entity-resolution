"""Stage-1 candidate pruner (part of blocking).

A small LightGBM over retrieval signals only (channel cosines, ranks, gap to the record's best
candidate). It is trained on raw (unpruned) candidates of a training sample, then applied
inside blocking: per S2/S3 record keep its top PRUNE_TOP candidates with p >= PRUNE_PMIN.
The pruned set is what the matching model scores, i.e. candidate_pairs.tsv.

Cross-fit: the raw sample is split in two hash halves and a model is trained on each. Train
records in the first half are pruned by the second model and all other records by the first, so
no training record's p1 (also a stage-2 feature) comes from a model that saw its label.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, WORK_DIR

MODEL_PATH = WORK_DIR / "stage1.txt"      # trained on hash buckets [0, half)
ALT_PATH = WORK_DIR / "stage1_alt.txt"    # trained on hash buckets [half, pct)
META_PATH = WORK_DIR / "stage1.json"
PRUNE_TOP = int(os.environ.get("ER_PRUNE_TOP", 10))
PRUNE_PMIN = float(os.environ.get("ER_PRUNE_PMIN", 0.0005))
N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))


def feature_frame(cand: pl.DataFrame, channels, rank_cols, key: str = "oi") -> pl.DataFrame:
    cos = [f"cos_{c}" for c in channels]
    df = cand.with_columns(cos_sum=pl.sum_horizontal(cos), ncand=pl.len().over(key))
    return df.with_columns(
        *[(pl.col(c).max().over(key) - pl.col(c)).alias(f"{c}_gap")
          for c in cos + ["cos_sum"]])


def feature_names(channels, rank_cols) -> list[str]:
    cos = [f"cos_{c}" for c in channels]
    return cos + list(rank_cols) + [f"{c}_gap" for c in cos + ["cos_sum"]] + ["cos_sum", "ncand"]


def load_model():
    if not MODEL_PATH.exists():
        return None
    half = json.loads(META_PATH.read_text())["half"] if META_PATH.exists() else 0
    alt = lgb.Booster(model_file=str(ALT_PATH)) if half and ALT_PATH.exists() else None
    return {"main": lgb.Booster(model_file=str(MODEL_PATH)), "alt": alt, "half": half}


def prune(cand: pl.DataFrame, pruner: dict, channels, rank_cols, key: str = "oi",
          in_main_sample: np.ndarray | None = None, n_threads: int = N_THREADS) -> pl.DataFrame:
    """in_main_sample: per-row mask of records the main model was trained on (scored by alt)."""
    df = feature_frame(cand, channels, rank_cols, key)
    X = df.select(feature_names(channels, rank_cols)).cast(pl.Float32).to_numpy()
    p = pruner["main"].predict(X, num_threads=n_threads)
    if pruner["alt"] is not None and in_main_sample is not None and in_main_sample.any():
        p[in_main_sample] = pruner["alt"].predict(X[in_main_sample], num_threads=n_threads)
    df = df.with_columns(p1=pl.Series(p, dtype=pl.Float32))
    df = df.filter((pl.col("p1") >= PRUNE_PMIN)
                   & (pl.col("p1").rank("ordinal", descending=True).over(key) <= PRUNE_TOP))
    return df.select(cand.columns + ["p1"])


def train(pct: int):
    from blocking import CHANNELS, RANK_COLS, candidates_path

    cand = pl.read_parquet(candidates_path("train", pct, raw=True))
    gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    cand = (cand.join(gt.with_columns(label=pl.lit(1, pl.Int8)),
                      left_on=["s1_id", "other_id"], right_on=["s1", "other"], how="left")
            .with_columns(pl.col("label").fill_null(0)))
    df = feature_frame(cand, CHANNELS, RANK_COLS, key="other_id")
    X = df.select(feature_names(CHANNELS, RANK_COLS)).cast(pl.Float32).to_numpy()
    y = df["label"].to_numpy()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, seed=SEED, verbose=-1, num_threads=N_THREADS)
    half = pct // 2
    first = (df["other_id"].hash(SEED) % 100 < half).to_numpy()
    jobs = [(MODEL_PATH, first), (ALT_PATH, ~first)]
    if not (half and first.any() and (~first).any()):
        print("stage-1: sample too small to cross-fit, training a single model")
        half, jobs = 0, [(MODEL_PATH, np.ones(len(y), bool))]
        ALT_PATH.unlink(missing_ok=True)
    for path, rows in jobs:
        lgb.train(params, lgb.Dataset(X[rows], y[rows]), 200).save_model(str(path))
        print(f"stage-1 trained on {rows.sum():,} raw candidate pairs ({y[rows].sum():,} positive)"
              f" -> {path}")
    META_PATH.write_text(json.dumps({"half": half, "pct": pct}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=6, help="train sample the raw candidates came from")
    train(ap.parse_args().pct)
