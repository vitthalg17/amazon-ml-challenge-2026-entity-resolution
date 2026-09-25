"""Stage-1 candidate pruner (part of blocking).

A small LightGBM over retrieval signals only (channel cosines, ranks, gap to the record's best
candidate). It is trained on raw (unpruned) candidates of a training sample, then applied
inside blocking: per S2/S3 record keep its top PRUNE_TOP candidates with p >= PRUNE_PMIN.
The pruned set is what the matching model scores, i.e. candidate_pairs.tsv.
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, WORK_DIR

MODEL_PATH = WORK_DIR / "stage1.txt"
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
    return lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None


def prune(cand: pl.DataFrame, model, channels, rank_cols, key: str = "oi") -> pl.DataFrame:
    df = feature_frame(cand, channels, rank_cols, key)
    X = df.select(feature_names(channels, rank_cols)).to_numpy().astype(np.float32)
    df = df.with_columns(p1=pl.Series(model.predict(X, num_threads=N_THREADS), dtype=pl.Float32))
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
    X = df.select(feature_names(CHANNELS, RANK_COLS)).to_numpy().astype(np.float32)
    y = df["label"].to_numpy()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, seed=SEED, verbose=-1, num_threads=N_THREADS)
    model = lgb.train(params, lgb.Dataset(X, y), 200)
    model.save_model(str(MODEL_PATH))
    print(f"stage-1 trained on {len(y):,} raw candidate pairs ({y.sum():,} positive) -> {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=5, help="train sample the raw candidates came from")
    train(ap.parse_args().pct)
