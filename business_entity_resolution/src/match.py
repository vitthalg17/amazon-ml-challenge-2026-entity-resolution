"""Stage-2 matching model: features -> LightGBM -> one-S1-per-record assignment -> threshold.

train:   builds features on the pruned train candidates, trains K models on S1-entity folds,
         collects out-of-fold probabilities, then picks the threshold that maximizes macro F0.5.
predict: scores the test candidates with the fold-model average, applies the same decision
         rule, and writes output/matching_results.tsv + output/candidate_pairs.tsv.

Decision rule: every S2/S3 record belongs to at most one S1 (true in the training labels),
so each record is assigned only to its highest-probability S1, and only if p >= threshold.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import features
from blocking import candidates_path
from config import OUTPUT_DIR, SEED, WORK_DIR
from metrics import macro_f05
from prep import load_norm

N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 4))
N_FOLDS = int(os.environ.get("ER_FOLDS", 4))
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=N_THREADS)
N_ROUNDS = int(os.environ.get("ER_ROUNDS", 600))


def feature_path(split: str, pct: int):
    return WORK_DIR / f"{split}_features{'' if pct >= 100 else f'_p{pct}'}.parquet"


def build(split: str, pct: int) -> pl.DataFrame:
    t = time.time()
    cand = pl.read_parquet(candidates_path(split, pct))
    s1 = load_norm(split, 1, 100).join(cand.select(pl.col("s1_id").unique()),
                                                         left_on="entity_id", right_on="s1_id",
                                                         how="semi")
    oth = pl.concat([load_norm(split, s, pct) for s in (2, 3)])
    oth = oth.join(cand.select(pl.col("other_id").unique()), left_on="entity_id",
                   right_on="other_id", how="semi")
    df = features.build_features(cand, s1, oth)
    if split == "train":
        gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
        df = (df.join(gt.with_columns(label=pl.lit(1, pl.Int8)), left_on=["s1_id", "other_id"],
                      right_on=["s1", "other"], how="left")
              .with_columns(pl.col("label").fill_null(0)))
    df.write_parquet(feature_path(split, pct))
    print(f"{split}: {df.height:,} pairs x {len(features.FEATURES)} features in {time.time() - t:.0f}s")
    return df


def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("other_id", "s1_id", "country", "label")]


def decide(pairs: pl.DataFrame, t: float) -> pl.DataFrame:
    """Assign each S2/S3 record to its best S1 if p >= t. Returns (s1, other) pairs."""
    return (pairs.filter(pl.col("p") >= t)
            .sort("p", descending=True)
            .unique("other_id", keep="first")
            .select(pl.col("s1_id").alias("s1"), pl.col("other_id").alias("other")))


def train(pct: int):
    df = pl.read_parquet(feature_path("train", pct)) if feature_path("train", pct).exists() \
        else build("train", pct)
    cols = feature_cols(df)
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["label"].to_numpy()
    fold = (df["s1_id"].hash(SEED) % N_FOLDS).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(N_FOLDS):
        t = time.time()
        tr, va = fold != f, fold == f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=cols), N_ROUNDS,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_threads=N_THREADS)
        m.save_model(str(WORK_DIR / f"stage2_fold{f}.txt"))
        print(f"fold {f}: best_iter={m.best_iteration} ({time.time() - t:.0f}s)")
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)
    print("top features:", [(c, round(g / 1e3)) for g, c in imp[:25]])

    pairs = df.select("s1_id", "other_id").with_columns(p=pl.Series(oof))
    pairs.write_parquet(WORK_DIR / f"train_oof{'' if pct >= 100 else f'_p{pct}'}.parquet")
    best = evaluate(pairs, pct)
    with open(WORK_DIR / "decision.json", "w") as fh:
        json.dump({"threshold": best}, fh)


def evaluate(pairs: pl.DataFrame, pct: int) -> float:
    gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    s1_all = load_norm("train", 1, 100, columns=["entity_id"])["entity_id"]
    if pct < 100:
        # sampled S2/S3: score only entities that have a sampled true match or any candidate,
        # otherwise the ~all-empty remainder inflates the macro average.
        oth = pl.concat([load_norm("train", s, pct, columns=["entity_id"])
                         for s in (2, 3)])
        gt = gt.join(oth, left_on="other", right_on="entity_id", how="semi")
        s1_all = pl.concat([gt["s1"], pairs["s1_id"]]).unique()
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        r = macro_f05(decide(pairs, t), gt, s1_all)
        rows.append({"t": t, **r})
    res = pl.DataFrame(rows)
    print(res.select("t", "macro_f05", "singleton_acc", "nonsingleton_f05", "pair_precision",
                     "pair_recall"))
    best = res.sort("macro_f05", descending=True).row(0, named=True)
    print(f"best threshold {best['t']}: macro F0.5 = {best['macro_f05']:.4f} "
          f"(entities: {best['n_entities']:,}, singletons: {best['n_singletons']:,})")
    return float(best["t"])


def predict(pct: int = 100):
    df = pl.read_parquet(feature_path("test", pct)) if feature_path("test", pct).exists() \
        else build("test", pct)
    cols = feature_cols(df)
    X = df.select(cols).to_numpy().astype(np.float32)
    p = np.mean([lgb.Booster(model_file=str(WORK_DIR / f"stage2_fold{f}.txt"))
                 .predict(X, num_threads=N_THREADS) for f in range(N_FOLDS)], axis=0)
    t = json.load(open(WORK_DIR / "decision.json"))["threshold"]
    pairs = df.select("s1_id", "other_id").with_columns(p=pl.Series(p, dtype=pl.Float32))
    pairs.write_parquet(WORK_DIR / "test_scores.parquet")
    write_outputs(decide(pairs, t), df.select("s1_id", "other_id"))


def _write_lists(pairs: pl.DataFrame, s1_ids: pl.Series, col: str, path):
    lists = pairs.group_by("s1").agg(pl.col("other").unique().sort().str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
           .join(lists, left_on="source1_entity_id", right_on="s1", how="left")
           .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"wrote {path} ({out.height:,} rows, {(out[col] != '').sum():,} non-empty)")


def write_outputs(matches: pl.DataFrame, cands: pl.DataFrame):
    s1_ids = load_norm("test", 1, 100, columns=["entity_id"])["entity_id"]
    _write_lists(matches, s1_ids, "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    _write_lists(cands.rename({"s1_id": "s1", "other_id": "other"}), s1_ids,
                 "candidate_entity_ids", OUTPUT_DIR / "candidate_pairs.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "train", "predict"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "features":
        build(a.split, a.pct)
    elif a.cmd == "train":
        train(a.pct)
    else:
        predict(a.pct)
