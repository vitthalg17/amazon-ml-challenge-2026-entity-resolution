"""Stage-2 matching model: features -> LightGBM -> one-S1-per-record assignment -> decision rule.

train:   builds features on the pruned train candidates, trains K models on S1-entity folds,
         collects out-of-fold probabilities, then picks the decision rule and its parameter that
         maximize macro F0.5 on them.
predict: scores the test candidates with the fold-model average, applies the same decision
         rule, and writes output/matching_results.tsv + output/candidate_pairs.tsv.

Every S2/S3 record belongs to at most one S1 (true in the training labels), so each record is
assigned only to its highest-probability S1. Two rules then decide what to keep:
  threshold  keep the assignment if p >= t (one global cut)
  entity     per S1 entity, keep the top-k assigned records (k may be 0) that maximize the
             entity's expected F0.5, i.e. the metric itself (decision-theoretic F-measure
             optimisation, Ye et al. ICML 2012); `shift` recalibrates p in logit space
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


def load_features(split: str, pct: int) -> pl.DataFrame:
    """Cached features, rebuilt when the candidates were regenerated after the cache was written."""
    fp, cp = feature_path(split, pct), candidates_path(split, pct)
    if fp.exists() and fp.stat().st_mtime >= cp.stat().st_mtime:
        return pl.read_parquet(fp)
    return build(split, pct)


def build(split: str, pct: int) -> pl.DataFrame:
    t = time.time()
    cand = pl.read_parquet(candidates_path(split, pct))
    s1_all = load_norm(split, 1, 100)
    idf = features.token_idf(s1_all["name_core"]), features.token_idf(s1_all["addr_norm"])
    s1 = s1_all.join(cand.select(pl.col("s1_id").unique()), left_on="entity_id",
                     right_on="s1_id", how="semi")
    del s1_all
    # read only the S2/S3 rows that are candidates (lazy semi-join: never the full sources)
    ids = cand.lazy().select(pl.col("other_id").unique())
    oth = pl.concat([load_norm(split, s, pct, lazy=True)
                     .join(ids, left_on="entity_id", right_on="other_id", how="semi").collect()
                     for s in (2, 3)])
    df = features.build_features(cand, s1, oth, *idf)
    if split == "train":
        gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
        df = (df.join(gt.with_columns(label=pl.lit(1, pl.Int8)), left_on=["s1_id", "other_id"],
                      right_on=["s1", "other"], how="left")
              .with_columns(pl.col("label").fill_null(0)))
    df.write_parquet(feature_path(split, pct))
    print(f"{split}: {df.height:,} pairs x {len(features.FEATURES)} features in {time.time() - t:.0f}s")
    return df


FEATURES_PATH = WORK_DIR / "stage2_features.json"


def is_s1_context(c: str) -> bool:
    """Features that aggregate over all S2/S3 candidates of one S1 entity. Their scale depends on
    what fraction of S2/S3 was blocked, so they only transfer if train and test use the same %."""
    return c.endswith("_gap_s") or c in ("rank_s", "ncand_s")


def feature_cols(df: pl.DataFrame, density_matched: bool = True) -> list[str]:
    cols = [c for c in df.columns if c not in ("other_id", "s1_id", "country", "label")]
    return [c for c in cols if density_matched or not is_s1_context(c)]


def decide(pairs: pl.DataFrame, t: float) -> pl.DataFrame:
    """Assign each S2/S3 record to its best S1 if p >= t. Returns (s1, other) pairs."""
    return (pairs.filter(pl.col("p") >= t)
            .sort("p", descending=True)
            .unique("other_id", keep="first")
            .select(pl.col("s1_id").alias("s1"), pl.col("other_id").alias("other")))


def decide_entity(pairs: pl.DataFrame, shift: float) -> pl.DataFrame:
    """Per S1 entity keep the top-k assigned records maximizing expected F0.5 with
    q = sigmoid(logit(p) + shift):
      keep none: E[F] = P(entity is a singleton) = prod(1 - q) over all its candidates
      keep k:    E[F] ~= 1.25 * (sum of the k kept q) / (0.25 * sum of all its q + k)
    """
    lg = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log()
    q = pairs.with_columns(q=(1 / (1 + (-(lg + shift)).exp())).clip(1e-6, 1 - 1e-6))
    ent = q.group_by("s1_id").agg(q_all=pl.col("q").sum(),
                                  p_none=(1 - pl.col("q")).log().sum().exp())
    kept = (q.sort("q", descending=True).unique("other_id", keep="first")
            .sort(["s1_id", "q"], descending=[False, True])
            .join(ent, on="s1_id")
            .with_columns(k=pl.col("q").cum_count().over("s1_id"),
                          cum=pl.col("q").cum_sum().over("s1_id"))
            .with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("q_all") + pl.col("k")))
            .with_columns(k_best=pl.col("k").get(pl.col("ef").arg_max()).over("s1_id"),
                          ef_best=pl.col("ef").max().over("s1_id"))
            .filter((pl.col("ef_best") > pl.col("p_none")) & (pl.col("k") <= pl.col("k_best"))))
    return kept.select(pl.col("s1_id").alias("s1"), pl.col("other_id").alias("other"))


def apply_rule(pairs: pl.DataFrame, dec: dict) -> pl.DataFrame:
    if dec.get("rule") == "entity":
        return decide_entity(pairs, dec["shift"])
    return decide(pairs, dec["threshold"])


def train(pct: int, density_matched: bool = True):
    """density_matched=False when train uses a smaller %% of Source 2/3 than test (--train-pct):
    per-S1 aggregates then differ between train and test, so the S1-context features and the
    per-entity decision rule are left out."""
    df = load_features("train", pct)
    cols = feature_cols(df, density_matched)
    with open(FEATURES_PATH, "w") as fh:
        json.dump(cols, fh)
    X = df.select(cols).cast(pl.Float32).to_numpy()
    y = df["label"].to_numpy()
    fold = (df["s1_id"].hash(SEED) % N_FOLDS).to_numpy()
    # early stopping uses S1 entities held out of the training folds, not the OOF fold itself
    stop = (df["s1_id"].hash(SEED + 1) % 10 == 0).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(N_FOLDS):
        t = time.time()
        va = fold == f
        tr, es = ~va & ~stop, ~va & stop
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=cols), N_ROUNDS,
                      valid_sets=[lgb.Dataset(X[es], y[es])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_threads=N_THREADS)
        m.save_model(str(WORK_DIR / f"stage2_fold{f}.txt"))
        print(f"fold {f}: best_iter={m.best_iteration} ({time.time() - t:.0f}s)")
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)
    print("top features:", [(c, round(g / 1e3)) for g, c in imp[:25]])

    pairs = df.select("s1_id", "other_id").with_columns(p=pl.Series(oof))
    pairs.write_parquet(WORK_DIR / f"train_oof{'' if pct >= 100 else f'_p{pct}'}.parquet")
    best = evaluate(pairs, pct, rules=("threshold", "entity") if density_matched else ("threshold",))
    with open(WORK_DIR / "decision.json", "w") as fh:
        json.dump(best, fh)


def evaluate(pairs: pl.DataFrame, pct: int, rules=("threshold", "entity")) -> dict:
    """Grid-search both decision rules on OOF probabilities; returns the best as a decision dict."""
    gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    s1_all = load_norm("train", 1, 100, columns=["entity_id"])["entity_id"]
    if pct < 100:
        # sampled S2/S3: score only entities that have a sampled true match or any candidate,
        # otherwise the ~all-empty remainder inflates the macro average.
        oth = pl.concat([load_norm("train", s, pct, columns=["entity_id"])
                         for s in (2, 3)])
        gt = gt.join(oth, left_on="other", right_on="entity_id", how="semi")
        s1_all = pl.concat([gt["s1"], pairs["s1_id"]]).unique()
    grids = {"threshold": np.round(np.arange(0.02, 0.99, 0.01), 2),
             "entity": np.round(np.arange(-3.0, 3.01, 0.25), 2)}
    grids = {r: g for r, g in grids.items() if r in rules}
    rows = []
    for rule, grid in grids.items():
        key = "shift" if rule == "entity" else "threshold"
        for v in grid:
            dec = {"rule": rule, key: float(v)}
            rows.append({"rule": rule, "param": float(v),
                         **macro_f05(apply_rule(pairs, dec), gt, s1_all)})
    res = pl.DataFrame(rows)
    cols = ["param", "macro_f05", "singleton_acc", "nonsingleton_f05", "pair_precision",
            "pair_recall"]
    best = {}
    for rule in grids:
        b = res.filter(pl.col("rule") == rule).sort("macro_f05", descending=True)
        print(f"rule={rule}: top settings"); print(b.select(cols).head(5))
        best[rule] = b.row(0, named=True)
    win = max(best.values(), key=lambda r: r["macro_f05"])
    print("  ".join(f"{r}: {b['macro_f05']:.4f}" for r, b in best.items())
          + f"  -> using {win['rule']} (param {win['param']}; entities: {win['n_entities']:,}, "
            f"singletons: {win['n_singletons']:,})")
    return {"rule": win["rule"], "shift" if win["rule"] == "entity" else "threshold": win["param"]}


def predict(pct: int = 100):
    df = load_features("test", pct)
    with open(FEATURES_PATH) as fh:
        cols = json.load(fh)  # exactly the features the fold models were trained on
    X = df.select(cols).cast(pl.Float32).to_numpy()
    p = np.mean([lgb.Booster(model_file=str(WORK_DIR / f"stage2_fold{f}.txt"))
                 .predict(X, num_threads=N_THREADS) for f in range(N_FOLDS)], axis=0)
    with open(WORK_DIR / "decision.json") as fh:
        dec = json.load(fh)
    pairs = df.select("s1_id", "other_id").with_columns(p=pl.Series(p, dtype=pl.Float32))
    pairs.write_parquet(WORK_DIR / "test_scores.parquet")
    print("decision:", dec)
    write_outputs(apply_rule(pairs, dec), df.select("s1_id", "other_id"))


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
    ap.add_argument("--test-pct", type=int, default=None, help="train only: %% test will use")
    a = ap.parse_args()
    if a.cmd == "features":
        build(a.split, a.pct)
    elif a.cmd == "train":
        train(a.pct, density_matched=(a.test_pct or a.pct) == a.pct)
    else:
        predict(a.pct)
