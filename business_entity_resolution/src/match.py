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
import subprocess
import sys
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


PAIRS_PER_PART = int(os.environ.get("ER_FEATURE_PART", 400_000))  # memory knob


def feature_dir(split: str, pct: int):
    return WORK_DIR / f"{split}_features{'' if pct >= 100 else f'_p{pct}'}"


def _parts(split: str, pct: int) -> list:
    d = feature_dir(split, pct)
    return sorted(d.glob("part_*.parquet")) if (d / "_DONE").exists() else []


def _features_fresh(split: str, pct: int) -> bool:
    d = feature_dir(split, pct)
    return (d / "_DONE").exists() and         (d / "_DONE").stat().st_mtime >= candidates_path(split, pct).stat().st_mtime


def build(split: str, pct: int) -> None:
    """Features in parts of ~PAIRS_PER_PART pairs, so peak memory stays a few GB even for the
    ~13M test pairs: pair features chunk by chunk (only that chunk's S2/S3 rows are loaded), then
    context features over all pairs from a slim numeric table, then joined back part by part.
    Output: <work>/<split>_features[_pN]/part_*.parquet (+ _DONE marker)."""
    t = time.time()
    d = feature_dir(split, pct)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        f.unlink()
    cp = candidates_path(split, pct)
    n_parts = max(1, -(-pl.scan_parquet(cp).select(pl.len()).collect().item() // PAIRS_PER_PART))
    # Every part runs in its own short-lived process: the libraries do not hand freed memory
    # back to the OS within one process, so a long loop grows by ~1 GB per part.
    for i in range(n_parts):
        _child("feature-part", split, pct, i, n_parts)
    ctx = features.context_features(pl.scan_parquet(sorted(d.glob("ctxbase_*.parquet"))))
    ctx.write_parquet(d / "ctx.parquet")
    del ctx
    for i in range(n_parts):
        _child("ctx-join", split, pct, i, n_parts)
    for f in [*d.glob("ctxbase_*.parquet"), d / "ctx.parquet"]:
        f.unlink()
    (d / "_DONE").touch()
    n_cols = len(pl.read_parquet_schema(d / "part_000.parquet"))
    print(f"{split}: {n_parts} parts, {n_cols} columns in {time.time() - t:.0f}s")


def _child(cmd: str, split: str, pct: int, i: int, n_parts: int):
    rc = subprocess.call([sys.executable, "-u", __file__, cmd, "--split", split, "--pct", str(pct),
                          "--part", str(i), "--nparts", str(n_parts)])
    if rc:
        raise SystemExit(f"{cmd} part {i} failed (exit code {rc})")


def feature_part(split: str, pct: int, i: int, n_parts: int):
    """Pair features for part i (the S2/S3 records whose id hashes to i), plus its slim
    context-base table. Each input is streamed with a plain row filter, so only this part's
    rows are ever in memory."""
    d = feature_dir(split, pct)

    def in_part(col: str) -> pl.Expr:
        return pl.col(col).hash(SEED) % n_parts == i

    s1 = load_norm(split, 1, 100, columns=features.SIDE_COLS)  # 1.7-2.2M rows x 7 short columns
    idf = features.token_idf(s1["name_core"]), features.token_idf(s1["addr_norm"])
    c = pl.scan_parquet(candidates_path(split, pct)).filter(in_part("other_id")).collect(
        engine="streaming")
    oth = pl.concat([load_norm(split, s, pct, columns=features.SIDE_COLS, lazy=True)
                     .filter(in_part("entity_id")).collect(engine="streaming") for s in (2, 3)])
    df = features.pair_features(c, s1, oth, *idf)
    del oth, c, s1
    if split == "train":
        gt = (pl.scan_parquet(WORK_DIR / "train_gt_pairs.parquet").filter(in_part("other"))
              .with_columns(label=pl.lit(1, pl.Int8)).collect(engine="streaming"))
        df = (df.join(gt, left_on=["s1_id", "other_id"], right_on=["s1", "other"], how="left")
              .with_columns(pl.col("label").fill_null(0)))
    df.write_parquet(d / f"part_{i:03d}.parquet")
    features.context_base(df).with_columns(_p=pl.lit(i, pl.Int32)).write_parquet(
        d / f"ctxbase_{i:03d}.parquet")
    print(f"    part {i + 1}/{n_parts}: {df.height:,} pairs", flush=True)


def ctx_join(split: str, pct: int, i: int):
    d = feature_dir(split, pct)
    ctx = (pl.scan_parquet(d / "ctx.parquet").filter(pl.col("_p") == i).drop("_p")
           .collect(engine="streaming"))
    part = pl.read_parquet(d / f"part_{i:03d}.parquet")
    part = features.with_keys(part).join(ctx, on=features.KEYS, how="left").drop(features.KEYS)
    part.write_parquet(d / f"part_{i:03d}.parquet")


def load_features(split: str, pct: int, columns=None) -> pl.DataFrame:
    """All feature parts in one frame (train: a few M pairs). Rebuilt if the candidates are newer."""
    if not _features_fresh(split, pct):
        build(split, pct)
    return pl.concat([pl.read_parquet(f, columns=columns) for f in _parts(split, pct)])


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
    """Scores the test feature parts one at a time (the full test feature matrix never exists)."""
    if not _features_fresh("test", pct):
        build("test", pct)
    with open(FEATURES_PATH) as fh:
        cols = json.load(fh)  # exactly the features the fold models were trained on
    models = [lgb.Booster(model_file=str(WORK_DIR / f"stage2_fold{f}.txt")) for f in range(N_FOLDS)]
    scored = []
    for f in _parts("test", pct):
        df = pl.read_parquet(f, columns=["s1_id", "other_id", *cols])
        X = df.select(cols).cast(pl.Float32).to_numpy()
        p = np.mean([m.predict(X, num_threads=N_THREADS) for m in models], axis=0)
        scored.append(df.select("s1_id", "other_id").with_columns(p=pl.Series(p, dtype=pl.Float32)))
        del df, X
    pairs = pl.concat(scored)
    with open(WORK_DIR / "decision.json") as fh:
        dec = json.load(fh)
    pairs.write_parquet(WORK_DIR / "test_scores.parquet")
    print("decision:", dec)
    write_outputs(apply_rule(pairs, dec), pairs.select("s1_id", "other_id"))


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
    ap.add_argument("cmd", choices=["features", "train", "predict", "feature-part", "ctx-join"])
    ap.add_argument("--part", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--nparts", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--test-pct", type=int, default=None, help="train only: %% test will use")
    a = ap.parse_args()
    if a.cmd == "features":
        build(a.split, a.pct)
    elif a.cmd == "feature-part":
        feature_part(a.split, a.pct, a.part, a.nparts)
    elif a.cmd == "ctx-join":
        ctx_join(a.split, a.pct, a.part)
    elif a.cmd == "train":
        train(a.pct, density_matched=(a.test_pct or a.pct) == a.pct)
    else:
        predict(a.pct)
