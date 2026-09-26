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
At prediction time two guards follow (see finalize): at most MAX_PER_S1 records per S1 entity
(the training maximum is 11), and a stricter probability floor for countries absent from
training (France), where there are no labels to calibrate against.

decide:  re-applies the decision + guards to saved test scores (seconds, no re-scoring), e.g.
         python match.py decide --fr-delta 0.2
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
              seed=SEED, verbose=-1, num_threads=N_THREADS,
              # identical models on every re-run (same data + same ER_THREADS)
              deterministic=True, force_row_wise=True)
N_ROUNDS = int(os.environ.get("ER_ROUNDS", 2000))  # early stopping decides; 600 was always hit
FIT_SEED = SEED + 7  # row sample used for fitting (independent of the blocking/stage-1 buckets)
MAX_PER_S1 = int(os.environ.get("ER_MAX_PER_S1", 11))
UNSEEN_COUNTRIES = ("France",)  # in test only
FR_DELTA = float(os.environ.get("ER_FR_DELTA", 0.1))  # added to the threshold for those countries


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


def load_features(split: str, pct: int, columns=None, row_filter: pl.Expr | None = None):
    """Feature parts in one frame, optionally only rows matching row_filter (streamed per part).
    Rebuilt first if the candidates are newer than the cached features."""
    if not _features_fresh(split, pct):
        build(split, pct)
    out = []
    for f in _parts(split, pct):
        lf = pl.scan_parquet(f)
        if row_filter is not None:
            lf = lf.filter(row_filter)
        out.append((lf.select(columns) if columns else lf).collect(engine="streaming"))
    return pl.concat(out)


def fit_rows(fit_pct: int) -> pl.Expr:
    """S2/S3 records whose pairs are used to fit the stage-2 model (the rest are held out)."""
    return pl.col("other_id").hash(FIT_SEED) % 100 < fit_pct


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


def _hashed(pairs: pl.DataFrame) -> pl.DataFrame:
    """Same pairs with 64-bit hashed ids: sorts/joins/windows over ~13M pairs cost far less."""
    return pairs.select(s1_id=pl.col("s1_id").hash(), other_id=pl.col("other_id").hash(), p="p")


def apply_rule_ids(pairs: pl.DataFrame, dec: dict) -> pl.DataFrame:
    """apply_rule on hashed ids, mapped back to the original string ids of the kept pairs."""
    kept = apply_rule(_hashed(pairs), dec)
    ids = pairs.select(_s=pl.col("s1_id").hash(), _o=pl.col("other_id").hash(),
                       s1_str="s1_id", other_str="other_id")
    return kept.join(ids, left_on=["s1", "other"], right_on=["_s", "_o"]).select(
        s1="s1_str", other="other_str")


def finalize(pairs: pl.DataFrame, dec: dict, fr_delta: float = FR_DELTA,
             max_per_s1: int = MAX_PER_S1) -> pl.DataFrame:
    """apply_rule, then the prediction-time guards: for S1 entities in UNSEEN_COUNTRIES keep only
    records with p >= threshold + fr_delta (threshold 0.5 under the entity rule), and keep at
    most max_per_s1 records (highest p) per S1 entity."""
    m = apply_rule_ids(pairs, dec).join(
        pairs.select(pl.col("s1_id").alias("s1"), pl.col("other_id").alias("other"), "p"),
        on=["s1", "other"])
    unseen = (load_norm("test", 1, 100, columns=["entity_id", "country"])
              .filter(pl.col("country").is_in(UNSEEN_COUNTRIES))["entity_id"])
    floor = dec.get("threshold", 0.5) + fr_delta
    n0 = m.height
    m = m.filter(~pl.col("s1").is_in(unseen.implode()) | (pl.col("p") >= floor))
    n1 = m.height
    m = m.filter(pl.col("p").rank("ordinal", descending=True).over("s1") <= max_per_s1)
    print(f"guards: {n0 - n1:,} unseen-country pairs below p={floor:.2f} dropped, "
          f"{n1 - m.height:,} beyond {max_per_s1} per S1 dropped")
    return m.select("s1", "other")


def train(pct: int, density_matched: bool = True, fit_pct: int = 100):
    """density_matched=False when train uses a smaller %% of Source 2/3 than test (--train-pct):
    per-S1 aggregates then differ between train and test, so the S1-context features and the
    per-entity decision rule are left out.
    fit_pct < 100: fit only on that share of the S2/S3 records (memory); the rest is scored as a
    genuine hold-out by `calibrate`, giving a full-density validation."""
    df = load_features("train", pct, row_filter=fit_rows(fit_pct) if fit_pct < 100 else None)
    print(f"fitting on {df.height:,} pairs (fit_pct={fit_pct})", flush=True)
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
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], num_threads=N_THREADS)
        m.save_model(str(WORK_DIR / f"stage2_fold{f}.txt"))
        print(f"fold {f}: best_iter={m.best_iteration} ({time.time() - t:.0f}s)")
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)
    print("top features:", [(c, round(g / 1e3)) for g, c in imp[:25]])

    pairs = df.select("s1_id", "other_id").with_columns(p=pl.Series(oof))
    pairs.write_parquet(WORK_DIR / f"train_oof{'' if pct >= 100 else f'_p{pct}'}.parquet")


def calibrate(pct: int, density_matched: bool = True, fit_pct: int = 100):
    """Choose the decision rule on held-out probabilities: out-of-fold for the fitted rows, and
    the fold-model average for rows never used in fitting (fit_pct < 100). With pct=100 this is
    a full-density validation over every training S1 entity - the closest match to the test."""
    pairs = pl.read_parquet(WORK_DIR / f"train_oof{'' if pct >= 100 else f'_p{pct}'}.parquet")
    if fit_pct < 100:
        with open(FEATURES_PATH) as fh:
            cols = json.load(fh)
        models = [lgb.Booster(model_file=str(WORK_DIR / f"stage2_fold{f}.txt"))
                  for f in range(N_FOLDS)]
        held = [pairs.with_columns(pl.col("p").cast(pl.Float32))]
        for f in _parts("train", pct):
            df = (pl.scan_parquet(f).filter(~fit_rows(fit_pct))
                  .select("s1_id", "other_id", *cols).collect(engine="streaming"))
            X = df.select(cols).cast(pl.Float32).to_numpy()
            p = np.mean([m.predict(X, num_threads=N_THREADS) for m in models], axis=0)
            held.append(df.select("s1_id", "other_id").with_columns(p=pl.Series(p, dtype=pl.Float32)))
            del df, X
        pairs = pl.concat(held)
        print(f"calibrating on {pairs.height:,} held-out pairs", flush=True)
    best = evaluate(pairs, pct, rules=("threshold", "entity") if density_matched else ("threshold",))
    with open(WORK_DIR / "decision.json", "w") as fh:
        json.dump(best, fh)


def evaluate(pairs: pl.DataFrame, pct: int, rules=("threshold", "entity")) -> dict:
    """Grid-search both decision rules on OOF probabilities; returns the best as a decision dict."""
    gt = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    s1_all = load_norm("train", 1, 100, columns=["entity_id"])["entity_id"]
    pairs = pairs.with_columns(pl.col("p").cast(pl.Float32))
    if pct < 100:
        # sampled S2/S3: score only entities that have a sampled true match or any candidate,
        # otherwise the ~all-empty remainder inflates the macro average.
        oth = pl.concat([load_norm("train", s, pct, columns=["entity_id"])
                         for s in (2, 3)])
        gt = gt.join(oth, left_on="other", right_on="entity_id", how="semi")
        s1_all = pl.concat([gt["s1"], pairs["s1_id"]]).unique()
    # hashed ids from here on: ~13M pairs x ~50 settings stays light
    s1_all = s1_all.hash()
    gt = gt.select(s1=pl.col("s1").hash(), other=pl.col("other").hash())
    pairs = _hashed(pairs)
    grids = {"threshold": np.round(np.arange(0.20, 0.91, 0.02), 2),
             "entity": np.round(np.arange(-2.0, 2.01, 0.25), 2)}
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
    write_outputs(finalize(pairs, dec), pairs.select("s1_id", "other_id"))


def redecide(fr_delta: float, max_per_s1: int):
    """Re-apply the decision to saved test scores with other guard settings."""
    pairs = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    with open(WORK_DIR / "decision.json") as fh:
        dec = json.load(fh)
    print("decision:", dec, f"fr_delta={fr_delta} max_per_s1={max_per_s1}")
    write_outputs(finalize(pairs, dec, fr_delta, max_per_s1), pairs.select("s1_id", "other_id"))


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
    ap.add_argument("cmd", choices=["features", "train", "predict", "decide", "feature-part",
                                    "ctx-join"])
    ap.add_argument("--part", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--nparts", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--test-pct", type=int, default=None, help="train only: %% test will use")
    ap.add_argument("--fit-pct", type=int, default=100)
    ap.add_argument("--fr-delta", type=float, default=FR_DELTA,
                    help="decide: extra probability required for France matches")
    ap.add_argument("--max-per-s1", type=int, default=MAX_PER_S1, help="decide: cap per S1 entity")
    a = ap.parse_args()
    if a.cmd == "features":
        build(a.split, a.pct)
    elif a.cmd == "feature-part":
        feature_part(a.split, a.pct, a.part, a.nparts)
    elif a.cmd == "ctx-join":
        ctx_join(a.split, a.pct, a.part)
    elif a.cmd == "decide":
        redecide(a.fr_delta, a.max_per_s1)
    elif a.cmd == "train":
        train(a.pct, density_matched=(a.test_pct or a.pct) == a.pct, fit_pct=a.fit_pct)
    else:
        predict(a.pct)
