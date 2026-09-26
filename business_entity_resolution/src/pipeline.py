"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python pipeline.py                      # the submission configuration (all steps)
    python pipeline.py --pct 2              # smoke test on a 2% hash-sample of Source 2/3
    python pipeline.py --steps calibrate predict validate   # resume from a step

Defaults are the configuration that produces the submission: all of train and test is blocked
and featurized (so validation happens at the test's density), the stage-2 / stage-3 models are
fitted on 30% of the train S2/S3 records (memory) and the other 70% is an honest hold-out used
to choose the decision rule. Every run writes its settings to <work>/run_config.json.

Steps (in order):
  ingest               raw TSV -> parquet, ground truth exploded into pairs
  translit             learn Indic-script -> Latin token dictionary from all training matches
  prep_train/prep_test normalize names and addresses
  stage1_data          unpruned candidates on a small train slice (stage-1 training data)
  stage1               train the two cross-fitted stage-1 candidate pruners
  block_train/test     blocking with stage-1 pruning -> {split}_candidates.parquet
  eval_blocking        recall ceiling of the pruned train candidates vs ground truth
  match_features_*     pair features on the pruned candidates
  train                stage-2 LightGBM on the fit rows (S1-entity folds, out-of-fold scores)
  score                honest stage-2 probabilities for every train and test pair
  stage3               cluster-consistency features + stage-3 re-scoring (cluster.py)
  calibrate            choose the decision rule on the held-out train probabilities
  predict              apply it (+ guards) to the test probabilities, write both output TSVs
  validate             run the official validator on the outputs
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# reproducible defaults (children inherit them); override in the environment if needed
DEFAULT_ENV = {
    "ER_THREADS": "16",            # LightGBM is deterministic for a fixed thread count
    "ER_BLOCK_CHUNK": "100000",    # S2/S3 records per blocking chunk (memory)
    "ER_FEATURE_PART": "200000",   # candidate pairs per feature part (memory)
    "PYTHONIOENCODING": "utf-8",   # logs redirected to files: polars tables print box characters
}
for k, v in DEFAULT_ENV.items():
    os.environ.setdefault(k, v)

from config import DATA_DIR, OUTPUT_DIR, WORK_DIR  # noqa: E402

STEPS = ["ingest", "translit", "prep_train", "prep_test", "stage1_data", "stage1",
         "block_train", "block_test", "eval_blocking", "match_features_train",
         "match_features_test", "train", "score", "stage3", "calibrate", "predict", "validate"]


def run_step(step: str, train_pct: int, test_pct: int, stage1_pct: int, fit_pct: int):
    import blocking
    import eval_blocking
    import match
    import prep
    import stage1

    matched = train_pct == test_pct
    if step == "ingest":
        import ingest
        ingest.main()
    elif step == "translit":
        import normalize
        import translit
        translit.build(100)  # cheap (only Indic-script pairs), so always from all of train
        normalize.reload_translit()
    elif step == "prep_train":
        prep.prep("train", train_pct)
    elif step == "prep_test":
        prep.prep("test", test_pct)
    elif step == "stage1_data":
        blocking.run("train", min(train_pct, stage1_pct), raw=True)
    elif step == "stage1":
        stage1.train(min(train_pct, stage1_pct))
    elif step == "block_train":
        blocking.run("train", train_pct)
    elif step == "block_test":
        blocking.run("test", test_pct)
    elif step == "eval_blocking":
        eval_blocking.evaluate(train_pct)
    elif step == "match_features_train":
        match.build("train", train_pct)
    elif step == "match_features_test":
        match.build("test", test_pct)
    elif step == "train":
        match.train(train_pct, density_matched=matched, fit_pct=fit_pct)
    elif step == "score":
        match.score(train_pct, test_pct, fit_pct)
    elif step == "stage3":
        if os.environ.get("ER_STAGE3", "1") == "0" or not matched:
            print("stage 3 skipped (disabled, or train/test densities differ)")
        else:
            import cluster
            cluster.run(train_pct, test_pct, fit_pct)
    elif step == "calibrate":
        match.calibrate(train_pct, density_matched=matched, fit_pct=fit_pct)
    elif step == "predict":
        match.predict(test_pct)
    elif step == "validate":
        validator = DATA_DIR.parent / "utils" / "validate_submission.py"
        rc = subprocess.call([sys.executable, str(validator),
                              "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
                              "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
                              "--test-dir", str(DATA_DIR / "test")])
        if rc:
            raise SystemExit("validator FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pct", type=int, default=100,
                    help="%% of Source 2/3 used for BOTH splits (smoke tests)")
    ap.add_argument("--train-pct", type=int, default=None,
                    help="%% of train Source 2/3 to block/featurize (overrides --pct for train)")
    ap.add_argument("--test-pct", type=int, default=None,
                    help="%% of test Source 2/3 (overrides --pct for test)")
    ap.add_argument("--fit-pct", type=int, default=30,
                    help="%% of the blocked train S2/S3 records used to FIT the models; the rest "
                         "is the hold-out that chooses the decision rule")
    ap.add_argument("--stage1-pct", type=int, default=int(os.environ.get("ER_STAGE1_PCT", 2)),
                    help="%% of train Source 2/3 used to train the two cross-fitted pruners")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--one-step", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    train_pct = a.train_pct if a.train_pct is not None else a.pct
    test_pct = a.test_pct if a.test_pct is not None else a.pct
    steps = [s for s in STEPS if s in a.steps]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.one_step:  # child process: run exactly one step in-process
        run_step(steps[0], train_pct, test_pct, a.stage1_pct, a.fit_pct)
        return
    cfg = {"started": f"{datetime.now():%Y-%m-%d %H:%M}", "train_pct": train_pct,
           "test_pct": test_pct, "fit_pct": a.fit_pct, "stage1_pct": a.stage1_pct,
           "steps": steps, "env": {k: v for k, v in os.environ.items() if k.startswith("ER_")}}
    (WORK_DIR / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg), flush=True)
    for step in steps:
        t = time.time()
        print(f"\n===== {step} =====", flush=True)
        # every step in a fresh process: memory from earlier steps is fully returned to the OS
        rc = subprocess.call([sys.executable, "-u", __file__, "--one-step", "--steps", step,
                              "--train-pct", str(train_pct), "--test-pct", str(test_pct),
                              "--stage1-pct", str(a.stage1_pct), "--fit-pct", str(a.fit_pct)])
        if rc:
            raise SystemExit(f"step {step} failed (exit code {rc})")
        print(f"===== {step} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
