"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python pipeline.py                      # full run (all steps)
    python pipeline.py --pct 2              # smoke test on a 2% hash-sample of Source 2/3
    python pipeline.py --steps block_test match_features_test predict   # resume from a step

Steps (in order):
  ingest               raw TSV -> parquet, ground truth exploded into pairs
  translit             learn Indic-script -> Latin token dictionary from training matches
  prep_train/prep_test normalize names and addresses
  stage1_data          unpruned candidates on a small train sample (stage-1 training data)
  stage1               train the stage-1 candidate pruner
  block_train/test     full blocking with stage-1 pruning  -> {split}_candidates.parquet
  eval_blocking        recall ceiling of the pruned train candidates vs ground truth
  match_features_*     pair features on the pruned candidates
  train                stage-2 LightGBM (S1-entity folds), OOF macro F0.5, threshold choice
  predict              score test, assign, write both output TSVs
  validate             run the official validator on the outputs
"""
import argparse
import subprocess
import sys
import time

from config import DATA_DIR, OUTPUT_DIR

STEPS = ["ingest", "translit", "prep_train", "prep_test", "stage1_data", "stage1",
         "block_train", "block_test", "eval_blocking", "match_features_train",
         "match_features_test", "train", "predict", "validate"]


def run_step(step: str, pct: int, stage1_pct: int):
    import blocking
    import eval_blocking
    import match
    import prep
    import stage1

    if step == "ingest":
        import ingest
        ingest.main()
    elif step == "translit":
        import normalize
        import translit
        name_map, addr_map = translit.learn(translit.load_pairs(min(pct, 100)))
        import json
        with open(translit.TRANSLIT_PATH, "w", encoding="utf-8") as f:
            json.dump({"name": name_map, "addr": addr_map}, f, ensure_ascii=False)
        normalize.reload_translit()
        print(f"translit: {len(name_map)} name / {len(addr_map)} address entries")
    elif step in ("prep_train", "prep_test"):
        prep.prep(step.split("_")[1], pct)
    elif step == "stage1_data":
        blocking.run("train", min(pct, stage1_pct), raw=True)
    elif step == "stage1":
        stage1.train(min(pct, stage1_pct))
    elif step in ("block_train", "block_test"):
        blocking.run(step.split("_")[1], pct)
    elif step == "eval_blocking":
        eval_blocking.evaluate(pct)
    elif step in ("match_features_train", "match_features_test"):
        match.build(step.split("_")[-1], pct)
    elif step == "train":
        match.train(pct)
    elif step == "predict":
        match.predict(pct)
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
    ap.add_argument("--pct", type=int, default=100, help="%% of Source 2/3 to use (smoke tests)")
    ap.add_argument("--stage1-pct", type=int, default=3,
                    help="%% of train Source 2/3 used to train the stage-1 pruner")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    a = ap.parse_args()
    for step in [s for s in STEPS if s in a.steps]:
        t = time.time()
        print(f"\n===== {step} =====", flush=True)
        run_step(step, a.pct, a.stage1_pct)
        print(f"===== {step} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
