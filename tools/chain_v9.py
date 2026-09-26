"""Wait for the v8 run to end, then start v9 (--fit-pct 50) on the better of v7 / v8.

v9 re-uses that run's blocking and features and redoes train, score, stage 3, calibrate and
predict with half of the train S2/S3 records fitted (v7/v8: 30%). "Better" = the held-out
validation on the test distractor mix (decision.json macro_f05_w).
Code: ../er_v9 (v8 + train change) or ../er_v9b (v7 + the same train change).
Log: logs/full_runs/v9.chain.log
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "full_runs"


def log(msg):
    with open(LOGS / "v9.chain.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def v8_state():
    f = LOGS / "v8.driver.log"
    t = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
    if "save_submission exit" in t:
        return "done"
    if "FAILED" in t:
        return "failed"
    return None


def main():
    log("waiting for v8")
    while (st := v8_state()) is None:
        time.sleep(60)
    v7 = json.loads((ROOT / "work" / "decision.json").read_text())["macro_f05_w"]
    v8 = None
    if st == "done":
        v8 = json.loads((ROOT / "work_v8" / "decision.json").read_text())["macro_f05_w"]
    use8 = v8 is not None and v8 >= v7
    log(f"v8 {st}: validation v7 {v7:.5f}, v8 {v8} -> v9 on {'v8' if use8 else 'v7'}")
    env = os.environ.copy()
    env.update(ER_BLOCK_CHUNK="50000", PYTHONIOENCODING="utf-8",
               ER_OUTPUT_DIR=str(ROOT / "output_v9"),
               ER_DATA_DIR=str(ROOT / "student_resource" / "dataset"),
               ER_WORK_DIR=str(ROOT / ("work_v8" if use8 else "work")))
    src = ROOT.parent / ("er_v9" if use8 else "er_v9b") / "business_entity_resolution" / "src"
    title = f"v9 {'v8' if use8 else 'v7'} + fit on 50pct of train records"
    rc = subprocess.call([sys.executable, str(ROOT / "tools" / "overnight.py"), "--name", "v9",
                          "--src", str(src), "--train-pct", "100", "--test-pct", "100",
                          "--fit-pct", "50", "--steps", "train", "score", "stage3", "calibrate",
                          "predict", "validate", "--title", title], env=env, cwd=ROOT)
    log(f"v9 driver exit {rc}")


if __name__ == "__main__":
    main()
