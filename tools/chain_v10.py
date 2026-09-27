"""Take over the v10 run after the train blocking that is already running: wait for it, repeat it
only if it did not finish, then run the remaining steps with --fit-pct 40 (the orphans add ~20%
candidate pairs; 40% keeps the training step at the size v9 handled safely).

    python tools/chain_v10.py <pid of the running block_train watchdog>
"""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "full_runs"
REST = ["eval_blocking", "match_features_train", "match_features_test", "train", "score",
        "stage3", "calibrate", "predict", "validate"]


def log(msg):
    with open(LOGS / "v10.chain.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # no idle sleep
    pid = int(sys.argv[1])
    log(f"waiting for block_train (pid {pid})")
    while psutil.pid_exists(pid):
        time.sleep(20)
    text = (LOGS / "v10_block_train.log").read_text(encoding="utf-8", errors="replace")
    done = "===== block_train done" in text
    steps = REST if done else ["block_train"] + REST
    log(f"block_train {'finished' if done else 'did NOT finish -> repeating it'}; steps {steps}")
    env = os.environ.copy()
    rc = subprocess.call([sys.executable, str(ROOT / "tools" / "overnight.py"), "--name", "v10",
                          "--src", str(ROOT.parent / "er_v10" / "business_entity_resolution" / "src"),
                          "--train-pct", "100", "--test-pct", "100", "--fit-pct", "40",
                          "--steps", *steps, "--title",
                          "v10 v9 + orphan simulation, orphan-aware stage 3, open-set countries"],
                         env=env, cwd=ROOT)
    log(f"driver exit {rc}")


if __name__ == "__main__":
    main()
