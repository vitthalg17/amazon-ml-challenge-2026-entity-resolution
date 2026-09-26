"""Unattended driver: runs pipeline steps one at a time under the RAM watchdog, resumes, retries.

    python tools/overnight.py --name v7 --train-pct 100 --test-pct 100 --fit-pct 30 \
        --steps prep_train block_train ... validate --title "v7 ..."

- Each step runs as `pipeline.py --steps <step>` inside tools/run_guarded.py (below-normal
  priority, stopped if available RAM stays under --min-avail-mb).
- Finished steps are recorded in logs/full_runs/<name>.state.json, so a re-run continues where
  the last one stopped (the steps themselves also resume from their finished parts).
- After a failure (watchdog stop or crash) it waits until RAM is free again and retries the
  step; from the second retry on, the memory knob of that step (chunk / part size) is halved.
- When every step is done, the output is saved as the next submissions/vN folder.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "business_entity_resolution" / "src"
LOGS = ROOT / "logs" / "full_runs"
# memory knob per step: (env var, default, floor)
KNOBS = {
    "block_train": ("ER_BLOCK_CHUNK", 50_000, 10_000),
    "block_test": ("ER_BLOCK_CHUNK", 50_000, 10_000),
    "stage1_data": ("ER_BLOCK_CHUNK", 50_000, 10_000),
    "match_features_train": ("ER_FEATURE_PART", 200_000, 50_000),
    "match_features_test": ("ER_FEATURE_PART", 200_000, 50_000),
    "stage3": ("ER_SIB_PART", 400_000, 100_000),
}


def say(log, msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_for_ram(log, need_mb):
    t0 = time.time()
    while True:
        avail = psutil.virtual_memory().available / 2**20
        if avail >= need_mb:
            return
        if time.time() - t0 < 5 or int(time.time() - t0) % 300 < 30:
            say(log, f"waiting for RAM: {avail:.0f} MB available, need {need_mb} MB")
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-pct", type=int, default=100)
    ap.add_argument("--test-pct", type=int, default=100)
    ap.add_argument("--fit-pct", type=int, default=30)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--min-avail-mb", type=int, default=1500)
    ap.add_argument("--start-mb", type=int, default=4500,
                    help="free RAM required before a step (re)starts")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--title", default=None, help="save the output as a submission when done")
    a = ap.parse_args()

    log = LOGS / f"{a.name}.driver.log"
    state_file = LOGS / f"{a.name}.state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": [], "env": {}}
    env = os.environ.copy()
    env.update(state["env"])
    say(log, f"driver start: {a.steps} (done so far: {state['done']})")
    for step in a.steps:
        if step in state["done"]:
            continue
        for attempt in range(a.retries + 1):
            wait_for_ram(log, a.start_mb)
            step_log = LOGS / f"{a.name}_{step}{'' if attempt == 0 else f'_retry{attempt}'}.log"
            say(log, f"{step}: attempt {attempt + 1}, log {step_log.name}, "
                     f"env { {k: v for k, v in env.items() if k.startswith('ER_')} }")
            t = time.time()
            rc = subprocess.call(
                [sys.executable, str(ROOT / "tools" / "run_guarded.py"),
                 "--min-avail-mb", str(a.min_avail_mb), "--log", str(step_log), "--cwd", str(SRC),
                 "--", sys.executable, "-u", "pipeline.py", "--steps", step,
                 "--train-pct", str(a.train_pct), "--test-pct", str(a.test_pct),
                 "--fit-pct", str(a.fit_pct)], env=env)
            say(log, f"{step}: exit {rc} after {(time.time() - t) / 60:.1f} min")
            if rc == 0:
                state["done"].append(step)
                state_file.write_text(json.dumps(state, indent=2))
                break
            if attempt >= 1 and step in KNOBS:  # second failure on: smaller pieces
                var, default, floor = KNOBS[step]
                cur = int(env.get(var, default))
                env[var] = state["env"][var] = str(max(floor, cur // 2))
                state_file.write_text(json.dumps(state, indent=2))
                say(log, f"{step}: {var} {cur} -> {env[var]}")
            time.sleep(60)
        else:
            say(log, f"{step}: FAILED after {a.retries + 1} attempts - stopping")
            sys.exit(1)
    say(log, "all steps done")
    if a.title:
        combined = LOGS / f"{a.name}.log"
        with open(combined, "w", encoding="utf-8") as out:
            for f in sorted(LOGS.glob(f"{a.name}_*.log"), key=lambda p: p.stat().st_mtime):
                out.write(f"\n##### {f.name}\n")
                out.write(f.read_text(encoding="utf-8", errors="replace"))
        rc = subprocess.call([sys.executable, str(ROOT / "tools" / "save_submission.py"),
                              a.title, "--log", str(combined)], cwd=ROOT)
        say(log, f"save_submission exit {rc}")


if __name__ == "__main__":
    main()
