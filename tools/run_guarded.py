"""Run a command with a memory watchdog and low CPU priority (protects a small laptop).

    python tools/run_guarded.py --min-avail-mb 1500 --log run.log -- python -u pipeline.py ...

- The command runs at below-normal priority, so the desktop stays responsive.
- Every 3 s the system's *available* RAM is checked; if it stays under --min-avail-mb for two
  checks in a row, the whole process tree is stopped (exit code 99) before Windows starts
  swapping heavily. Everything already written to disk is kept, so the run can be resumed with
  pipeline.py --steps ...
- Once a minute a status line (available RAM, the tree's memory, CPU) goes to <log>.watchdog.
"""
import argparse
import subprocess
import sys
import time

import psutil


def tree_rss_mb(p: psutil.Process) -> float:
    total = 0
    for q in [p, *p.children(recursive=True)]:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**20


def kill_tree(p: psutil.Process):
    procs = [*p.children(recursive=True), p]
    for q in procs:
        try:
            q.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=10)
    for q in alive:
        try:
            q.kill()
        except psutil.Error:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-avail-mb", type=int, default=1500)
    ap.add_argument("--log", required=True, help="stdout/stderr of the command go here")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

    flags = psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
    with open(a.log, "w", encoding="utf-8") as out, open(a.log + ".watchdog", "w") as wd:
        proc = subprocess.Popen(cmd, cwd=a.cwd, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=flags)
        if sys.platform != "win32":
            psutil.Process(proc.pid).nice(10)
        p = psutil.Process(proc.pid)
        t0, low_streak, last_log = time.time(), 0, 0.0
        min_avail, peak_rss = 1e12, 0.0
        wd.write(f"started pid {proc.pid}: {' '.join(cmd)}\n")
        wd.flush()
        while proc.poll() is None:
            avail = psutil.virtual_memory().available / 2**20
            rss = tree_rss_mb(p)
            min_avail, peak_rss = min(min_avail, avail), max(peak_rss, rss)
            low_streak = low_streak + 1 if avail < a.min_avail_mb else 0
            now = time.time()
            if now - last_log >= 60:
                wd.write(f"[{time.strftime('%H:%M:%S')}] +{(now - t0) / 60:5.1f} min  "
                         f"avail {avail:6.0f} MB  job {rss:6.0f} MB  "
                         f"cpu {psutil.cpu_percent():3.0f}%  (min avail {min_avail:.0f}, "
                         f"peak job {peak_rss:.0f})\n")
                wd.flush()
                last_log = now
            if low_streak >= 2:
                wd.write(f"WATCHDOG: available RAM {avail:.0f} MB < {a.min_avail_mb} MB -> "
                         f"stopping the job to protect the machine\n")
                wd.flush()
                kill_tree(p)
                out.write(f"\nWATCHDOG STOPPED THE JOB: available RAM fell to {avail:.0f} MB\n")
                out.flush()
                sys.exit(99)
            time.sleep(3)
        wd.write(f"finished with exit code {proc.returncode} after {(time.time() - t0) / 60:.1f} "
                 f"min; min available {min_avail:.0f} MB, peak job memory {peak_rss:.0f} MB\n")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
