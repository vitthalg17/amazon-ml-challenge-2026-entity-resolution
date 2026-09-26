"""Save the current output/ as a numbered submission version, so no run overwrites another.

    python tools/save_submission.py "france fix + 2000 rounds" --log logs/full_runs/run9.log

Creates submissions/vN_<slug>/ with matching_results.tsv, candidate_pairs.tsv, decision.json
and NOTES.md (git commit, run settings, validation score read from the run log), then updates
submissions/README.md. Fill in the leaderboard score later with --lb:

    python tools/save_submission.py --lb v2 0.9571
"""
import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
INDEX = SUB / "README.md"
HEADER = ("# Submissions\n\n"
          "One folder per leaderboard upload. Upload `matching_results.tsv` from the folder.\n\n"
          "| Version | Date | Code (branch @ commit) | Validation | Leaderboard | What changed |\n"
          "|---|---|---|---|---|---|\n")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rows() -> list[list[str]]:
    if not INDEX.exists():
        return []
    out = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if line.startswith("| v"):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def write_index(table: list[list[str]]):
    table.sort(key=lambda r: int(re.match(r"v(\d+)", r[0]).group(1)))
    INDEX.write_text(HEADER + "".join("| " + " | ".join(r) + " |\n" for r in table),
                     encoding="utf-8")


def save(title: str, log: str | None):
    table = rows()
    n = max([int(re.match(r"v(\d+)", r[0]).group(1)) for r in table] + [0]) + 1
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    d = SUB / f"v{n}_{slug}"
    d.mkdir(parents=True)
    # only the leaderboard file at the top level; the blocking set is for the final zip only
    shutil.copy2(ROOT / "output" / "matching_results.tsv", d / "matching_results.tsv")
    (d / "for_final_zip").mkdir()
    shutil.copy2(ROOT / "output" / "candidate_pairs.tsv", d / "for_final_zip" / "candidate_pairs.tsv")
    dec = ROOT / "work" / "decision.json"
    if dec.exists():
        shutil.copy2(dec, d / "decision.json")
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")
    dirty = " (+ uncommitted changes)" if git("status", "--porcelain", "business_entity_resolution") else ""
    val, cmd = "?", "?"
    if log and Path(log).exists():
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"threshold: ([0-9.]+)(?:\s+entity: ([0-9.]+))?\s+-> using (\w+) \(param ([-0-9.]+)", text)
        if m:
            t, e, rule, param = m[-1]
            val = max(t, e or "0")
            val = f"{val} ({rule} {param})"
        c = re.findall(r"^train_pct=.*$", text, re.M)
        cmd = c[-1] if c else "?"
    decision = json.loads(dec.read_text()) if dec.exists() else {}
    (d / "NOTES.md").write_text(
        f"# v{n}: {title}\n\n"
        f"- Saved: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"- Code: `{branch}` @ `{commit}`{dirty}\n"
        f"- Run settings: `{cmd}`\n"
        f"- Run log: `{log or '?'}`\n"
        f"- Decision rule: `{decision}`\n"
        f"- Validation (held-out train, macro F0.5): {val}\n"
        f"- Leaderboard: (fill in with `python tools/save_submission.py --lb v{n} <score>`)\n",
        encoding="utf-8")
    table.append([f"v{n}", f"{datetime.now():%m-%d %H:%M}", f"`{branch}` @ `{commit}`{dirty}",
                  val, "?", title])
    write_index(table)
    print(f"saved {d}")


def set_lb(version: str, score: str):
    table = rows()
    for r in table:
        if r[0] == version:
            r[4] = score
    write_index(table)
    for d in SUB.glob(f"{version}_*"):
        notes = d / "NOTES.md"
        notes.write_text(re.sub(r"- Leaderboard: .*", f"- Leaderboard: {score}",
                                notes.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{version}: leaderboard {score}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("title", nargs="?")
    ap.add_argument("--log")
    ap.add_argument("--lb", nargs=2, metavar=("VERSION", "SCORE"))
    a = ap.parse_args()
    if a.lb:
        set_lb(*a.lb)
    else:
        save(a.title or "run", a.log)
