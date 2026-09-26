"""Build the final submission package from a saved submission folder.

    python tools/make_final_zip.py --submission submissions/v7_... --team TEAM \
        [--code-repo ../er_v8] [--doc Documentation.md]

<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv   (from the submission folder)
  code/business_entity_resolution/{src/*.py, README.md, requirements*.txt, tools/*.py}
  Documentation_template.md                                  (the filled-in write-up)
Checks that the two TSVs are the ones the submission folder holds and that the code folder is
the commit recorded in its NOTES.md.
"""
import argparse
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="submissions/vN_... folder")
    ap.add_argument("--team", required=True)
    ap.add_argument("--code-repo", default=str(ROOT), help="checkout holding the run's code")
    ap.add_argument("--doc", default=str(ROOT / "Documentation.md"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sub, repo = Path(a.submission), Path(a.code_repo)
    notes = (sub / "NOTES.md").read_text(encoding="utf-8")
    m = re.search(r"Code: `[^`]*` @ `([0-9a-f]+)`", notes)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    if m and not head.startswith(m.group(1)) and not m.group(1).startswith(head):
        print(f"WARNING: submission was made with commit {m.group(1)}, code folder is at {head}")
    out = Path(a.out or ROOT / f"{a.team}_submission.zip")
    pkg = repo / "business_entity_resolution"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub / "matching_results.tsv", "output/matching_results.tsv")
        z.write(sub / "for_final_zip" / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        for f in sorted((pkg / "src").glob("*.py")):
            z.write(f, f"code/business_entity_resolution/src/{f.name}")
        for name in ("README.md", "requirements.txt", "requirements-min.txt"):
            if (pkg / name).exists():
                z.write(pkg / name, f"code/business_entity_resolution/{name}")
        for f in ("run_guarded.py", "overnight.py"):
            if (repo / "tools" / f).exists():
                z.write(repo / "tools" / f, f"code/business_entity_resolution/tools/{f}")
        z.write(a.doc, "Documentation_template.md")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB, {len(names)} files)")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
