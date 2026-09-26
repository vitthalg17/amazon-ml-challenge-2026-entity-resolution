"""Normalize every source once and cache it: work/{split}_s{n}_norm.parquet.

--pct N keeps a deterministic N% hash-sample of Source 2/3 (Source 1 is always kept whole),
which is how blocking/model code is sanity-checked on a small machine.

Train records are normalized cross-fitted: a record whose S1 entity (or, for S1 itself and for
unmatched records, whose own id) hashes to fold f uses the transliteration dictionary learned
without fold f, so train features look like test features (dictionary never saw the pair).
"""
import argparse
import os
import re
import time

import polars as pl

from config import SEED, WORK_DIR, pq_path
from normalize import n_translit_folds, normalize, use_translit

PREP_SLICE = int(os.environ.get("ER_PREP_SLICE", 1_000_000))  # rows normalized at a time


def norm_path(split: str, source: int, pct: int = 100):
    tag = "" if pct >= 100 else f"_p{pct}"
    return WORK_DIR / f"{split}_s{source}_norm{tag}.parquet"


def _source_file(split: str, source: int, pct: int):
    """Smallest up-to-date normalized file that contains the pct sample: the exact sample, the
    full file, or a larger sample (e.g. the 6% stage-1 sample read from a --train-pct 30 prep)."""
    p = norm_path(split, source, pct if source > 1 else 100)
    full = norm_path(split, source, 100)
    # a sampled file older than the full one was normalized with outdated rules: re-sample
    if p.exists() and (p == full or not full.exists() or p.stat().st_mtime >= full.stat().st_mtime):
        return p
    if full.exists():
        return full
    larger = sorted((int(m.group(1)), f) for f in WORK_DIR.glob(f"{split}_s{source}_norm_p*.parquet")
                    if (m := re.search(r"_p(\d+)\.parquet$", f.name)) and int(m.group(1)) >= pct)
    if larger:
        return larger[0][1]
    raise FileNotFoundError(f"no normalized {split} source {source} covering {pct}%: run prep")


def load_norm(split: str, source: int, pct: int = 100, columns=None, lazy: bool = False):
    """Normalized source; for S2/S3 with pct < 100, a deterministic hash sample."""
    lf = pl.scan_parquet(_source_file(split, source, pct))
    if source > 1 and pct < 100:
        lf = lf.filter(pl.col("entity_id").hash(SEED) % 100 < pct)
    if columns:
        lf = lf.select(columns)
    return lf if lazy else lf.collect()


def _normalize_crossfit(lf: pl.LazyFrame, source: int, n: int) -> pl.DataFrame:
    if source == 1:
        key = pl.col("entity_id")
    else:
        gt = (pl.scan_parquet(WORK_DIR / "train_gt_pairs.parquet")
              .group_by("other").agg(pl.col("s1").min()))
        lf = lf.join(gt, left_on="entity_id", right_on="other", how="left")
        key = pl.coalesce("s1", "entity_id")
    df = (lf.with_columns(_tl_fold=key.hash(SEED) % n).drop("s1", strict=False)
          .with_row_index("_row").collect())
    parts = []
    for f in range(n):
        use_translit(f)  # read by the map_elements UDFs at collect time
        parts.append(normalize(df.filter(pl.col("_tl_fold") == f)).collect())
    use_translit()
    return pl.concat(parts).sort("_row").drop("_row", "_tl_fold")


def prep(split: str, pct: int = 100):
    n_folds = n_translit_folds()
    for s in (1, 2, 3):
        t = time.time()
        lf = pl.scan_parquet(pq_path(split, s))
        if s > 1 and pct < 100:
            lf = lf.filter(pl.col("entity_id").hash(SEED) % 100 < pct)
        # normalize in row slices written as parts, then stream them into one file: peak memory
        # is one slice, not the whole source (test sources have ~5M rows)
        out = norm_path(split, s, pct if s > 1 else 100)
        # any other normalized version of this source was made with older rules: remove it so
        # no later step can fall back to it (load_norm picks the smallest covering file)
        for old in WORK_DIR.glob(f"{split}_s{s}_norm*.parquet"):
            old.unlink()
        n = lf.select(pl.len()).collect().item()
        parts = []
        for i, off in enumerate(range(0, max(n, 1), PREP_SLICE)):
            sl = lf.slice(off, PREP_SLICE)
            df = (_normalize_crossfit(sl, s, n_folds) if split == "train" and n_folds
                  else normalize(sl).collect())
            parts.append(out.with_name(f"{out.stem}.part{i:02d}.parquet"))
            df.write_parquet(parts[-1])
            del df
        pl.scan_parquet(parts).sink_parquet(out)
        for f in parts:
            f.unlink()
        print(f"{split} s{s}: {n:,} rows normalized in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    a = ap.parse_args()
    prep(a.split, a.pct)
