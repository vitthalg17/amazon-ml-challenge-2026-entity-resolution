"""Normalize every source once and cache it: work/{split}_s{n}_norm.parquet.

--pct N keeps a deterministic N% hash-sample of Source 2/3 (Source 1 is always kept whole),
which is how blocking/model code is sanity-checked on a small machine.
"""
import argparse
import time

import polars as pl

from config import SEED, WORK_DIR, pq_path
from normalize import normalize


def norm_path(split: str, source: int, pct: int = 100):
    tag = "" if pct >= 100 else f"_p{pct}"
    return WORK_DIR / f"{split}_s{source}_norm{tag}.parquet"


def load_norm(split: str, source: int, pct: int = 100, columns=None) -> pl.DataFrame:
    """Normalized source; for S2/S3 with pct < 100, a deterministic hash sample."""
    p = norm_path(split, source, pct if source > 1 else 100)
    if p.exists():
        return pl.read_parquet(p, columns=columns)
    lf = pl.scan_parquet(norm_path(split, source, 100))
    if source > 1 and pct < 100:
        lf = lf.filter(pl.col("entity_id").hash(SEED) % 100 < pct)
    return (lf.select(columns) if columns else lf).collect()


def prep(split: str, pct: int = 100):
    for s in (1, 2, 3):
        t = time.time()
        lf = pl.scan_parquet(pq_path(split, s))
        if s > 1 and pct < 100:
            lf = lf.filter(pl.col("entity_id").hash(SEED) % 100 < pct)
        df = normalize(lf).collect()
        df.write_parquet(norm_path(split, s, pct if s > 1 else 100))
        print(f"{split} s{s}: {df.height:,} rows normalized in {time.time() - t:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--pct", type=int, default=100)
    a = ap.parse_args()
    prep(a.split, a.pct)
