"""Blocking recall against training ground truth (on the S2/S3 records that were blocked)."""
import argparse

import polars as pl

from blocking import CHANNELS, candidates_path
from config import WORK_DIR
from prep import load_norm


def evaluate(pct: int = 100, raw: bool = False, ks=(1, 2, 3, 5, 10)):
    cand = pl.read_parquet(candidates_path("train", pct, raw))
    oth = pl.concat([load_norm("train", s, pct,
                                     columns=["entity_id", "country", "addr_norm"])
                     for s in (2, 3)])
    gt = (pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
          .join(oth, left_on="other", right_on="entity_id"))
    n_gt, n_oth = gt.height, oth.height
    print(f"records blocked: {n_oth:,}  true pairs among them: {n_gt:,} "
          f"(distractors: {n_oth - n_gt:,})")
    print(f"candidate pairs: {cand.height:,} ({cand.height / n_oth:.1f} per S2/S3 record)")

    hit = gt.join(cand, left_on=["s1", "other"], right_on=["s1_id", "other_id"], how="left")
    chans = list(CHANNELS) + ["exact", "exact_addr"]
    rows = []
    for k in ks:
        flags = {ch: (pl.col(f"rank_{ch}") < k).fill_null(False) for ch in chans}
        rows.append(hit.select(pl.lit(k).alias("k"), *[f.mean().alias(ch) for ch, f in flags.items()],
                               pl.any_horizontal(list(flags.values())).mean().alias("union")))
    print("pair recall@k (per channel and union):")
    print(pl.concat(rows))

    found = hit.with_columns(found=pl.col("cos_combo").is_not_null(),
                             src=pl.col("other").str.slice(0, 2),
                             addr_empty=pl.col("addr_norm") == "")
    print(found.group_by("country", "src").agg(recall=pl.col("found").mean(), n=pl.len())
          .sort("country", "src"))
    print(found.group_by("addr_empty").agg(recall=pl.col("found").mean(), n=pl.len()))
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    evaluate(a.pct, a.raw)
