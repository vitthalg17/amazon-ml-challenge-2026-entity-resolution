"""Light EDA: ground-truth structure, country mix, and side-by-side sampled examples."""
import sys

import polars as pl

from config import SEED, WORK_DIR, pq_path

pl.Config.set_tbl_rows(60)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(250)


def gt_stats():
    gt = pl.read_parquet(WORK_DIR / "train_gt.parquet")
    pairs = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    n = gt.height
    per = pairs.group_by("s1").agg(
        n=pl.len(),
        n2=pl.col("other").str.starts_with("S2-").sum(),
        n3=pl.col("other").str.starts_with("S3-").sum(),
    )
    print(f"S1 entities: {n:,}  with matches: {per.height:,}  singletons: {n - per.height:,} "
          f"({(n - per.height) / n:.1%})")
    print("matches per S1 (non-singletons):")
    print(per["n"].value_counts().sort("n").head(20))
    print("S2 matches per S1:"); print(per["n2"].value_counts().sort("n2").head(10))
    print("S3 matches per S1:"); print(per["n3"].value_counts().sort("n3").head(10))

    dup = pairs.group_by("other").len().filter(pl.col("len") > 1)
    print(f"S2/S3 ids matched to >1 S1: {dup.height:,}")
    for s in (2, 3):
        tot = pl.scan_parquet(pq_path("train", s)).select(pl.len()).collect().item()
        m = pairs.filter(pl.col("other").str.starts_with(f"S{s}-"))["other"].n_unique()
        print(f"S{s}: {tot:,} records, {m:,} matched to some S1 ({m / tot:.1%}); "
              f"{tot - m:,} unmatched distractors")


def country_mix():
    for split in ("train", "test"):
        for s in (1, 2, 3):
            vc = (pl.scan_parquet(pq_path(split, s)).group_by("country").len()
                  .sort("len", descending=True).collect())
            print(split, s, dict(zip(vc["country"], vc["len"])))


def examples(n_match=12, n_single=8):
    gt = pl.read_parquet(WORK_DIR / "train_gt.parquet")
    pairs = pl.read_parquet(WORK_DIR / "train_gt_pairs.parquet")
    s1 = pl.scan_parquet(pq_path("train", 1))
    matched = pairs["s1"].unique().sample(n_match, seed=SEED)
    single = gt.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
    single = single["source1_entity_id"].sample(n_single, seed=SEED)

    sel = pairs.filter(pl.col("s1").is_in(matched.implode()))
    others = pl.concat([
        pl.scan_parquet(pq_path("train", s)).filter(pl.col("entity_id").is_in(sel["other"].implode()))
        for s in (2, 3)
    ]).collect()
    s1c = s1.filter(pl.col("entity_id").is_in(pl.concat([matched, single]).implode())).collect()

    for sid in matched:
        r = s1c.filter(pl.col("entity_id") == sid).row(0)
        print(f"\n=== {r[0]} [{r[3]}]  {r[1]!r} | {r[2]!r}")
        ids = sel.filter(pl.col("s1") == sid)["other"]
        for o in others.filter(pl.col("entity_id").is_in(ids.implode())).sort("entity_id").iter_rows():
            print(f"   {o[0]:>14} [{o[3]}]  {o[1]!r} | {o[2]!r}")
    print("\n##### SINGLETONS")
    for r in s1c.filter(pl.col("entity_id").is_in(single.implode())).iter_rows():
        print(f"   {r[0]} [{r[3]}]  {r[1]!r} | {r[2]!r}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("gt", "all"):
        gt_stats()
    if what in ("country", "all"):
        country_mix()
    if what in ("ex", "all"):
        examples()
