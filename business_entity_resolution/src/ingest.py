"""Convert raw TSVs to Parquet (strings only, no quote handling) and explode ground truth."""
import polars as pl

from config import SOURCES, SPLITS, WORK_DIR, gt_path, pq_path, raw_path

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def main():
    for split in SPLITS:
        for s in SOURCES:
            df = read_tsv(raw_path(split, s))
            assert df.columns == COLS, df.columns
            assert df["entity_id"].n_unique() == df.height
            df.write_parquet(pq_path(split, s))
            print(split, s, df.shape)

    gt = read_tsv(gt_path())
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "other"})
    )
    gt.write_parquet(WORK_DIR / "train_gt.parquet")
    pairs.write_parquet(WORK_DIR / "train_gt_pairs.parquet")
    print("gt", gt.shape, "pairs", pairs.shape)


if __name__ == "__main__":
    main()
