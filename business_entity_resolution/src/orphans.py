"""Make the training data look like the test data: hide a share of the Source 1 entities.

Test has the same number of true matches per S1 entity as train (3.46) but 5.75 instead of 4.68
Source 2/3 records per entity. Its 9.97 M records are exactly what a train-like set with 2.13 M
entities would have, while test has 1.73 M: about 18.6 % of the entities were left out of
Source 1 and their records kept. Those "orphan" records look like real businesses and compete
for similar-named entities - hard distractors that plain train data does not have (test has
twice the share of records with an uncertain best score, and 25-65 % more candidates per S1).

This step removes the same share of S1 entities from train (by hash), and their labels: their
records become orphans. Blocking, features, training and calibration then run on data with the
test's distractor density and kind. Run after prep_train, before block_train.
Env: ER_ORPHAN_SHARE (default 0.186; 0 disables).
"""
import os

import polars as pl

from config import SEED, WORK_DIR
from prep import norm_path

SHARE = float(os.environ.get("ER_ORPHAN_SHARE", 0.186))


def hidden(col: str) -> pl.Expr:
    return pl.col(col).hash(SEED + 99) % 10_000 < int(round(SHARE * 10_000))


def _filter(path, col: str) -> tuple[int, int]:
    before = pl.scan_parquet(path).select(pl.len()).collect().item()
    tmp = path.with_suffix(".tmp")
    pl.scan_parquet(path).filter(~hidden(col)).sink_parquet(tmp)
    os.replace(tmp, path)  # new file (never writes through a hard link)
    return before, pl.scan_parquet(path).select(pl.len()).collect().item()


def run():
    if SHARE <= 0:
        print("orphans: disabled")
        return
    s1 = _filter(norm_path("train", 1, 100), "entity_id")
    gt = _filter(WORK_DIR / "train_gt_pairs.parquet", "s1")
    print(f"orphans: hid {SHARE:.1%} of train S1 entities: S1 {s1[0]:,} -> {s1[1]:,}, "
          f"true pairs {gt[0]:,} -> {gt[1]:,} (their records are now distractors)", flush=True)


if __name__ == "__main__":
    run()
