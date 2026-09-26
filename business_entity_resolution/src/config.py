"""Paths and global settings. Override the data/work/output roots with env vars."""
import os
from pathlib import Path

import _compat  # noqa: F401  (must run before scipy/sklearn are imported anywhere)

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]  # business_entity_resolution/


def _default_data_dir() -> Path:
    # student_resource/ next to business_entity_resolution/ (preferred: keeps the data out of
    # the code folder that gets zipped), or inside it
    for base in (ROOT, PKG):
        if (base / "student_resource" / "dataset").is_dir():
            return base / "student_resource" / "dataset"
    return ROOT / "student_resource" / "dataset"


DATA_DIR = Path(os.environ.get("ER_DATA_DIR", _default_data_dir()))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SEED = 42

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def raw_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def gt_path() -> Path:
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def pq_path(split: str, source: int) -> Path:
    return WORK_DIR / f"{split}_s{source}.parquet"
