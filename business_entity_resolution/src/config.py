"""Paths and global settings. Override the data/work/output roots with env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "student_resource" / "dataset"))
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
