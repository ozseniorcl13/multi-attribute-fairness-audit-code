"""Canonical project paths for data and experiment runs."""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
RESULTS_ROOT = PROJECT_ROOT / "results"
RUN_RESULTS_ROOT = RESULTS_ROOT / "runs"


def raw_data_dir(dataset: str) -> Path:
    return RAW_DATA_ROOT / dataset


def run_results_dir(dataset: str) -> Path:
    return RUN_RESULTS_ROOT / dataset
