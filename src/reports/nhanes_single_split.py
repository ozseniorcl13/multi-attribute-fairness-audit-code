"""Summarize the recovered repeated single-split NHANES experiment."""

from common.paths import run_results_dir
from reports.nhanes import main as summarize_nhanes


def main() -> None:
    summarize_nhanes(
        output_dir=run_results_dir("nhanes_single_split"),
        predictions_filename="predictions_internal.csv.gz",
    )


if __name__ == "__main__":
    main()
