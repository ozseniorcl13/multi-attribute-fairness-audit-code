"""Command-line entry point for the final paper experiments."""

from __future__ import annotations

import argparse
import importlib
import sys

from common.config import DEFAULT_SEEDS, MODEL_NAMES
from common.paths import raw_data_dir, run_results_dir

DATASETS = ("yale-ed", "brfss-2023", "diabetes-130", "nhanes")
SENSITIVITY_DATASETS = ("nhanes-single-split",)
MODULES = {
    "yale-ed": "experiments.yale_ed",
    "brfss-2023": "experiments.brfss_2023",
    "diabetes-130": "experiments.diabetes_130",
    "nhanes": "experiments.nhanes",
    "nhanes-single-split": "experiments.nhanes_single_split",
}
REPORTS = {
    "yale-ed": "reports.yale_ed",
    "brfss-2023": "reports.brfss_2023",
    "diabetes-130": "reports.diabetes_130",
    "nhanes": "reports.nhanes",
    "nhanes-single-split": "reports.nhanes_single_split",
}
REQUIRED_INPUTS = {
    "yale-ed": ("5v_cleandf.RData",),
    "brfss-2023": ("LLCP2023.XPT",),
    "diabetes-130": ("diabetic_data.csv",),
    "nhanes": ("P_DEMO.xpt", "P_GHB.xpt", "P_BMX.xpt", "P_BPXO.xpt"),
}


def _invoke(module_name: str, function_name: str = "main", arguments: tuple[str, ...] = ()) -> None:
    module = importlib.import_module(module_name)
    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        getattr(module, function_name)()
    finally:
        sys.argv = previous_argv


def _require_data(dataset: str) -> None:
    folder = raw_data_dir(dataset.replace("-", "_"))
    missing = [name for name in REQUIRED_INPUTS[dataset] if not (folder / name).is_file()]
    if missing:
        names = ", ".join(str(folder / name) for name in missing)
        raise FileNotFoundError(f"Required input file(s) not found: {names}")


def _run_yale_ed() -> None:
    _require_data("yale-ed")
    runner = MODULES["yale-ed"]
    output = run_results_dir("yale_ed")
    methods = "baseline,reweighing,threshold_optimizer"
    models = ",".join(MODEL_NAMES)
    _invoke(runner, arguments=(
        "--seeds", str(DEFAULT_SEEDS[0]), "--methods", methods, "--models", models,
        "--output-dir", str(output / "core"),
    ))
    _invoke(runner, arguments=(
        "--seeds", ",".join(str(seed) for seed in DEFAULT_SEEDS[1:]), "--methods", methods, "--models", models,
        "--output-dir", str(output / "repeat_seeds"),
    ))
    _invoke(REPORTS["yale-ed"])


def _run_dataset(dataset: str) -> None:
    if dataset == "yale-ed":
        _run_yale_ed()
        return
    _require_data("nhanes" if dataset == "nhanes-single-split" else dataset)
    if dataset in ("nhanes", "nhanes-single-split"):
        _invoke("datasets.nhanes_cohort")
    _invoke(MODULES[dataset])
    _invoke(REPORTS[dataset])


def _summarize(dataset: str) -> None:
    if dataset == "yale-ed":
        _invoke(REPORTS[dataset])
    elif dataset == "all":
        for name in DATASETS:
            _invoke(REPORTS[name])
        _invoke(REPORTS["nhanes-single-split"])
        _invoke("reports.nhanes_validation_comparison")
        _invoke("reports.paper_inputs", function_name="assemble_paper_inputs")
        _invoke("reports.eo_group_changes")
    else:
        _invoke(REPORTS[dataset])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="health-fairness",
        description="Run or summarize the final health fairness experiments.",
    )
    parser.add_argument("action", choices=("run", "prepare", "summarize"))
    parser.add_argument("dataset", choices=(*DATASETS, *SENSITIVITY_DATASETS, "all"))
    args = parser.parse_args(argv)

    try:
        if args.action == "prepare":
            if args.dataset != "nhanes":
                parser.error("the prepare command is currently available for nhanes only")
            _require_data("nhanes")
            _invoke("datasets.nhanes_cohort")
        elif args.action == "summarize":
            _summarize(args.dataset)
        elif args.dataset == "all":
            for dataset in DATASETS:
                _require_data(dataset)
            for dataset in DATASETS:
                _run_dataset(dataset)
            _run_dataset("nhanes-single-split")
            _invoke("reports.nhanes_validation_comparison")
            _invoke("reports.paper_inputs", function_name="assemble_paper_inputs")
            _invoke("reports.eo_group_changes")
        else:
            _run_dataset(args.dataset)
    except FileNotFoundError as error:
        parser.error(str(error))
