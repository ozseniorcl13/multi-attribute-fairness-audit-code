import pandas as pd
import numpy as np

from common.paths import run_results_dir
from experiments.yale_ed import AUDIT_AXES, TARGET_AXES, sensitive_frame
from reports.paper_inputs import SPECS
from reports.seed_dispersion import summarize


def test_yale_axes_and_age_groups_are_part_of_standard_protocol():
    axis = "race_x_gender_x_age_band"
    assert {"age_band", axis}.issubset(TARGET_AXES)
    assert axis in AUDIT_AXES
    assert SPECS["yale"]["main_axis"] == SPECS["yale"]["main_target"] == axis
    data = pd.DataFrame({"gender": ["F", "M", "F", "M"], "race": ["A"] * 4,
                         "ethnicity": ["E"] * 4, "age": [39, 40, 65, None]})
    groups = sensitive_frame(data)
    assert groups.age_band.tolist() == ["18-39", "40-64", "65+", "Missing"]
    assert groups[axis].tolist() == ["A × F × 18-39", "A × M × 40-64", "A × F × 65+", "A × M × Missing"]


def test_yale_primary_result_and_seed_dispersion():
    path = run_results_dir("yale_ed") / "fairness_gaps_by_split.csv"
    gaps = pd.read_csv(path)
    main = gaps[(gaps.method == "baseline") & (gaps.audit_axis == "race_x_gender_x_age_band")]
    assert len(main) == 9
    assert main.groups_in_gap_n.eq(18).all()
    assert np.isclose(((main.tpr_gap + main.fpr_gap) / 2).mean(), 0.3280748382)
    sample = pd.DataFrame({"seed": [1, 2, 3], "condition": ["x"] * 3, "value": [1., 2., 3.]})
    result = summarize(sample, ["condition"], "value").iloc[0]
    assert result["mean"] == 2.0 and result["sd"] == 1.0
