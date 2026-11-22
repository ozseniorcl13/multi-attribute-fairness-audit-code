# Yale ED admission data

Source: the authors' public [Yale EMMLC admission-prediction
repository](https://github.com/yaleemmlc/admissionprediction/tree/master/Results).
Download `Results/5v_cleandf.RData` and place it at
`data/raw/yale_ed/5v_cleandf.RData`.

The source repository describes the file as a de-identified analysis dataframe
and asks users to cite Hong, Haimovich, and Taylor (2018), *Predicting hospital
admission at emergency department triage using machine learning*, PLOS ONE,
13(7), e0201016. The project versions this 98.65 MiB file by explicit study
decision. It is close to GitHub's 100 MiB hard limit, so do not replace it with
a newer copy without updating the manifest and auditing the repository history.
Verify it with `uv run python scripts/verify_raw_data.py`.
