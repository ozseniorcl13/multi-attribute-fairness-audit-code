# Data layout and versioning policy

Raw source data is kept outside the Python package. This repository versions the
three inputs selected for the final study: Diabetes 130-US Hospitals, the four
NHANES XPT files, and the Yale ED release. BRFSS remains download-on-demand
because its extracted file is too large for ordinary GitHub tracking.

- raw/<dataset>/: source files, with inclusion depending on license, size, and
  source terms.
- processed/nhanes/: derived NHANES cohort written by the preparation command.
- ../results/runs/: generated experiment summaries and paper-input tables.

See each dataset folder's README for its source, license/reuse notes, and
download or citation instructions. `raw_manifest.csv` records expected file
sizes and SHA-256 hashes.

## Dataset decision

| Dataset | Pinned size | Version directly in Git? | Decision for this repository |
|---|---:|---|---|
| Diabetes 130-US Hospitals | 19.16 MB (18.27 MiB) | **Yes** | Versioned in `raw/diabetes_130/`. UCI identifies the dataset as CC BY 4.0; retain attribution and the license link. |
| NHANES 2017--March 2020 (four XPT files) | 7.34 MB (7.00 MiB) | **Yes** | Versioned in `raw/nhanes/`. Keep CDC/NCHS provenance, cycle, and citation instructions with the files. |
| BRFSS 2023 combined XPT | 1.21 GB (1.12 GiB) | **No** | Download locally with `scripts/download_brfss_2023.py`; do not place the extracted file in Git. |
| Yale ED (`5v_cleandf.RData`) | 103.45 MB (98.65 MiB) | **Yes, by explicit project decision** | Versioned in `raw/yale_ed/`. It is close to GitHub's 100 MiB hard limit and triggers the large-file warning; retain the authors' citation and source terms. |

The repository therefore versions the three selected inputs, their provenance
READMEs, and the manifest. Git LFS is not used: Yale is below GitHub's 100 MiB
hard limit, while BRFSS remains excluded and is downloaded by script.
