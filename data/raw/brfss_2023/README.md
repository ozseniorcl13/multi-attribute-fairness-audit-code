# BRFSS 2023

The combined landline and cellular SAS Transport file is available from the
[CDC 2023 BRFSS data page](https://www.cdc.gov/brfss/annual_data/annual_2023.html).

Run from the repository root:

```bash
uv run python scripts/download_brfss_2023.py
```

The script downloads the pinned CDC archive, extracts `LLCP2023.XPT`, and
checks its size and SHA-256 against `data/raw_manifest.csv`. The extracted file
is about 1.2 GB and is intentionally excluded from Git. The normal integrity
check skips this optional file; use `uv run python scripts/verify_raw_data.py
--all` after downloading it if you want to require every manifest entry.
