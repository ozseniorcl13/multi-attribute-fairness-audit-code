#!/usr/bin/env python3
"""Download the official CDC BRFSS 2023 XPT and verify the pinned input."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/brfss_2023"
OUTPUT = DATA_DIR / "LLCP2023.XPT"
URL = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"
EXPECTED_BYTES = 1_205_554_400
EXPECTED_SHA256 = "3d3bf8ef5195bde227828ddc4c90745b76e8b304f8f5b9a043b6d99895fd1615"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        current_bytes = OUTPUT.stat().st_size
        current_hash = sha256(OUTPUT)
        if current_bytes == EXPECTED_BYTES and current_hash == EXPECTED_SHA256:
            print(f"Pinned BRFSS input already exists and is verified: {OUTPUT}")
            return

    with tempfile.TemporaryDirectory(prefix="brfss2023-") as temp_dir:
        temp = Path(temp_dir)
        archive = temp / "LLCP2023XPT.zip"
        extracted = temp / "LLCP2023.XPT"
        print(f"Downloading official CDC archive: {URL}", flush=True)
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("curl is required to download from CDC; install it in WSL and retry.")
        subprocess.run(
            [curl, "--fail", "--location", "--retry", "3", "--output", str(archive), URL],
            check=True,
        )

        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if Path(name.strip()).name.lower() == "llcp2023.xpt"]
            if len(members) != 1:
                raise RuntimeError(f"Expected one LLCP2023.XPT in CDC ZIP; found {members}")
            # Stream to disk without extracting arbitrary archive paths.
            with bundle.open(members[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)

        actual_bytes = extracted.stat().st_size
        actual_hash = sha256(extracted)
        if actual_bytes != EXPECTED_BYTES or actual_hash != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded BRFSS input differs from the version used in this study. "
                f"Expected {EXPECTED_BYTES} bytes / {EXPECTED_SHA256}; "
                f"got {actual_bytes} bytes / {actual_hash}. "
                "Do not replace the pinned study input without auditing and updating "
                "the manifest and experiment outputs."
            )
        extracted.replace(OUTPUT)
    print(f"Verified and saved {OUTPUT}")


if __name__ == "__main__":
    main()
