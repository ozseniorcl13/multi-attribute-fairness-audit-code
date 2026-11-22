"""Check that local raw data files match the project's recorded fingerprints."""

from __future__ import annotations

import csv
import hashlib
import argparse
from pathlib import Path

from common.paths import PROJECT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="also require the excluded BRFSS input to be present",
    )
    args = parser.parse_args()

    manifest = PROJECT_ROOT / "data" / "raw_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))

    failed = False
    for record in records:
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            if record["dataset"] == "brfss_2023" and not args.all:
                print(f"SKIPPED optional {record['path']} (download with scripts/download_brfss_2023.py)")
                continue
            print(f"MISSING {record['path']}")
            failed = True
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        status = "OK" if actual_size == int(record["bytes"]) and actual_hash == record["sha256"] else "MISMATCH"
        print(f"{status} {record['path']} ({actual_size} bytes)")
        failed |= status != "OK"
    if failed:
        raise SystemExit("One or more raw files are missing or differ from the recorded versions.")
    verified = len(records) if args.all else len(records) - sum(r["dataset"] == "brfss_2023" for r in records)
    print(f"Verified {verified} required raw dataset files.")


if __name__ == "__main__":
    main()
