#!/usr/bin/env python3
"""Measure and publish the local CACTI equivalent of paper Table II."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from workflow.cacti.characterize_cache import DEFAULT_CACTI, DEFAULT_CONFIG, characterize
from workflow.common import PROJECT_ROOT, write_json


L1_SIZES = ["16kB", "32kB", "64kB", "128kB"]
L2_SIZES = ["128kB", "256kB", "512kB", "1024kB", "2048kB"]
REPORT_FIELDS = (
    "level", "size", "size_bytes", "access_time_ns",
    "access_cycles_unrounded", "access_cycles", "area_mm2", "width_mm",
    "height_mm", "associativity", "bank_count", "output_width_bits",
    "technology_nm", "temperature_k", "config_sha256",
    "raw_output_sha256", "cacti_record_id", "characterization_id",
    "cacti_git_revision", "cacti_executable_sha256", "base_config_sha256",
)


def write_reports(characterization: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = {"l1d": 0, "l2": 1}
    records = sorted(
        characterization["records"],
        key=lambda record: (order.get(record["level"], 99), record["size_bytes"]),
    )
    provenance = characterization["provenance"]
    rows = []
    for record in records:
        rows.append({
            **{field: record.get(field) for field in REPORT_FIELDS},
            "characterization_id": characterization["characterization_id"],
            "cacti_git_revision": provenance.get("cacti_git_revision"),
            "cacti_executable_sha256": provenance["cacti_executable_sha256"],
            "base_config_sha256": provenance["base_config_sha256"],
        })
    result = {
        "schema_version": 1,
        "description": "Local architecture-faithful CACTI Table-II equivalent; paper values are not inputs",
        "frequency_ghz": characterization["frequency_ghz"],
        "rounding": characterization["rounding"],
        "characterization_id": characterization["characterization_id"],
        "provenance": provenance,
        "records": rows,
    }
    json_path = output_dir / "local_45nm_table_ii_equivalent.json"
    csv_path = output_dir / "local_45nm_table_ii_equivalent.csv"
    write_json(json_path, result)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=REPORT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cacti", type=Path, default=DEFAULT_CACTI)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data/cacti/local_45nm_table_ii_equivalent_artifacts",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "data/cacti"
    )
    args = parser.parse_args()
    result = characterize(
        args.cacti.resolve(), args.base_config.resolve(), args.output_dir.resolve(),
        L1_SIZES, L2_SIZES, 2.0, technology_nm=45, temperature_k=320,
        device_type=0, interconnect_projection_type=1,
    )
    json_path, csv_path = write_reports(result, args.report_dir.resolve())
    print(f"Local Table-II equivalent: {len(result['records'])} rows")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
