#!/usr/bin/env python3
"""CLI entry point for transient-ROM calibration case materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json
from workflow.transient.rom.contracts import parse_settings
from workflow.transient.rom.materialize_calibration import execute_calibration_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--power-windows", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hotspot", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    report = execute_calibration_cases(
        args.modules, args.power_windows, args.config, read_json(args.design),
        args.output_dir, parse_settings(config), hotspot=args.hotspot,
    )
    print(
        "Transient ROM cases: "
        f"{report['training_hotspot_calls']} training, "
        f"{report['holdout_hotspot_calls']} holdout"
    )


if __name__ == "__main__":
    main()
