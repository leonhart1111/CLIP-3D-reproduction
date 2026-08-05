#!/usr/bin/env python3
"""Convert every gem5 time window to a McPAT module-power sample."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json
from workflow.mcpat.gem5_to_mcpat import convert
from workflow.mcpat.parse_mcpat import parse_mcpat_text


DEFAULT_MCPAT = PROJECT_ROOT / "tools/src/mcpat/mcpat"


def run_windows(windows_manifest: Path, output_dir: Path, config: dict,
                mcpat: Path = DEFAULT_MCPAT) -> dict:
    manifest = read_json(windows_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not mcpat.is_file():
        raise FileNotFoundError(mcpat)
    mcpat_config = config.get("mcpat", {})
    allowed_mcpat = {
        "temperature_k", "device_type", "longer_channel_device",
        "interconnect_projection_type", "opt_for_clk",
    }
    unknown_mcpat = set(mcpat_config) - allowed_mcpat
    if unknown_mcpat:
        raise ValueError(f"unsupported mcpat settings: {sorted(unknown_mcpat)}")
    settings = {
        key: mcpat_config[key] for key in (
            "temperature_k", "device_type", "longer_channel_device",
            "interconnect_projection_type",
        ) if key in mcpat_config
    }
    opt_for_clk = int(mcpat_config.get("opt_for_clk", 0))
    run_settings = {
        "mcpat_settings": settings,
        "opt_for_clk": opt_for_clk,
    }
    records = []
    expected_names: list[str] | None = None
    started = time.perf_counter()

    for window in manifest["windows"]:
        index = int(window["index"])
        source_dir = Path(window["directory"])
        window_dir = output_dir / f"window_{index:04d}"
        result_path = window_dir / "window_power.json"
        if result_path.is_file():
            existing = read_json(result_path)
            if (
                existing.get("source_stats_sha256") == window["stats_sha256"]
                and existing.get("run_settings") == run_settings
            ):
                records.append(existing)
                names = [module["name"] for module in existing["modules"]]
                expected_names = expected_names or names
                if names != expected_names:
                    raise ValueError(f"module names changed in cached window {index}")
                if (index + 1) % 10 == 0 or index + 1 == len(manifest["windows"]):
                    print(
                        f"Windowed McPAT: {index + 1}/{len(manifest['windows'])} "
                        "(reused)",
                        flush=True,
                    )
                continue

        window_dir.mkdir(parents=True, exist_ok=True)
        xml_path = window_dir / "input.xml"
        convert(source_dir, xml_path, settings=settings)
        command = [
            str(mcpat.resolve()), "-infile", str(xml_path),
            "-print_level", "5", "-opt_for_clk", str(opt_for_clk),
        ]
        process = subprocess.run(
            command, cwd=mcpat.parent, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (window_dir / "mcpat.out").write_text(process.stdout, encoding="utf-8")
        if (
            process.returncode != 0
            or "McPAT (version 1.3" not in process.stdout
            or "results" not in process.stdout
        ):
            raise RuntimeError(f"McPAT failed for window {index}: {window_dir / 'mcpat.out'}")
        parsed = parse_mcpat_text(process.stdout)
        parsed["command"] = command
        write_json(window_dir / "mcpat.json", parsed)
        names = [module["name"] for module in parsed["modules"]]
        expected_names = expected_names or names
        if names != expected_names:
            raise ValueError(f"module names changed in window {index}")
        record = {
            "schema_version": 1,
            "index": index,
            "source_stats": str((source_dir / "stats.txt").resolve()),
            "source_stats_sha256": window["stats_sha256"],
            "run_settings": run_settings,
            "start_tick": window["start_tick"],
            "end_tick": window["end_tick"],
            "duration_s": window["duration_s"],
            "is_partial": window["is_partial"],
            "modules": parsed["modules"],
            "totals": {
                field: sum(float(module[field]) for module in parsed["modules"])
                for field in ("dynamic_power_w", "leakage_power_w", "total_power_w")
            },
            "mcpat_json": str((window_dir / "mcpat.json").resolve()),
        }
        write_json(result_path, record)
        records.append(record)
        if (index + 1) % 10 == 0 or index + 1 == len(manifest["windows"]):
            print(
                f"Windowed McPAT: {index + 1}/{len(manifest['windows'])}",
                flush=True,
            )

    result = {
        "schema_version": 1,
        "source_windows": str(windows_manifest.resolve()),
        "window_count": len(records),
        "nominal_sample_interval_ms": manifest["nominal_sample_interval_ms"],
        "module_names": expected_names or [],
        "run_settings": run_settings,
        "elapsed_seconds": time.perf_counter() - started,
        "windows": records,
    }
    write_json(output_dir / "power_windows.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mcpat", type=Path, default=DEFAULT_MCPAT)
    args = parser.parse_args()
    result = run_windows(
        args.windows_manifest.resolve(), args.output_dir.resolve(),
        read_json(args.config), args.mcpat.resolve(),
    )
    print(
        f"McPAT power trace complete: {result['window_count']} windows, "
        f"elapsed={result['elapsed_seconds']:.1f} s"
    )


if __name__ == "__main__":
    main()
