#!/usr/bin/env python3
"""Run one multi-row detailed-3D HotSpot trace and summarize temperatures."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json


DEFAULT_HOTSPOT = PROJECT_ROOT / "tools/src/hotspot/hotspot"


def read_named_temperatures(path: Path) -> list[tuple[str, float]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values.append((fields[0], float(fields[-1])))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"no temperatures in {path}")
    return values


def parse_ttrace(path: Path, interval_s: float) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"temperature trace has no samples: {path}")
    names = lines[0].split()
    samples = []
    for index, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != len(names):
            raise ValueError(
                f"temperature trace row {index} has {len(fields)} values, "
                f"expected {len(names)}"
            )
        values = [float(value) for value in fields]
        peak_index = max(range(len(values)), key=values.__getitem__)
        samples.append({
            "index": index,
            "time_s": (index + 1) * interval_s,
            "peak_unit": names[peak_index],
            "tmax_k": values[peak_index],
            "tmax_c": values[peak_index] - 273.15,
            "tavg_c": sum(values) / len(values) - 273.15,
        })
    return samples


def run_hotspot_transient(case_dir: Path, hotspot: Path = DEFAULT_HOTSPOT,
                          initial_temperature: str = "steady",
                          steady_source: Path | None = None) -> dict:
    case_dir = case_dir.resolve()
    manifest = read_json(case_dir / "transient_trace_manifest.json")
    interval_s = float(manifest["sample_interval_s"])
    if initial_temperature not in ("steady", "ambient"):
        raise ValueError("initial_temperature must be steady or ambient")
    if not hotspot.is_file():
        raise FileNotFoundError(hotspot)

    initial_file = None
    if initial_temperature == "steady":
        if steady_source is None or not steady_source.is_file():
            raise FileNotFoundError(
                "steady initialization requires the selected steady HotSpot steady.txt"
            )
        initial_file = case_dir / "initial.steady.txt"
        shutil.copy2(steady_source, initial_file)

    ttrace = case_dir / "transient.ttrace"
    command = [
        str(hotspot.resolve()),
        "-c", "hotspot.config",
        "-p", "power_transient.ptrace",
        "-grid_layer_file", "stack.lcf",
        "-materials_file", "materials.txt",
        "-model_type", "grid",
        "-detailed_3D", "on",
        "-o", ttrace.name,
        "-steady_file", "average_power.steady.txt",
        "-grid_steady_file", "average_power.grid.steady.txt",
    ]
    if initial_file is not None:
        command.extend(("-init_file", initial_file.name))
    started = time.perf_counter()
    process = subprocess.run(
        command, cwd=case_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    (case_dir / "hotspot_transient.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0 or not ttrace.is_file():
        raise RuntimeError(
            f"transient HotSpot failed (rc={process.returncode}); "
            f"see {case_dir / 'hotspot_transient.log'}"
        )
    samples = parse_ttrace(ttrace, interval_s)
    if len(samples) != int(manifest["window_count"]):
        raise ValueError(
            f"HotSpot emitted {len(samples)} samples for "
            f"{manifest['window_count']} power windows"
        )

    ambient_c = float(read_json(case_dir / "hotspot_manifest.json")["ambient_c"])
    initial_peak = {
        "time_s": 0.0,
        "peak_unit": "ambient",
        "tmax_c": ambient_c,
        "tmax_k": ambient_c + 273.15,
    }
    if initial_file is not None:
        peak_name, peak_k = max(read_named_temperatures(initial_file), key=lambda item: item[1])
        initial_peak = {
            "time_s": 0.0,
            "peak_unit": peak_name,
            "tmax_k": peak_k,
            "tmax_c": peak_k - 273.15,
        }
    trace_peak = max(samples, key=lambda item: item["tmax_c"])
    overall_peak = max([initial_peak, *samples], key=lambda item: item["tmax_c"])
    with (case_dir / "transient_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("index", "time_s", "peak_unit", "tmax_c", "tavg_c")
        )
        writer.writeheader()
        writer.writerows({key: sample[key] for key in writer.fieldnames} for sample in samples)
    result = {
        "schema_version": 1,
        "command": command,
        "return_code": process.returncode,
        "elapsed_seconds": elapsed,
        "initial_temperature": initial_temperature,
        "initial_peak": initial_peak,
        "sample_interval_s": interval_s,
        "sample_count": len(samples),
        "trace_peak": trace_peak,
        "overall_peak": overall_peak,
        "tmax_c": overall_peak["tmax_c"],
        "temperature_trace": str(ttrace.resolve()),
        "summary_csv": str((case_dir / "transient_summary.csv").resolve()),
        "samples": samples,
    }
    write_json(case_dir / "transient_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--initial-temperature", choices=("steady", "ambient"),
                        default="steady")
    parser.add_argument("--steady-source", type=Path)
    args = parser.parse_args()
    result = run_hotspot_transient(
        args.case_dir, args.hotspot.resolve(), args.initial_temperature,
        args.steady_source.resolve() if args.steady_source else None,
    )
    print(
        f"Transient HotSpot Tmax={result['tmax_c']:.3f} C over "
        f"{result['sample_count']} samples"
    )


if __name__ == "__main__":
    main()
