#!/usr/bin/env python3
"""Split cumulative periodic gem5 statistics into independent time windows."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.transient.validation import validate_window_timeline


BEGIN = "---------- Begin Simulation Statistics ----------"
END = "---------- End Simulation Statistics   ----------"


def parse_section(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("-") or line[0].isspace():
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value):
            values[fields[0]] = value
    return values


def parse_sections(path: Path) -> list[dict[str, float]]:
    chunks = path.read_text(encoding="utf-8").split(BEGIN)[1:]
    sections = [parse_section(chunk.split(END, 1)[0]) for chunk in chunks]
    return [section for section in sections if section]


def write_stats(path: Path, stats: dict[str, float]) -> None:
    lines = [BEGIN]
    for name in sorted(stats):
        value = stats[name]
        if abs(value - round(value)) < 1e-9:
            rendered = str(int(round(value)))
        else:
            rendered = f"{value:.12g}"
        lines.append(f"{name:<55} {rendered}")
    lines.append(END)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_windows(transient_r1_dir: Path, output_dir: Path) -> dict:
    transient_r1_dir = transient_r1_dir.resolve()
    output_dir = output_dir.resolve()
    metadata = read_json(transient_r1_dir / "r1_metadata.json")
    if metadata.get("transient_statistics") is not True:
        raise ValueError("R1 metadata does not identify a transient-statistics run")
    if metadata.get("transient_stats_mode") != "cumulative":
        raise ValueError("only cumulative periodic statistics are supported")
    sample_ticks = int(metadata["sample_interval_ticks"])
    sample_s = float(metadata["sample_interval_s"])
    measurement_start = int(metadata["measurement_start_tick"])
    sections = parse_sections(transient_r1_dir / "stats.txt")
    if not sections:
        raise ValueError("stats.txt contains no simulation-statistics sections")

    output_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[str, float] = {}
    previous_end = measurement_start
    windows = []
    dropped_sections = []
    for section_index, current in enumerate(sections):
        if "finalTick" not in current:
            raise ValueError(f"statistics section {section_index} has no finalTick")
        end_tick = int(current["finalTick"])
        if end_tick <= measurement_start:
            dropped_sections.append({
                "section_index": section_index,
                "reason": "statistics dump precedes the measured transient ROI",
                "final_tick": end_tick,
            })
            continue
        duration_ticks = end_tick - previous_end
        if duration_ticks <= 0:
            dropped_sections.append({
                "section_index": section_index,
                "reason": "non-positive duration (usually a duplicate final dump)",
                "final_tick": end_tick,
            })
            continue

        delta = {}
        clamped_negative = []
        for name, value in current.items():
            difference = value - previous.get(name, 0.0)
            if difference < 0:
                clamped_negative.append(name)
                difference = 0.0
            delta[name] = difference
        sim_freq = float(current.get("simFreq", 1.0 / sample_s * sample_ticks))
        delta["simFreq"] = sim_freq
        delta["simTicks"] = float(duration_ticks)
        delta["finalTick"] = float(end_tick)
        delta["simSeconds"] = duration_ticks / sim_freq

        window_index = len(windows)
        window_dir = output_dir / f"window_{window_index:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        write_stats(window_dir / "stats.txt", delta)
        stats_sha256 = hashlib.sha256(
            (window_dir / "stats.txt").read_bytes()
        ).hexdigest()
        window_metadata = {
            **metadata,
            "stage": "CLIP-3D transient power window",
            "window_index": window_index,
            "source_section_index": section_index,
            "window_start_tick": previous_end,
            "window_end_tick": end_tick,
            "window_duration_ticks": duration_ticks,
            "window_duration_s": duration_ticks / sim_freq,
            "nominal_sample_interval_s": sample_s,
            "is_partial_window": duration_ticks != sample_ticks,
        }
        write_json(window_dir / "r1_metadata.json", window_metadata)
        windows.append({
            "index": window_index,
            "source_section_index": section_index,
            "directory": str(window_dir),
            "stats": str(window_dir / "stats.txt"),
            "stats_sha256": stats_sha256,
            "start_tick": previous_end,
            "end_tick": end_tick,
            "duration_ticks": duration_ticks,
            "duration_s": duration_ticks / sim_freq,
            "is_partial": duration_ticks != sample_ticks,
            "clamped_negative_stat_count": len(clamped_negative),
        })
        previous = current
        previous_end = end_tick

    if not windows:
        raise ValueError("no positive-duration statistics windows were produced")
    manifest = {
        "schema_version": 1,
        "source_r1": str(transient_r1_dir),
        "source_stats": str((transient_r1_dir / "stats.txt").resolve()),
        "stats_mode": "cumulative snapshots converted to per-window deltas",
        "nominal_sample_interval_ms": sample_s * 1000.0,
        "nominal_sample_interval_ticks": sample_ticks,
        "measurement_start_tick": measurement_start,
        "window_count": len(windows),
        "total_duration_s": sum(window["duration_s"] for window in windows),
        "windows": windows,
        "dropped_sections": dropped_sections,
        "note": (
            "Negative deltas from non-additive gem5 formulas are clamped to zero. "
            "The McPAT mapping consumes additive event/cycle counters."
        ),
    }
    manifest["timeline_audit"] = validate_window_timeline(manifest)
    write_json(output_dir / "windows_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transient-r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = split_windows(args.transient_r1_dir, args.output_dir)
    print(
        f"Split {result['window_count']} windows over "
        f"{result['total_duration_s']:.6f} simulated seconds"
    )


if __name__ == "__main__":
    main()
