#!/usr/bin/env python3
"""Run the unchanged CLIP-3D R1 configuration with periodic statistics dumps.

This is deliberately a wrapper instead of a modification to ``clip_r1.py``.
It installs one cumulative statistics event immediately after the original
configuration resets its warm-up statistics, then executes the original file.
The resulting ``stats.txt`` therefore contains one cumulative section per
sampling interval plus the original final section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import m5


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--sample-ms", type=positive_float, default=10.0)
    wrapper.add_argument("--canonical-source-r1", type=Path, required=True)
    wrapper_args, original_args = wrapper.parse_known_args()
    sys.argv = [sys.argv[0], *original_args]

    sample_interval_s = wrapper_args.sample_ms / 1000.0
    original_reset = m5.stats.reset
    scheduling: dict[str, int] = {}

    def reset_and_schedule() -> None:
        original_reset()
        reset_calls = scheduling.get("reset_calls", 0) + 1
        scheduling["reset_calls"] = reset_calls
        start_tick = int(m5.curTick())
        # gem5 performs one statistics reset while instantiate() is still at
        # tick zero.  The original R1 then performs its explicit measurement
        # reset after warm-up (or as the second tick-zero reset when warm-up is
        # disabled).  Only the latter defines the power-sampling ROI.
        if start_tick == 0 and reset_calls == 1:
            return
        sample_ticks = int(m5.ticks.fromSeconds(sample_interval_s))
        if sample_ticks <= 0:
            raise ValueError("sample interval rounds to zero gem5 ticks")
        m5.stats.periodicStatDump(0)
        m5.stats.schedEvent(
            True,
            False,
            start_tick + sample_ticks,
            sample_ticks,
        )
        scheduling["measurement_start_tick"] = start_tick
        scheduling["sample_interval_ticks"] = sample_ticks

    m5.stats.reset = reset_and_schedule
    original = Path(__file__).resolve().with_name("clip_r1.py")
    try:
        runpy.run_path(str(original), run_name="__main__")
    finally:
        m5.stats.reset = original_reset

    if "measurement_start_tick" not in scheduling:
        raise RuntimeError("the original R1 configuration never reset statistics")

    metadata_path = Path(m5.options.outdir) / "r1_metadata.json"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata.update({
        "transient_statistics": True,
        "transient_stats_mode": "cumulative",
        "sample_interval_ms": wrapper_args.sample_ms,
        "sample_interval_s": sample_interval_s,
        "sample_interval_ticks": scheduling["sample_interval_ticks"],
        "measurement_start_tick": scheduling["measurement_start_tick"],
        "measurement_end_tick": int(m5.curTick()),
        "canonical_source_r1": str(wrapper_args.canonical_source_r1.resolve()),
        "canonical_source_metadata_sha256": "sha256:" + hashlib.sha256(
            (wrapper_args.canonical_source_r1 / "r1_metadata.json").read_bytes()
        ).hexdigest(),
        "transient_wrapper": str(Path(__file__).resolve()),
        "base_r1_config": str(original),
    })
    temporary = metadata_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    temporary.replace(metadata_path)


main()
