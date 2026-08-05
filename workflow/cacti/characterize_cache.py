#!/usr/bin/env python3
"""Generate isolated CACTI configs and characterize CLIP-3D caches at 45 nm."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

from workflow.common import PROJECT_ROOT, parse_size_bytes, write_json


DEFAULT_CACTI = PROJECT_ROOT / "tools/src/cacti/cacti"
DEFAULT_CONFIG = PROJECT_ROOT / "tools/src/cacti/cache.cfg"


def replace_directive(text: str, directive: str, value: str) -> str:
    pattern = re.compile(
        rf"^(?!\s*//)\s*-{re.escape(directive)}(?=\s|$).*$", re.M | re.I
    )
    replacement = f"-{directive} {value}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"cannot find one active CACTI directive -{directive}")
    return updated


def make_config(base: str, size_bytes: int, associativity: int) -> str:
    text = base
    text = replace_directive(text, "size (bytes)", str(size_bytes))
    text = replace_directive(text, "block size (bytes)", "64")
    text = replace_directive(text, "associativity", str(associativity))
    text = replace_directive(text, "technology (u)", "0.045")
    # Use a full 64-byte cache line and keep the local CACTI result as the
    # single source of cache area and latency.
    text = replace_directive(text, "output/input bus width", "512")
    text = replace_directive(text, "Core count", "4")
    return text


def parse_cacti_output(text: str) -> dict[str, float]:
    patterns = {
        "access_time_ns": r"Access time \(ns\):\s*([0-9.eE+-]+)",
        "cycle_time_ns": r"Cycle time \(ns\):\s*([0-9.eE+-]+)",
        "read_energy_nj": r"Total dynamic read energy per access \(nJ\):\s*([0-9.eE+-]+)",
        "write_energy_nj": r"Total dynamic write energy per access \(nJ\):\s*([0-9.eE+-]+)",
        "leakage_power_mw": r"Total leakage power of a bank \(mW\):\s*([0-9.eE+-]+)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"CACTI output lacks {key}")
        result[key] = float(match.group(1))
    dimensions = re.search(r"Cache height x width \(mm\):\s*([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)", text)
    if not dimensions:
        raise ValueError("CACTI output lacks cache dimensions")
    height, width = map(float, dimensions.groups())
    result.update({"height_mm": height, "width_mm": width, "area_mm2": height * width})
    return result


def characterize(cacti: Path, base_config: Path, output_dir: Path,
                 l1_sizes: list[str], l2_sizes: list[str],
                 frequency_ghz: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config.read_text(encoding="utf-8", errors="replace")
    records = []
    for level, sizes, associativity in (("l1d", l1_sizes, 2), ("l2", l2_sizes, 8)):
        for size_text in sizes:
            size_bytes = parse_size_bytes(size_text)
            stem = f"{level}_{size_bytes}"
            cfg = output_dir / f"{stem}.cfg"
            raw = output_dir / f"{stem}.out"
            cfg.write_text(make_config(base, size_bytes, associativity), encoding="utf-8")
            process = subprocess.run(
                [str(cacti.resolve()), "-infile", str(cfg.resolve())],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # CACTI resolves tech_params/* relative to its working directory.
                cwd=cacti.resolve().parent,
            )
            raw.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"CACTI failed for {level} {size_text}; see {raw}")
            values = parse_cacti_output(process.stdout)
            value_source = "local CACTI run"
            clock_ns = 1.0 / frequency_ghz
            records.append({
                "level": level, "size": size_text, "size_bytes": size_bytes,
                "associativity": associativity, "technology_nm": 45,
                **values,
                "clock_period_ns": clock_ns,
                "access_cycles_unrounded": values["access_time_ns"] / clock_ns,
                "access_cycles": max(1, int(math.floor(values["access_time_ns"] / clock_ns + 0.5))),
                "value_source": value_source,
                "config": str(cfg.resolve()), "raw_output": str(raw.resolve()),
            })
    result = {
        "schema_version": 1, "frequency_ghz": frequency_ghz,
        "cache_value_source": "local CACTI run",
        "rounding": "nearest integer, floor(x + 0.5), minimum one cycle",
        "records": records,
        "cacti_parameters": ["45 nm", "L1 associativity 2", "L2 associativity 8"],
    }
    write_json(output_dir / "cacti_characterization.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cacti", type=Path, default=DEFAULT_CACTI)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l1-sizes", nargs="+", default=["16kB", "32kB", "64kB", "128kB"])
    parser.add_argument("--l2-sizes", nargs="+", default=["128kB", "256kB", "512kB", "1024kB", "2048kB"])
    parser.add_argument("--frequency-ghz", type=float, default=2.0)
    args = parser.parse_args()
    result = characterize(args.cacti, args.base_config, args.output_dir,
                          args.l1_sizes, args.l2_sizes, args.frequency_ghz)
    print(f"CACTI characterized {len(result['records'])} cache geometries")


if __name__ == "__main__":
    main()
