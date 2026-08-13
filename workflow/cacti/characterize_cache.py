#!/usr/bin/env python3
"""Generate isolated CACTI configs and characterize CLIP-3D caches at 45 nm."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from workflow.cache_contract import (
    build_cache_contract,
    cache_access_cycles,
    stable_identity,
)
from workflow.common import PROJECT_ROOT, parse_size_bytes, sha256_file, write_json


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


def make_config(base: str, contract: dict) -> str:
    text = base
    text = replace_directive(text, "size (bytes)", str(contract["size_bytes"]))
    text = replace_directive(
        text, "block size (bytes)", str(contract["line_size_bytes"])
    )
    text = replace_directive(text, "associativity", str(contract["associativity"]))
    text = replace_directive(text, "UCA bank count", str(contract["bank_count"]))
    text = replace_directive(
        text, "technology (u)", f'{contract["technology_nm"] / 1000:g}'
    )
    text = replace_directive(
        text, "output/input bus width", str(contract["output_width_bits"])
    )
    text = replace_directive(
        text, "operating temperature (K)", str(contract["temperature_k"])
    )
    for directive in (
        "Data array cell type", "Data array peripheral type",
        "Tag array cell type", "Tag array peripheral type",
    ):
        text = replace_directive(text, directive, f'- "{contract["device_type"]}"')
    text = replace_directive(
        text, "access mode (normal, sequential, fast)",
        f'- "{contract["access_mode"]}"',
    )
    text = replace_directive(
        text, "Cache model (NUCA, UCA)", f'- "{contract["cacti_model"]}"'
    )
    text = replace_directive(
        text, "Interconnect projection", f'- "{contract["interconnect_projection"]}"'
    )
    text = replace_directive(text, "Core count", str(contract["core_count"]))
    text = replace_directive(
        text, "Cache level (L2/L3)", f'- "{contract["cacti_cache_level"]}"'
    )
    text = replace_directive(
        text, "Add ECC", f'- "{str(bool(contract["ecc"])).lower()}"'
    )
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines()) + "\n"


def normalized_text(text: str) -> str:
    """Preserve tool evidence while removing non-semantic trailing whitespace."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines()) + "\n"


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


def local_git_revision(directory: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--show-toplevel", "HEAD"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if process.returncode != 0:
        return None
    lines = process.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != directory.resolve():
        return None
    return lines[1]


def characterize(cacti: Path, base_config: Path, output_dir: Path,
                 l1_sizes: list[str] | None, l2_sizes: list[str] | None,
                 frequency_ghz: float, contracts: list[dict] | None = None,
                 *, technology_nm: int = 45, temperature_k: int = 320,
                 device_type: int = 0,
                 interconnect_projection_type: int = 1) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config.read_text(encoding="utf-8", errors="replace")
    if contracts is None:
        contracts = []
        for level, sizes, associativity in (
            ("l1d", l1_sizes or [], 2), ("l2", l2_sizes or [], 8)
        ):
            for size_text in sizes:
                metadata = {
                    "l1i_size": size_text, "l1d_size": size_text,
                    "l2_size": size_text, "l1_associativity": 2,
                    "l2_associativity": 8, "cache_line_bytes": 64,
                    "num_cores": 4,
                }
                generated = build_cache_contract(
                    metadata, technology_nm=technology_nm,
                    temperature_k=temperature_k, device_type=device_type,
                    interconnect_projection_type=interconnect_projection_type,
                )
                contracts.append(next(
                    record for record in generated["records"]
                    if record["level"] == level
                ))
    if not contracts:
        raise ValueError("CACTI characterization requires at least one cache contract")

    executable_hash = sha256_file(cacti)
    base_config_hash = sha256_file(base_config)
    records = []
    for contract in contracts:
            level = str(contract["level"])
            size_text = str(contract["size"])
            size_bytes = int(contract["size_bytes"])
            stem = f"{level}_{size_bytes}"
            cfg = output_dir / f"{stem}.cfg"
            raw = output_dir / f"{stem}.out"
            cfg.write_text(make_config(base, contract), encoding="utf-8")
            process = subprocess.run(
                [str(cacti.resolve()), "-infile", str(cfg.resolve())],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # CACTI resolves tech_params/* relative to its working directory.
                cwd=cacti.resolve().parent,
            )
            raw.write_text(normalized_text(process.stdout), encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"CACTI failed for {level} {size_text}; see {raw}")
            values = parse_cacti_output(process.stdout)
            clock_ns = 1.0 / frequency_ghz
            record = {
                **contract,
                **values,
                "clock_period_ns": clock_ns,
                "access_cycles_unrounded": values["access_time_ns"] / clock_ns,
                "access_cycles": cache_access_cycles(
                    values["access_time_ns"], frequency_ghz
                ),
                "cycle_cycles_unrounded": values["cycle_time_ns"] / clock_ns,
                "cycle_cycles": cache_access_cycles(
                    values["cycle_time_ns"], frequency_ghz
                ),
                "value_source": "local CACTI run",
                "config": str(cfg.resolve()), "raw_output": str(raw.resolve()),
                "config_sha256": sha256_file(cfg),
                "raw_output_sha256": sha256_file(raw),
            }
            record["cacti_record_id"] = stable_identity({
                key: value for key, value in record.items()
                if key not in ("config", "raw_output", "cacti_record_id")
            })
            records.append(record)
    result = {
        "schema_version": 2, "frequency_ghz": frequency_ghz,
        "cache_value_source": "local CACTI run",
        "rounding": "ceiling, minimum one cycle; 1e-12 tolerance at exact integer boundaries",
        "records": records,
        "provenance": {
            "cacti_executable": str(cacti.resolve()),
            "cacti_executable_sha256": executable_hash,
            "cacti_git_revision": local_git_revision(cacti.resolve().parent),
            "base_config": str(base_config.resolve()),
            "base_config_sha256": base_config_hash,
        },
    }
    result["characterization_id"] = stable_identity({
        key: value for key, value in result.items()
        if key != "characterization_id"
    })
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
    parser.add_argument("--temperature-k", type=int, default=320)
    args = parser.parse_args()
    result = characterize(args.cacti, args.base_config, args.output_dir,
                          args.l1_sizes, args.l2_sizes, args.frequency_ghz,
                          temperature_k=args.temperature_k)
    print(f"CACTI characterized {len(result['records'])} cache geometries")


if __name__ == "__main__":
    main()
