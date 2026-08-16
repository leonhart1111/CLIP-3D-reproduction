#!/usr/bin/env python3
"""The sole strict execution boundary for a patched McPAT binary."""

from __future__ import annotations

import subprocess
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, sha256_file, write_json
from workflow.mcpat.cache_metrics import MARKER
from workflow.mcpat.gem5_to_mcpat import convert
from workflow.mcpat.parse_mcpat import parse_mcpat_text


DEFAULT_MCPAT = PROJECT_ROOT / "tools/src/mcpat/mcpat"
PATCH_FILE = PROJECT_ROOT / "patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch"
BUILD_PROVENANCE = PROJECT_ROOT / "tools/build/mcpat/build_provenance.json"
VERSION_HEADER = "McPAT (version 1.3"
MCPAT_PROVENANCE_SCHEMA_VERSION = 1
MCPAT_PROVENANCE_AUTHORITY = "CLIP strict patched McPAT 1.3 runner"
_MODEL_SETTING_KEYS = {
    "temperature_k", "device_type", "longer_channel_device",
    "interconnect_projection_type",
}
_RUN_SETTING_KEYS = _MODEL_SETTING_KEYS | {"opt_for_clk"}


def _validate_patched_binary(executable: Path) -> dict:
    """Validate the marker and, for the configured binary, its build record."""
    executable = Path(executable).resolve()
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"McPAT executable is not executable: {executable}")
    binary_bytes = executable.read_bytes()
    if MARKER.encode("utf-8") not in binary_bytes:
        raise ValueError(f"McPAT binary lacks required patch marker: {executable}")

    evidence = {"binary_sha256": sha256_file(executable)}
    if executable == DEFAULT_MCPAT.resolve():
        if not BUILD_PROVENANCE.is_file():
            raise FileNotFoundError(
                f"McPAT build provenance is missing: {BUILD_PROVENANCE}"
            )
        build = read_json(BUILD_PROVENANCE)
        expected_patch = sha256_file(PATCH_FILE)
        if build.get("binary_sha256") != evidence["binary_sha256"]:
            raise ValueError("McPAT build provenance does not match executable")
        if build.get("patch_sha256") != expected_patch:
            raise ValueError("McPAT build provenance does not match audited patch")
        evidence["build_provenance_sha256"] = sha256_file(BUILD_PROVENANCE)
    return evidence


def run_mcpat(r1_dir: Path, output_dir: Path, settings: dict,
              executable: Path = DEFAULT_MCPAT) -> dict:
    """Convert and execute McPAT exactly once, retaining native evidence on error."""
    settings = dict(settings or {})
    unknown = set(settings) - _RUN_SETTING_KEYS
    if unknown:
        raise ValueError(f"unsupported McPAT settings: {sorted(unknown)}")
    executable = Path(executable).resolve()
    binary_evidence = _validate_patched_binary(executable)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "input.xml"
    mapping_path = output_dir / "mapping_report.json"
    mapping = convert(
        Path(r1_dir), xml_path, report_path=mapping_path,
        settings={key: settings[key] for key in _MODEL_SETTING_KEYS if key in settings},
    )
    command = [
        str(executable), "-infile", str(xml_path), "-print_level", "5",
        "-opt_for_clk", str(int(settings.get("opt_for_clk", 0))),
    ]
    output_path = output_dir / "mcpat.out"
    try:
        process = subprocess.run(
            command, cwd=executable.parent, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_text = process.stdout or ""
    except OSError as error:
        output_text = f"McPAT launch failed: {error}\n"
        output_path.write_text(output_text, encoding="utf-8")
        raise RuntimeError(f"McPAT launch failed; see {output_path}") from error
    output_path.write_text(output_text, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"McPAT failed; see {output_path}")
    if VERSION_HEADER not in output_text:
        raise RuntimeError(f"McPAT output lacks version header; see {output_path}")
    if MARKER not in output_text:
        raise RuntimeError(f"McPAT output lacks embedded CACTI-P marker; see {output_path}")
    try:
        parsed = parse_mcpat_text(
            output_text, expected_core_count=4, require_granular_cores=True,
            require_embedded_cacti=True,
        )
    except ValueError as error:
        raise RuntimeError(f"strict McPAT parse failed; see {output_path}") from error

    parsed["command"] = command
    parsed["cache_contract"] = mapping["cache_contract"]
    parsed["optimization_constraints"] = mapping["optimization_constraints"]
    parsed["provenance"] = {
        "schema_version": MCPAT_PROVENANCE_SCHEMA_VERSION,
        "authority": MCPAT_PROVENANCE_AUTHORITY,
        "hashes": {
            "xml_sha256": sha256_file(xml_path),
            "mapping_sha256": sha256_file(mapping_path),
            "output_sha256": sha256_file(output_path),
            "binary_sha256": binary_evidence["binary_sha256"],
            "patch_sha256": sha256_file(PATCH_FILE),
            **({
                "build_provenance_sha256": binary_evidence[
                    "build_provenance_sha256"
                ],
            } if "build_provenance_sha256" in binary_evidence else {}),
        },
    }
    write_json(output_dir / "mcpat.json", parsed)
    return parsed
