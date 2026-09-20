#!/usr/bin/env python3
"""Select and materialize the Gemmini-like accelerator testbench.

This module deliberately does not clone or modify Chipyard/Gemmini.  The
third-party repositories are external inputs to the CLIP workflow.  Instead,
the manifest freezes the workload contract and this command emits a checked
build plan that can be executed after the repositories have been obtained.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any

from workflow.common import PROJECT_ROOT, read_json, write_json


_SUPPORTED_FAMILIES = {"gemm", "conv2d", "transformer"}
_SUPPORTED_STATUSES = {"selected", "optional", "planned"}
_REQUIRED_OUTPUTS = {
    "cycles",
    "instructions",
    "dynamic_power_w",
    "leakage_power_w",
    "area_mm2",
}


def _fail(message: str) -> None:
    raise ValueError(f"invalid accelerator testbench: {message}")


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        _fail(f"{label} must be finite and positive")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(manifest: dict) -> dict:
    """Validate the immutable testbench contract and return the same object."""
    if not isinstance(manifest, dict):
        _fail("top-level value must be an object")
    if manifest.get("schema_version") != 1:
        _fail("schema_version must be 1")
    if not isinstance(manifest.get("name"), str) or not manifest["name"]:
        _fail("name must be a non-empty string")

    accelerator = manifest.get("accelerator")
    if not isinstance(accelerator, dict):
        _fail("accelerator must be an object")
    if accelerator.get("family") != "gemmini-systolic-array":
        _fail("accelerator.family must be gemmini-systolic-array")
    for key in ("generator_repo", "chipyard_repo", "default_config"):
        if not isinstance(accelerator.get(key), str) or not accelerator[key]:
            _fail(f"accelerator.{key} must be a non-empty string")
    mesh = accelerator.get("mesh")
    if not isinstance(mesh, dict):
        _fail("accelerator.mesh must be an object")
    rows = _positive_int(mesh.get("rows"), "accelerator.mesh.rows")
    cols = _positive_int(mesh.get("cols"), "accelerator.mesh.cols")
    if rows != cols:
        _fail("the MVP requires a square systolic mesh")
    _positive_number(accelerator.get("scratchpad_kb"),
                     "accelerator.scratchpad_kb")
    _positive_number(accelerator.get("dma_bus_bytes"),
                     "accelerator.dma_bus_bytes")
    if accelerator.get("dataflow") not in {"output-stationary", "weight-stationary"}:
        _fail("accelerator.dataflow must be output-stationary or weight-stationary")

    workloads = manifest.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        _fail("workloads must be a non-empty list")
    ids: set[str] = set()
    for workload in workloads:
        if not isinstance(workload, dict):
            _fail("each workload must be an object")
        identifier = workload.get("id")
        if not isinstance(identifier, str) or not identifier:
            _fail("each workload needs a non-empty id")
        if identifier in ids:
            _fail(f"duplicate workload id: {identifier}")
        ids.add(identifier)
        family = workload.get("family")
        if family not in _SUPPORTED_FAMILIES:
            _fail(f"{identifier}.family must be one of {sorted(_SUPPORTED_FAMILIES)}")
        if workload.get("status") not in _SUPPORTED_STATUSES:
            _fail(f"{identifier}.status must be one of {sorted(_SUPPORTED_STATUSES)}")
        source = workload.get("source")
        if not isinstance(source, dict):
            _fail(f"{identifier}.source must be an object")
        for key in ("kind", "entrypoint"):
            if not isinstance(source.get(key), str) or not source[key]:
                _fail(f"{identifier}.source.{key} must be a non-empty string")
        shape = workload.get("shape")
        if not isinstance(shape, dict) or not shape:
            _fail(f"{identifier}.shape must be a non-empty object")
        for key, value in shape.items():
            _positive_int(value, f"{identifier}.shape.{key}")
        outputs = workload.get("required_outputs")
        if not isinstance(outputs, list) or not outputs:
            _fail(f"{identifier}.required_outputs must be a non-empty list")
        missing = _REQUIRED_OUTPUTS.difference(outputs)
        if missing:
            _fail(f"{identifier}.required_outputs misses {sorted(missing)}")
        if workload.get("status") == "selected" and not workload.get("build"):
            _fail(f"selected workload {identifier} needs a build description")

    suites = manifest.get("suites")
    if not isinstance(suites, dict) or not suites:
        _fail("suites must be a non-empty object")
    for suite_name, suite in suites.items():
        if not isinstance(suite_name, str) or not suite_name:
            _fail("suite names must be non-empty strings")
        if not isinstance(suite, dict):
            _fail(f"suite {suite_name} must be an object")
        suite_ids = suite.get("workloads")
        if not isinstance(suite_ids, list) or not suite_ids:
            _fail(f"suite {suite_name}.workloads must be a non-empty list")
        unknown = set(suite_ids).difference(ids)
        if unknown:
            _fail(f"suite {suite_name} references unknown workloads {sorted(unknown)}")
        if len(set(suite_ids)) != len(suite_ids):
            _fail(f"suite {suite_name} contains duplicate workloads")

    policy = manifest.get("selection_policy")
    if not isinstance(policy, dict):
        _fail("selection_policy must be an object")
    if policy.get("thermal_grid") != "32x32_per_active_tier":
        _fail("selection_policy.thermal_grid must be 32x32_per_active_tier")
    if policy.get("active_tiers") != 2:
        _fail("selection_policy.active_tiers must be 2")
    return manifest


def load_and_validate(path: Path | str) -> dict:
    path = Path(path)
    return validate_manifest(read_json(path))


def select_workloads(manifest: dict, suite: str) -> list[dict]:
    """Return workload records in manifest order for one named suite."""
    validate_manifest(manifest)
    suites = manifest["suites"]
    if suite not in suites:
        raise ValueError(f"unknown accelerator testbench suite: {suite}")
    by_id = {workload["id"]: workload for workload in manifest["workloads"]}
    return [by_id[identifier] for identifier in suites[suite]["workloads"]]


def build_plan(manifest: dict, suite: str, external_root: Path | str) -> dict:
    """Create a reproducible, non-executing build/run plan.

    The plan intentionally uses logical workload entrypoints.  Exact binary
    names are revision-dependent in Gemmini, so a plan cannot silently claim
    that an upstream binary exists when it has not been built and checked.
    """
    validate_manifest(manifest)
    workloads = select_workloads(manifest, suite)
    external_root = Path(external_root).expanduser()
    chipyard = external_root / "chipyard"
    gemmini = external_root / "gemmini"
    accelerator = manifest["accelerator"]
    commands = [
        {
            "step": "obtain-chipyard",
            "cwd": str(external_root),
            "command": (
                f"git clone --recursive {shlex.quote(accelerator['chipyard_repo'])} "
                f"{shlex.quote(str(chipyard))}"
            ),
            "run_if": f"not exists: {chipyard}",
        },
        {
            "step": "obtain-gemmini",
            "cwd": str(external_root),
            "command": (
                f"git clone --recursive {shlex.quote(accelerator['generator_repo'])} "
                f"{shlex.quote(str(gemmini))}"
            ),
            "run_if": f"not exists: {gemmini}",
        },
        {
            "step": "chipyard-tools",
            "cwd": str(chipyard),
            "command": "./build-setup.sh",
            "run_if": "first checkout only; follow upstream prerequisites",
        },
        {
            "step": "build-verilator",
            "cwd": str(chipyard),
            "command": (
                "bash -lc " + shlex.quote(
                    f"source env.sh && make -C sims/verilator "
                    f"CONFIG={accelerator['default_config']}"
                )
            ),
            "run_if": "once per pinned Chipyard/Gemmini revision",
        },
        {
            "step": "build-rocc-tests",
            "cwd": str(chipyard),
            "command": (
                "bash -lc " + shlex.quote(
                    "source env.sh && "
                    "cd generators/gemmini/software/gemmini-rocc-tests && "
                    "./build.sh"
                )
            ),
            "run_if": "when the checked-out revision provides build.sh",
        },
    ]
    for workload in workloads:
        source = workload["source"]
        commands.append({
            "step": f"run-{workload['id']}",
            "cwd": str(chipyard),
            "command": (
                "<run the revision-specific Gemmini binary> "
                f"--workload {shlex.quote(workload['id'])} "
                f"--entrypoint {shlex.quote(source['entrypoint'])}"
            ),
            "required_outputs": workload["required_outputs"],
            "note": workload.get("build", {}).get("note", ""),
        })
    return {
        "schema_version": 1,
        "manifest_name": manifest["name"],
        "suite": suite,
        "external_root": str(external_root),
        "upstream": {
            "chipyard_repo": accelerator["chipyard_repo"],
            "gemmini_repo": accelerator["generator_repo"],
            "chipyard_revision": accelerator.get("chipyard_revision"),
            "gemmini_revision": accelerator.get("generator_revision"),
            "default_config": accelerator["default_config"],
        },
        "workloads": [
            {
                "id": workload["id"],
                "family": workload["family"],
                "shape": workload["shape"],
                "status": workload["status"],
                "source": workload["source"],
                "required_outputs": workload["required_outputs"],
            }
            for workload in workloads
        ],
        "commands": commands,
    }


def materialize_selection(manifest_path: Path | str, suite: str,
                          output: Path | str, external_root: Path | str) -> dict:
    """Validate and write a suite selection plus its build plan."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_and_validate(manifest_path)
    plan = build_plan(manifest, suite, external_root)
    raw_manifest = manifest_path.read_bytes()
    selection = {
        **plan,
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_bytes(raw_manifest),
        "provenance": {
            "third_party_source_is_external": True,
            "transformer_status": "planned_custom_gemm_chain",
            "thermal_label_source": "CLIP HotSpot 32x32 two-active-tier contract",
        },
    }
    write_json(output, selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--external-root", type=Path,
                        default=PROJECT_ROOT / "external")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    manifest = load_and_validate(args.manifest)
    plan = build_plan(manifest, args.suite, args.external_root)
    if args.output:
        materialize_selection(args.manifest, args.suite, args.output,
                              args.external_root)
    if args.print_plan or not args.output:
        print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
