"""Validate the fixed Balanced-50 selection and its complete layout roots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import math
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, sha256_file
from workflow.floorplan.layout_metrics import communication_weights_from_model
from workflow.r1_catalog import ArchitectureKey, expected_keys
from workflow.run_lifting_sweep import (
    CLIP3D_REQUIRED_ARTIFACTS,
    required_artifacts,
    validate_corrected_physical_artifacts,
)


EXPECTED_CLASSIFICATION = {
    "mode": "operational-exploratory-traffic-weighted",
    "non_formal": True,
    "paper_equivalent": False,
    "shared_parameter_accepted": False,
}
EXPECTED_PAIRS = [
    ("16kB", "128kB"), ("16kB", "512kB"), ("16kB", "2048kB"),
    ("32kB", "256kB"), ("32kB", "1024kB"),
    ("64kB", "128kB"), ("64kB", "512kB"), ("64kB", "2048kB"),
    ("128kB", "256kB"), ("128kB", "1024kB"),
]


def _resolve_reference(manifest: dict, name: str) -> tuple[Path, str]:
    """Resolve one manifest pin without allowing paths to escape the project."""
    reference = manifest.get(name)
    if not isinstance(reference, dict):
        raise ValueError(f"selection manifest {name} must be an object")
    relative = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"selection manifest {name} path is missing")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"selection manifest {name} path must be project-relative")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"selection manifest {name} sha256 is invalid")
    resolved = (PROJECT_ROOT / relative_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"selection manifest {name} path must be project-relative") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"selection manifest {name} is missing: {resolved}")
    if sha256_file(resolved) != digest:
        raise ValueError(f"selection manifest {name} sha256 does not match {resolved}")
    return resolved, digest


def load_selection(path: Path) -> dict:
    """Load a selection manifest and verify its immutable project-local references."""
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must contain an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported selection manifest schema")
    for name in ("canonical_grid_config", "experiment_config"):
        _resolve_reference(manifest, name)
    return manifest


def selection_keys(manifest: dict) -> list[ArchitectureKey]:
    """Return the manifest's explicit point order without deriving a sample."""
    points = manifest.get("points")
    if not isinstance(points, list):
        raise ValueError("selection manifest points must be an array")
    keys = []
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"selection point {index} must be an object")
        try:
            key = ArchitectureKey(
                str(point["workload"]), str(point["l1d_size"]), str(point["l2_size"])
            )
        except KeyError as error:
            raise ValueError(f"selection point {index} is missing {error.args[0]}") from error
        keys.append(key)
    return keys


def validate_selection(manifest: dict, grid: dict) -> dict:
    """Reject a selection that is not the fixed, balanced subset of the canonical grid."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported selection manifest schema")
    if manifest.get("experiment_classification") != EXPECTED_CLASSIFICATION:
        raise ValueError("selection manifest experiment classification is not exploratory")
    try:
        workloads = list(grid["workloads"])
        l1d_sizes = list(grid["l1d_sizes"])
        l2_sizes = list(grid["l2_sizes"])
    except (KeyError, TypeError) as error:
        raise ValueError("canonical grid must define workloads, l1d_sizes, and l2_sizes") from error
    canonical = expected_keys(grid)
    canonical_set = set(canonical)
    keys = selection_keys(manifest)
    counts = manifest.get("expected_counts")
    expected_counts = {
        "workloads": len(workloads),
        "canonical_points": len(canonical),
        "points_per_workload": len(EXPECTED_PAIRS),
        "selected_points": len(workloads) * len(EXPECTED_PAIRS),
    }
    if counts != expected_counts:
        raise ValueError(f"selection manifest expected_counts must equal {expected_counts}")
    if len(keys) != expected_counts["selected_points"]:
        raise ValueError(f"selection must contain {expected_counts['selected_points']} points")
    if len(set(keys)) != len(keys):
        raise ValueError("selection contains duplicate architecture keys")
    unknown = [key for key in keys if key not in canonical_set]
    if unknown:
        raise ValueError(f"selection contains non-canonical key: {unknown[0]}")
    expected = [ArchitectureKey(workload, l1d_size, l2_size)
                for workload in workloads for l1d_size, l2_size in EXPECTED_PAIRS]
    if keys != expected:
        raise ValueError("selection keys do not match the entire predeclared order")
    for workload in workloads:
        pairs = [(key.l1d_size, key.l2_size) for key in keys if key.workload == workload]
        if {l1d for l1d, _l2 in pairs} != set(l1d_sizes):
            raise ValueError(f"selection for {workload} does not cover every L1D level")
        if {l2 for _l1d, l2 in pairs} != set(l2_sizes):
            raise ValueError(f"selection for {workload} does not cover every L2 level")
    return {
        "selection_name": manifest.get("selection_name"),
        "workload_count": len(workloads),
        "canonical_count": len(canonical),
        "selected_count": len(keys),
        "points_per_workload": len(EXPECTED_PAIRS),
    }


def _read_object(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain an object: {path}")
    return value


def _validate_communication_profile(summary: dict, point: Path) -> None:
    try:
        weights = communication_weights_from_model(
            {"communication_profile": summary.get("communication_profile")}, required=True
        )
    except ValueError as error:
        raise ValueError(f"communication profile invalid for {point}: {error}") from error
    if weights is None or set(weights) != {0, 1, 2, 3}:
        raise ValueError(f"communication profile invalid for {point}: expected cores 0..3")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError(f"communication profile invalid for {point}: weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"communication profile invalid for {point}: weights must sum to one")


def _validate_physical_model(point: Path) -> dict:
    """Require one hash-bound McPAT-native granular physical model."""
    try:
        return validate_corrected_physical_artifacts(point)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"corrected McPAT-native physical model is invalid at {point}: {error}"
        ) from error


def validate_layout_point(
        point: Path, method: str, key: ArchitectureKey, config: dict,
        config_path: Path, *, require_layout_only: bool,
        existing_r2_validator: Callable[[str, ArchitectureKey, Path], dict] | None,
        expected_r1: Path | None = None,
) -> dict:
    """Apply the root preflight contract to one selected physical point."""
    point = Path(point).resolve()
    config_path = Path(config_path).resolve()
    artifacts = required_artifacts + (
        CLIP3D_REQUIRED_ARTIFACTS if method == "clip3d" else ()
    )
    for artifact in artifacts:
        if not (point / artifact).is_file():
            raise ValueError(f"{method} root is incomplete at {point}: missing {artifact}")
    run_config = _read_object(point / "run_config.json", "run_config.json")
    summary = _read_object(point / "pipeline_summary.json", "pipeline_summary.json")
    physical = _validate_physical_model(point)
    if expected_r1 is not None \
            and physical.get("source_r1") != str(Path(expected_r1).resolve()):
        raise ValueError(f"{method} physical model R1 source mismatch at {point}")
    if run_config.get("layout_method") != method:
        raise ValueError(f"{method} layout method mismatch in run_config at {point}")
    if summary.get("layout_method", summary.get("layout_mode")) != method:
        raise ValueError(f"{method} layout method mismatch in pipeline summary at {point}")
    if run_config.get("config") != config:
        raise ValueError(f"{method} embedded config does not match {config_path} at {point}")
    source = run_config.get("source")
    if not isinstance(source, str) or Path(source).resolve() != config_path:
        raise ValueError(f"{method} run_config source does not match {config_path} at {point}")
    if config.get("delay", {}).get("wire_aggregation") == "traffic-weighted":
        _validate_communication_profile(summary, point)
    has_r2 = summary.get("ipc2") is not None or summary.get("bips2") is not None
    if has_r2:
        if require_layout_only:
            raise ValueError(f"layout-only preflight found R2 results in {method} at {point}")
        if existing_r2_validator is None:
            raise ValueError(
                "resume preflight found existing R2 results but "
                "existing_r2_validator is required"
            )
        decision = existing_r2_validator(method, key, point)
        if not isinstance(decision, dict) or decision.get("accepted") is not True:
            raise ValueError(f"existing_r2_validator did not accept {method} R2 at {point}")
    return {
        "cache_authority": physical["cache_authority"],
        "has_r2": has_r2,
    }


def _validate_root(root: Path, method: str, canonical: list[ArchitectureKey],
                   config: dict, config_path: Path, require_layout_only: bool,
                   existing_r2_validator: Callable[[str, ArchitectureKey, Path], dict] | None,
                   expected_r1_root: Path | None = None,
                   ) -> dict:
    root = root.resolve()
    expected_r1_root = (
        None if expected_r1_root is None else Path(expected_r1_root).resolve()
    )
    expected_paths = {key.relative_path().as_posix() for key in canonical}
    found_paths = {
        path.parent.relative_to(root).as_posix()
        for path in root.rglob("run_config.json")
    } if root.is_dir() else set()
    missing = sorted(expected_paths - found_paths)
    extra = sorted(found_paths - expected_paths)
    if missing or extra:
        raise ValueError(
            f"{method} root must contain exactly {len(canonical)} canonical points; "
            f"found {len(found_paths)}, missing={missing[:1]}, extra={extra[:1]}"
        )

    existing_r2 = 0
    physical_authority = None
    for key in canonical:
        point = root / key.relative_path()
        decision = validate_layout_point(
            point, method, key, config, config_path,
            require_layout_only=require_layout_only,
            existing_r2_validator=existing_r2_validator,
            expected_r1=(
                None if expected_r1_root is None
                else expected_r1_root / key.relative_path()
            ),
        )
        physical_authority = decision["cache_authority"]
        existing_r2 += int(decision["has_r2"])
    return {
        "root": str(root),
        "canonical_count": len(canonical),
        "existing_r2_count": existing_r2,
        "physical_model_authority": physical_authority,
    }


def validate_layout_roots(
        fixed_root: Path, clip_root: Path, keys: list[ArchitectureKey], config_path: Path,
        *, selection: dict,
        require_layout_only: bool = True,
        existing_r2_validator: Callable[[str, ArchitectureKey, Path], dict] | None = None,
        expected_r1_root: Path | None = None,
) -> dict:
    """Validate complete paired layout roots before layout-only or resumable R2 work."""
    config_path = Path(config_path).resolve()
    grid_path, _grid_sha256 = _resolve_reference(selection, "canonical_grid_config")
    pinned_config_path, pinned_config_sha256 = _resolve_reference(
        selection, "experiment_config"
    )
    if config_path != pinned_config_path or sha256_file(config_path) != pinned_config_sha256:
        raise ValueError("supplied experiment config does not match pinned selection reference/hash")
    config = _read_object(config_path, "experiment config")
    if config.get("experiment_classification") != EXPECTED_CLASSIFICATION:
        raise ValueError("experiment classification must be the declared non-formal traffic-weighted classification")
    grid = _read_object(grid_path, "canonical grid config")
    validate_selection(selection, grid)
    selected = selection_keys(selection)
    if keys != selected:
        raise ValueError("selected keys do not match the supplied selection manifest")
    canonical = expected_keys(grid)
    if len(canonical) != 100 or len(set(canonical)) != 100:
        raise ValueError("canonical grid must contain exactly 100 unique architecture keys")
    if len(keys) != 50 or len(set(keys)) != 50 or any(key not in set(canonical) for key in keys):
        raise ValueError("selected keys must contain exactly 50 unique canonical architecture keys")
    if not require_layout_only and existing_r2_validator is None:
        # The callback is mandatory in resume mode even if no R2 result happens
        # to be found; otherwise a later root change could silently bypass it.
        raise ValueError("resume preflight requires existing_r2_validator")
    fixed = _validate_root(
        Path(fixed_root), "fixed-bin", canonical, config, config_path,
        require_layout_only, existing_r2_validator, expected_r1_root,
    )
    clip3d = _validate_root(
        Path(clip_root), "clip3d", canonical, config, config_path,
        require_layout_only, existing_r2_validator, expected_r1_root,
    )
    return {
        "schema_version": 1,
        "mode": "layout-only" if require_layout_only else "resume",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selected_count": len(keys),
        "selected_pairs": [asdict(key) for key in keys],
        "selected_keys": [asdict(key) for key in keys],
        "fixed": fixed,
        "clip3d": clip3d,
    }
