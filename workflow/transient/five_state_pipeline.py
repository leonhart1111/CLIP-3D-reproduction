"""Thin entry adapter for the five-state backend and shared validation path."""

from __future__ import annotations

from pathlib import Path


def run_five_state_pipeline(
        source_r1_dir: Path, steady_preflight_dir: Path,
        output_dir: Path, config_path: Path,
        transient_r1_dir: Path | None, *, execute_r2: bool,
        rerun_r2: bool = False) -> dict:
    """Skip POD calibration while reusing the common final HotSpot/R2 stages."""
    from workflow.transient.rom.run_pipeline import _run_transient_pipeline_core

    return _run_transient_pipeline_core(
        source_r1_dir, steady_preflight_dir, output_dir, config_path,
        transient_r1_dir, calibrate=False, execute_r2=execute_r2,
        rom_package_dir=None, rerun_r2=rerun_r2,
        thermal_backend="five-state",
    )
