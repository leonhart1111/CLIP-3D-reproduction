#!/usr/bin/env python3
"""Regression tests for the two-point transient-ROM validation entry."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json
from workflow.experiments.transient_rom_balanced2 import (
    preflight_inputs,
    run_validation_set,
)


SCIENTIFIC_CONFIG = (
    PROJECT_ROOT / "configs/experiments/"
    "clip3d_transient_rom_lambda0020119_traffic_weighted_"
    "discrete_partition_exploratory.json"
)


class Balanced2PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.canonical = self.root / "canonical"
        self.periodic = self.root / "periodic"
        self.selection = self.root / "selection.json"
        self.baseline = self.root / "baseline.csv"
        self.config = self.root / "config.json"
        shutil.copy2(SCIENTIFIC_CONFIG, self.config)
        write_json(self.selection, {
            "schema_version": 1,
            "name": "transient_rom_balanced2",
            "experiment_classification": {
                "non_formal": True,
                "paper_equivalent": False,
            },
            "sample_interval_ms": 2.0,
            "points": [
                {"workload": "matmul", "l1d_size": "64kB", "l2_size": "512kB"},
                {"workload": "stencil", "l1d_size": "64kB", "l2_size": "512kB"},
            ],
        })
        for workload in ("matmul", "stencil"):
            self.make_point(workload)
        fields = [
            "workload", "l1d_size", "l2_size", "state",
            "fixed_tmax_c", "clip3d_tmax_c",
            "fixed_frequency_ghz", "clip3d_frequency_ghz",
            "fixed_ipc2", "clip3d_ipc2", "fixed_bips2", "clip3d_bips2",
            "bips2_improvement_percent",
        ]
        with self.baseline.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for workload in ("matmul", "stencil"):
                writer.writerow({
                    "workload": workload,
                    "l1d_size": "64kB",
                    "l2_size": "512kB",
                    "state": "success",
                    "fixed_tmax_c": "100.0",
                    "clip3d_tmax_c": "99.5",
                    "fixed_frequency_ghz": "1.0",
                    "clip3d_frequency_ghz": "1.1",
                    "fixed_ipc2": "2.0",
                    "clip3d_ipc2": "2.1",
                    "fixed_bips2": "2.0",
                    "clip3d_bips2": "2.31",
                    "bips2_improvement_percent": "15.5",
                })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_point(self, workload: str) -> tuple[Path, Path]:
        canonical = (
            self.canonical / workload / "l1d_64kB" / "l2_512kB"
        )
        canonical.mkdir(parents=True)
        metadata = {
            "profile": "paper",
            "workload": workload,
            "l1d_size": "64kB",
            "l2_size": "512kB",
            "instruction_window_scope": "cpu0",
        }
        write_json(canonical / "r1_metadata.json", metadata)
        write_json(canonical / "status.json", {"state": "success"})
        (canonical / "stats.txt").write_text("canonical stats\n", encoding="utf-8")

        periodic = self.periodic / f"{workload}_64kB_512kB"
        periodic.mkdir(parents=True)
        periodic_metadata = dict(metadata)
        periodic_metadata.update({
            "transient_statistics": True,
            "transient_stats_mode": "cumulative",
            "sample_interval_ms": 2.0,
            "sample_interval_ticks": 2_000_000_000,
            "measurement_start_tick": 10,
            "measurement_end_tick": 20,
            "canonical_source_r1": str(canonical.resolve()),
        })
        write_json(periodic / "r1_metadata.json", periodic_metadata)
        write_json(periodic / "status.json", {
            "state": "success",
            "source_r1": str(canonical.resolve()),
            "sample_interval_ms": 2.0,
        })
        (periodic / "stats.txt").write_text("periodic stats\n", encoding="utf-8")
        return canonical, periodic

    def invoke(self) -> dict:
        return preflight_inputs(
            self.selection, self.canonical, self.periodic,
            self.baseline, self.config,
        )

    def test_exact_selection_and_paths_are_accepted(self) -> None:
        result = self.invoke()
        self.assertEqual([point["key"] for point in result["points"]], [
            "matmul/l1d_64kB/l2_512kB",
            "stencil/l1d_64kB/l2_512kB",
        ])
        self.assertEqual(result["sample_interval_ms"], 2.0)
        self.assertTrue(result["non_formal"])
        self.assertFalse(result["paper_equivalent"])
        self.assertEqual(
            Path(result["points"][0]["periodic_r1"]),
            (self.periodic / "matmul_64kB_512kB").resolve(),
        )
        self.assertEqual(set(result["input_sha256"]), {
            "selection", "config", "steady_baseline_csv",
        })

    def test_reordered_or_extra_selection_is_rejected(self) -> None:
        selection = read_json(self.selection)
        selection["points"].reverse()
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "exact ordered points"):
            self.invoke()
        selection["points"].append(selection["points"][0])
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "exact ordered points"):
            self.invoke()

    def test_non_successful_baseline_is_rejected(self) -> None:
        text = self.baseline.read_text(encoding="utf-8")
        self.baseline.write_text(
            text.replace("matmul,64kB,512kB,success", "matmul,64kB,512kB,failed"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "successful steady baseline"):
            self.invoke()

    def test_periodic_sampling_must_be_exactly_two_ms(self) -> None:
        metadata_path = self.periodic / "matmul_64kB_512kB/r1_metadata.json"
        metadata = read_json(metadata_path)
        metadata["sample_interval_ms"] = 10.0
        write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "compatible 2 ms periodic R1"):
            self.invoke()

    def test_periodic_source_identity_must_match_canonical(self) -> None:
        status_path = self.periodic / "matmul_64kB_512kB/status.json"
        status = read_json(status_path)
        status["source_r1"] = str((self.root / "other").resolve())
        write_json(status_path, status)
        with self.assertRaisesRegex(ValueError, "compatible 2 ms periodic R1"):
            self.invoke()

    def test_canonical_metadata_must_match_selected_key(self) -> None:
        metadata_path = self.canonical / "matmul/l1d_64kB/l2_512kB/r1_metadata.json"
        metadata = read_json(metadata_path)
        metadata["l1d_size"] = "32kB"
        write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "canonical R1 metadata"):
            self.invoke()

    def test_config_must_remain_exploratory(self) -> None:
        config = read_json(self.config)
        config["experiment_classification"]["paper_equivalent"] = True
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, "exploratory"):
            self.invoke()

    def test_boolean_sample_interval_is_rejected(self) -> None:
        selection = read_json(self.selection)
        selection["sample_interval_ms"] = True
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "sample_interval_ms"):
            self.invoke()


class Balanced2RunnerTests(Balanced2PreflightTests):
    def setUp(self) -> None:
        super().setUp()
        self.output = self.root / "output"
        self.calls: list[list[str]] = []

    @staticmethod
    def argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def write_pipeline_result(self, command: list[str]) -> None:
        output = Path(self.argument(command, "--output-dir"))
        source = Path(self.argument(command, "--r1-dir")).resolve()
        periodic = Path(self.argument(command, "--transient-rom-r1-dir")).resolve()
        config = Path(self.argument(command, "--config")).resolve()
        is_r2 = "--run-r2" in command
        workload = read_json(source / "r1_metadata.json")["workload"]
        rom = output / "transient_rom"
        package = (
            Path(self.argument(command, "--transient-rom-package-dir")).resolve()
            if is_r2 else rom / "rom_package"
        )
        if not is_r2:
            package.mkdir(parents=True, exist_ok=True)
            write_json(package / "rom_acceptance.json", {"accepted": True})
        branches = {}
        for branch_name, frequency, ipc in (
            ("fixed_bin", 1.10, 2.0),
            ("clip3d", 1.12 if workload == "matmul" else 0.98, 2.1),
        ):
            branches[branch_name] = {
                "branch": "fixed-bin" if branch_name == "fixed_bin" else "clip3d",
                "state": "validated" if not is_r2 else "success",
                "failure": None,
                "validation_classification": "validated",
                "validated_f_sus_trans_hotspot_ghz": frequency,
                "measured_ipc2": ipc if is_r2 else None,
                "measured_bips2_trans": ipc * frequency if is_r2 else None,
                "r2_requested": is_r2,
                "r2_executed": is_r2,
            }
        summary = {
            "schema_version": 1,
            "thermal_mode": "transient-rom",
            "non_formal": True,
            "paper_equivalent": False,
            "state": "success" if is_r2 else "validated",
            "source_r1": str(source),
            "config": str(config),
            "transient_r1": str(periodic),
            "sample_interval_ms": 2.0,
            "rom_package": str(package.resolve()),
            "rom_acceptance": {"accepted": True},
            "training_hotspot_calls": 8,
            "holdout_initialization_hotspot_calls": 2,
            "holdout_transient_hotspot_calls": 2,
            "calibration_hotspot_calls": 12,
            "optimizer_hotspot_calls": 0,
            "final_initialization_hotspot_calls": 4,
            "final_transient_hotspot_calls": 4,
            "final_validation_hotspot_calls": 8,
            "r2_requested": is_r2,
            "r2_executed": is_r2,
            "branches": branches,
        }
        write_json(rom / "transient_rom_summary.json", summary)
        if is_r2:
            paired_dir = rom / "final_validation"
            paired_dir.mkdir(parents=True, exist_ok=True)
            fixed_bips = branches["fixed_bin"]["measured_bips2_trans"]
            clip_bips = branches["clip3d"]["measured_bips2_trans"]
            write_json(paired_dir / "paired_comparison.json", {
                "schema_version": 1,
                "non_formal": True,
                "paper_equivalent": False,
                "fixed_bin": branches["fixed_bin"],
                "clip3d": branches["clip3d"],
                "bips2_trans_improvement_percent": (
                    (clip_bips / fixed_bips - 1.0) * 100.0
                ),
            })

    def fake_invoke(self, command: list[str], stdout: Path, stderr: Path) -> int:
        self.calls.append(command)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("synthetic success\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        self.write_pipeline_result(command)
        return 0

    def run_set(self, *, execute_r2: bool = False) -> dict:
        return run_validation_set(
            self.selection, self.canonical, self.periodic,
            self.baseline, self.config, self.output,
            execute_r2=execute_r2, invoke=self.fake_invoke,
        )

    def test_thermal_phase_validates_both_points_before_r2(self) -> None:
        result = self.run_set()
        self.assertEqual(result["state"], "thermal_validated")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all("--transient-rom-calibrate" in call for call in self.calls))
        self.assertTrue(all("--run-r2" not in call for call in self.calls))
        self.assertIn("matmul_64kB_512kB", " ".join(self.calls[0]))
        self.assertIn("stencil_64kB_512kB", " ".join(self.calls[1]))

    def test_valid_thermal_checkpoints_are_reused_after_validation(self) -> None:
        self.run_set()
        self.calls.clear()
        result = self.run_set()
        self.assertEqual(result["state"], "thermal_validated")
        self.assertEqual(self.calls, [])

    def test_malformed_existing_checkpoint_is_not_overwritten(self) -> None:
        point = self.output / "thermal/matmul_64kB_512kB/transient_rom"
        point.mkdir(parents=True)
        write_json(point / "transient_rom_summary.json", {"state": "validated"})
        with self.assertRaisesRegex(ValueError, "thermal checkpoint"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_r2_is_gated_by_both_thermal_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "both thermal"):
            self.run_set(execute_r2=True)
        self.assertEqual(self.calls, [])

        self.run_set()
        self.calls.clear()
        result = self.run_set(execute_r2=True)
        self.assertEqual(result["state"], "r2_validated")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all("--run-r2" in call for call in self.calls))
        self.assertTrue(all("--transient-rom-calibrate" not in call for call in self.calls))
        self.assertTrue(all("--transient-rom-package-dir" in call for call in self.calls))


if __name__ == "__main__":
    unittest.main()
