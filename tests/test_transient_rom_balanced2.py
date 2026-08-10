#!/usr/bin/env python3
"""Regression tests for the two-point transient-ROM validation entry."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, sha256_file, write_json
from workflow.experiments.transient_rom_balanced2 import (
    preflight_inputs,
    run_validation_set,
    summarize_validation_set,
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

    @staticmethod
    def write_package_manifest(package: Path) -> None:
        classification = {
            "thermal_mode": "transient-rom",
            "non_formal": True,
            "paper_equivalent": False,
        }
        artifacts = []
        for path in sorted(package.rglob("*")):
            if not path.is_file() or path.name == "rom_artifact_manifest.json":
                continue
            artifacts.append({
                "path": path.relative_to(package).as_posix(),
                "sha256": sha256_file(path),
                "classification": dict(classification),
            })
        write_json(package / "rom_artifact_manifest.json", {
            "schema_version": 1,
            **classification,
            "scope": str(package.resolve()),
            "artifacts": artifacts,
        })

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
            write_json(package / "rom_acceptance.json", {
                "schema_version": 1,
                "thermal_mode": "transient-rom",
                "non_formal": True,
                "paper_equivalent": False,
                "accepted": True,
                "identity": {
                    "configuration_hash": "sha256:" + sha256_file(config),
                    "canonical_r1_metadata_hash": (
                        "sha256:" + sha256_file(source / "r1_metadata.json")
                    ),
                    "r1_input_hashes": {
                        "canonical_status": (
                            "sha256:" + sha256_file(source / "status.json")
                        ),
                        "canonical_metadata": (
                            "sha256:" + sha256_file(source / "r1_metadata.json")
                        ),
                        "canonical_stats": (
                            "sha256:" + sha256_file(source / "stats.txt")
                        ),
                        "periodic_status": (
                            "sha256:" + sha256_file(periodic / "status.json")
                        ),
                        "periodic_metadata": (
                            "sha256:" + sha256_file(periodic / "r1_metadata.json")
                        ),
                        "periodic_stats": (
                            "sha256:" + sha256_file(periodic / "stats.txt")
                        ),
                    },
                },
            })
            write_json(package / "calibration_manifest.json", {
                "schema_version": 1,
                "thermal_mode": "transient-rom",
                "non_formal": True,
                "paper_equivalent": False,
                "identity": read_json(package / "rom_acceptance.json")["identity"],
            })
            self.write_package_manifest(package)
        acceptance = read_json(package / "rom_acceptance.json")
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
            "rom_acceptance": acceptance,
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

    @staticmethod
    def verify_synthetic_package(package: Path, identity: dict, settings: object) -> dict:
        # The complete verifier has its own real-package regression suite. This
        # boundary double keeps the Balanced runner fixture small while still
        # enforcing the calibration-manifest identity consumed by this layer.
        acceptance = read_json(Path(package) / "rom_acceptance.json")
        manifest = read_json(Path(package) / "calibration_manifest.json")
        if manifest.get("identity") != identity:
            raise ValueError("reusable ROM calibration manifest evidence differs")
        return {"acceptance": acceptance}

    def run_set(self, *, execute_r2: bool = False) -> dict:
        with patch(
            "workflow.experiments.transient_rom_balanced2."
            "require_package_calibration_evidence",
            side_effect=self.verify_synthetic_package,
        ):
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

    def test_thermal_checkpoint_rejects_changed_config_bytes(self) -> None:
        # Break caught: a path-only checkpoint must not survive a scientific
        # configuration edit at that same path.
        self.run_set()
        config = read_json(self.config)
        config["balanced2_test_marker"] = "changed"
        write_json(self.config, config)
        self.calls.clear()

        with self.assertRaisesRegex(
            ValueError, "checkpoint.*identity|configuration hash"
        ):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_thermal_checkpoint_rejects_changed_r1_stats_bytes(self) -> None:
        # Break caught: a completed checkpoint must bind the exact canonical
        # and periodic statistics bytes used to produce it.
        self.run_set()
        stats = self.canonical / "matmul/l1d_64kB/l2_512kB/stats.txt"
        stats.write_text("changed canonical stats\n", encoding="utf-8")
        self.calls.clear()

        with self.assertRaisesRegex(
            ValueError, "checkpoint.*identity|R1 input hash identity"
        ):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_thermal_summary_config_path_cannot_be_rehashed_away(self) -> None:
        # Break caught: updating both a forged summary and its checkpoint hash
        # must not let the pipeline claim a different configuration path.
        self.run_set()
        point_output = self.output / "thermal/matmul_64kB_512kB"
        summary_path = point_output / "transient_rom/transient_rom_summary.json"
        summary = read_json(summary_path)
        summary["config"] = str((self.root / "other-config.json").resolve())
        write_json(summary_path, summary)
        checkpoint_path = point_output / "balanced2_checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        checkpoint["artifacts"]["summary_sha256"] = sha256_file(summary_path)
        write_json(checkpoint_path, checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_thermal_checkpoint_rejects_another_points_package(self) -> None:
        # Break caught: a self-consistently rehashed thermal checkpoint cannot
        # substitute the other selected point's accepted ROM package.
        self.run_set()
        point_output = self.output / "thermal/matmul_64kB_512kB"
        other = (
            self.output / "thermal/stencil_64kB_512kB/transient_rom/rom_package"
        ).resolve()
        summary_path = point_output / "transient_rom/transient_rom_summary.json"
        summary = read_json(summary_path)
        summary["rom_package"] = str(other)
        write_json(summary_path, summary)
        checkpoint_path = point_output / "balanced2_checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        checkpoint["artifacts"].update({
            "summary_sha256": sha256_file(summary_path),
            "rom_package": str(other),
            "rom_acceptance_sha256": sha256_file(other / "rom_acceptance.json"),
            "rom_manifest_sha256": sha256_file(
                other / "rom_artifact_manifest.json"
            ),
        })
        write_json(checkpoint_path, checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "thermal ROM package"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_thermal_checkpoint_rejects_uninventoried_package_file(self) -> None:
        # Break caught: hashing only the manifest JSON must not accept a live
        # package tree that contains an unrecorded artifact.
        self.run_set()
        package = (
            self.output / "thermal/matmul_64kB_512kB/transient_rom/rom_package"
        )
        (package / "unrecorded-model.bin").write_bytes(b"changed model")
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "manifest"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_changed_config_cannot_be_hidden_by_rewriting_checkpoint_identity(self) -> None:
        # Break caught: a self-declared checkpoint identity must be cross-bound
        # to the accepted ROM package's original configuration identity.
        self.run_set()
        config = read_json(self.config)
        config["balanced2_test_marker"] = "changed-and-rehashed"
        write_json(self.config, config)
        current = self.invoke()
        for point in current["points"]:
            checkpoint_path = (
                self.output / "thermal" / self._point_slug_for_test(point)
                / "balanced2_checkpoint.json"
            )
            checkpoint = read_json(checkpoint_path)
            checkpoint["scientific_identity"] = point["checkpoint_identity"]
            write_json(checkpoint_path, checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_changed_r1_cannot_be_hidden_by_rewriting_checkpoint_identity(self) -> None:
        # Break caught: the accepted ROM package, rather than only the mutable
        # outer checkpoint, must bind the raw R1 inputs used by the run.
        self.run_set()
        stats = self.canonical / "matmul/l1d_64kB/l2_512kB/stats.txt"
        stats.write_text("changed and synchronously rehashed stats\n", encoding="utf-8")
        current = self.invoke()
        for point in current["points"]:
            checkpoint_path = (
                self.output / "thermal" / self._point_slug_for_test(point)
                / "balanced2_checkpoint.json"
            )
            checkpoint = read_json(checkpoint_path)
            checkpoint["scientific_identity"] = point["checkpoint_identity"]
            write_json(checkpoint_path, checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "R1 input hash"):
            self.run_set()
        self.assertEqual(self.calls, [])

    @staticmethod
    def _point_slug_for_test(point: dict) -> str:
        return f"{point['workload']}_{point['l1d_size']}_{point['l2_size']}"

    def test_summary_acceptance_must_equal_live_package_acceptance(self) -> None:
        # Break caught: summary metadata cannot reinterpret an accepted package
        # after the package itself has been validated.
        self.run_set()
        point_output = self.output / "thermal/matmul_64kB_512kB"
        summary_path = point_output / "transient_rom/transient_rom_summary.json"
        summary = read_json(summary_path)
        summary["rom_acceptance"] = {**summary["rom_acceptance"], "marker": "forged"}
        write_json(summary_path, summary)
        checkpoint_path = point_output / "balanced2_checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        checkpoint["artifacts"]["summary_sha256"] = sha256_file(summary_path)
        write_json(checkpoint_path, checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_malformed_existing_checkpoint_is_not_overwritten(self) -> None:
        point = self.output / "thermal/matmul_64kB_512kB/transient_rom"
        point.mkdir(parents=True)
        write_json(point / "transient_rom_summary.json", {"state": "validated"})
        with self.assertRaisesRegex(ValueError, "thermal checkpoint"):
            self.run_set()
        self.assertEqual(self.calls, [])

    def test_early_failure_cannot_retry_in_same_output_root(self) -> None:
        # Break caught: a subprocess failure before point-output creation must
        # not let a retry truncate the first attempt's status and logs.
        def fail_before_output(command: list[str], stdout: Path, stderr: Path) -> int:
            self.calls.append(command)
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout.write_text("first-attempt stdout\n", encoding="utf-8")
            stderr.write_text("first-attempt stderr\n", encoding="utf-8")
            return 7

        with self.assertRaisesRegex(RuntimeError, "rc=7"):
            run_validation_set(
                self.selection, self.canonical, self.periodic,
                self.baseline, self.config, self.output,
                invoke=fail_before_output,
            )
        root_status = read_json(self.output / "status.json")
        self.assertEqual(root_status["state"], "failed")
        self.assertEqual(root_status["completed_points"], [])
        first_stdout = (
            self.output / "logs/thermal_matmul_64kB_512kB.stdout.log"
        ).read_text(encoding="utf-8")
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "existing attempt"):
            self.run_set()
        self.assertEqual(self.calls, [])
        self.assertEqual(
            (self.output / "logs/thermal_matmul_64kB_512kB.stdout.log")
            .read_text(encoding="utf-8"),
            first_stdout,
        )

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

    def test_r2_checkpoint_rejects_a_different_rom_package(self) -> None:
        # Break caught: an R2 result cannot be reused with any accepted package;
        # it must name the exact package produced by that point's thermal phase.
        self.run_set()
        self.run_set(execute_r2=True)
        summary_path = (
            self.output / "r2/matmul_64kB_512kB/transient_rom/"
            "transient_rom_summary.json"
        )
        summary = read_json(summary_path)
        other = self.output / "thermal/stencil_64kB_512kB/transient_rom/rom_package"
        summary["rom_package"] = str(other.resolve())
        write_json(summary_path, summary)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "ROM package"):
            self.run_set(execute_r2=True)
        self.assertEqual(self.calls, [])

    def test_r2_reuse_rejects_synchronously_rehashed_r1_checkpoints(self) -> None:
        # Break caught: even synchronously rewriting both checkpoint identities
        # and their cross-hash must not reuse measured R2 after raw R1 changes.
        self.run_set()
        self.run_set(execute_r2=True)
        stats = self.canonical / "matmul/l1d_64kB/l2_512kB/stats.txt"
        stats.write_text("changed before R2 reuse\n", encoding="utf-8")
        current = self.invoke()
        for point in current["points"]:
            slug = self._point_slug_for_test(point)
            thermal_checkpoint_path = (
                self.output / "thermal" / slug / "balanced2_checkpoint.json"
            )
            thermal_checkpoint = read_json(thermal_checkpoint_path)
            thermal_checkpoint["scientific_identity"] = point["checkpoint_identity"]
            write_json(thermal_checkpoint_path, thermal_checkpoint)

            r2_checkpoint_path = (
                self.output / "r2" / slug / "balanced2_checkpoint.json"
            )
            r2_checkpoint = read_json(r2_checkpoint_path)
            r2_checkpoint["scientific_identity"] = point["checkpoint_identity"]
            r2_checkpoint["artifacts"]["thermal_checkpoint_sha256"] = (
                sha256_file(thermal_checkpoint_path)
            )
            write_json(r2_checkpoint_path, r2_checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "both thermal") as raised:
            self.run_set(execute_r2=True)
        self.assertIsNotNone(raised.exception.__cause__)
        self.assertRegex(str(raised.exception.__cause__), "R1 input hash")
        self.assertEqual(self.calls, [])

    def test_r2_reuse_cross_checks_acceptance_against_calibration_manifest(self) -> None:
        # Break caught: rewriting acceptance, its inventory, both summaries,
        # and both checkpoints cannot sever the identity fixed at calibration.
        self.run_set()
        self.run_set(execute_r2=True)
        stats = self.canonical / "matmul/l1d_64kB/l2_512kB/stats.txt"
        stats.write_text("changed before acceptance rewrite\n", encoding="utf-8")
        point = self.invoke()["points"][0]
        slug = self._point_slug_for_test(point)
        package = self.output / "thermal" / slug / "transient_rom/rom_package"
        acceptance_path = package / "rom_acceptance.json"
        acceptance = read_json(acceptance_path)
        acceptance["identity"]["r1_input_hashes"] = {
            field: "sha256:" + point["checkpoint_identity"]["input_sha256"][field]
            for field in (
                "canonical_status", "canonical_metadata", "canonical_stats",
                "periodic_status", "periodic_metadata", "periodic_stats",
            )
        }
        write_json(acceptance_path, acceptance)
        self.write_package_manifest(package)
        package_manifest = package / "rom_artifact_manifest.json"

        thermal_output = self.output / "thermal" / slug
        thermal_summary_path = (
            thermal_output / "transient_rom/transient_rom_summary.json"
        )
        thermal_summary = read_json(thermal_summary_path)
        thermal_summary["rom_acceptance"] = acceptance
        write_json(thermal_summary_path, thermal_summary)
        thermal_checkpoint_path = thermal_output / "balanced2_checkpoint.json"
        thermal_checkpoint = read_json(thermal_checkpoint_path)
        thermal_checkpoint["scientific_identity"] = point["checkpoint_identity"]
        thermal_checkpoint["artifacts"].update({
            "summary_sha256": sha256_file(thermal_summary_path),
            "rom_acceptance_sha256": sha256_file(acceptance_path),
            "rom_manifest_sha256": sha256_file(package_manifest),
            "rom_acceptance_record": acceptance,
        })
        write_json(thermal_checkpoint_path, thermal_checkpoint)

        r2_output = self.output / "r2" / slug
        r2_summary_path = r2_output / "transient_rom/transient_rom_summary.json"
        r2_summary = read_json(r2_summary_path)
        r2_summary["rom_acceptance"] = acceptance
        write_json(r2_summary_path, r2_summary)
        r2_checkpoint_path = r2_output / "balanced2_checkpoint.json"
        r2_checkpoint = read_json(r2_checkpoint_path)
        r2_checkpoint["scientific_identity"] = point["checkpoint_identity"]
        r2_checkpoint["artifacts"].update({
            "summary_sha256": sha256_file(r2_summary_path),
            "rom_acceptance_sha256": sha256_file(acceptance_path),
            "rom_manifest_sha256": sha256_file(package_manifest),
            "rom_acceptance_record": acceptance,
            "thermal_checkpoint_sha256": sha256_file(thermal_checkpoint_path),
        })
        write_json(r2_checkpoint_path, r2_checkpoint)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "both thermal") as raised:
            self.run_set(execute_r2=True)
        self.assertIsNotNone(raised.exception.__cause__)
        self.assertRegex(str(raised.exception.__cause__), "calibration manifest")
        self.assertEqual(self.calls, [])

    def test_r2_checkpoint_rejects_paired_branches_that_differ_from_summary(self) -> None:
        # Break caught: a correct scalar improvement cannot hide a paired branch
        # payload copied from another run.
        self.run_set()
        self.run_set(execute_r2=True)
        paired_path = (
            self.output / "r2/matmul_64kB_512kB/transient_rom/"
            "final_validation/paired_comparison.json"
        )
        paired = read_json(paired_path)
        paired["fixed_bin"]["measured_ipc2"] = 99.0
        write_json(paired_path, paired)
        self.calls.clear()

        with self.assertRaisesRegex(ValueError, "paired.*branch"):
            self.run_set(execute_r2=True)
        self.assertEqual(self.calls, [])


class Balanced2ReportTests(Balanced2RunnerTests):
    def inputs(self) -> dict:
        return self.invoke()

    def summarize(self, *, require_r2: bool) -> dict:
        with patch(
            "workflow.experiments.transient_rom_balanced2."
            "require_package_calibration_evidence",
            side_effect=self.verify_synthetic_package,
        ):
            return summarize_validation_set(
                self.inputs(), self.output, require_r2=require_r2,
            )

    def test_thermal_report_keeps_measured_fields_empty(self) -> None:
        self.run_set()
        report = self.summarize(require_r2=False)
        self.assertEqual(report["state"], "thermal_validated")
        self.assertEqual(report["point_count"], 2)
        for row in report["points"]:
            self.assertIsNotNone(row["transient_fixed_frequency_ghz"])
            self.assertIsNotNone(row["transient_clip3d_frequency_ghz"])
            self.assertIsNone(row["transient_fixed_ipc2"])
            self.assertIsNone(row["transient_clip3d_bips2"])
            self.assertIsNone(row["transient_bips2_improvement_percent"])
        self.assertTrue((self.output / "summary.json").is_file())
        self.assertTrue((self.output / "summary.csv").is_file())

    def test_r2_report_recomputes_signed_gains_and_statistics(self) -> None:
        self.run_set()
        self.run_set(execute_r2=True)
        report = self.summarize(require_r2=True)
        self.assertEqual(report["state"], "r2_validated")
        self.assertEqual(report["transient_statistics"]["wins"], 1)
        self.assertEqual(report["transient_statistics"]["ties"], 0)
        self.assertEqual(report["transient_statistics"]["losses"], 1)
        matmul, stencil = report["points"]
        self.assertAlmostEqual(
            matmul["steady_bips2_improvement_percent"], 15.5,
        )
        self.assertGreater(matmul["transient_bips2_improvement_percent"], 0.0)
        self.assertLess(stencil["transient_bips2_improvement_percent"], 0.0)
        self.assertAlmostEqual(
            matmul["improvement_shift_percentage_points"],
            matmul["transient_bips2_improvement_percent"] - 15.5,
        )
        persisted = read_json(self.output / "summary.json")
        self.assertEqual(persisted, report)

    def test_csv_has_deterministic_header_and_six_decimal_temperatures(self) -> None:
        self.run_set()
        self.summarize(require_r2=False)
        with (self.output / "summary.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
            header = stream.seek(0) or stream.readline().strip()
        self.assertEqual(header.split(",")[:6], [
            "workload", "l1d_size", "l2_size", "state",
            "steady_fixed_tmax_c", "steady_clip3d_tmax_c",
        ])
        self.assertEqual(rows[0]["steady_fixed_tmax_c"], "100.000000")
        self.assertEqual(rows[0]["steady_clip3d_tmax_c"], "99.500000")

    def test_r2_report_rejects_thermal_only_predictions(self) -> None:
        self.run_set()
        with self.assertRaisesRegex(ValueError, "R2 checkpoint"):
            self.summarize(require_r2=True)


if __name__ == "__main__":
    unittest.main()
