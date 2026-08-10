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
from workflow.experiments.transient_rom_balanced2 import preflight_inputs


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


if __name__ == "__main__":
    unittest.main()
