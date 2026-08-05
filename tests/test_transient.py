from __future__ import annotations

import hashlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow.common import read_json, write_json
from workflow.run_lifting_pipeline import boolean_text
from workflow.transient.compare_layouts import compare_layout_results
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.generate_hotspot_trace import write_trace
from workflow.transient.run_hotspot_transient import (
    parse_ttrace,
    run_hotspot_transient,
    summarize_temperature_samples,
)
from workflow.transient.run_dual_layout_validation import run_dual_layout_validation
from workflow.transient.run_transient_r1 import command_from_metadata
from workflow.transient.run_transient_pipeline import prepare_power_windows
from workflow.transient.run_transient_pipeline import run_transient_pipeline
from workflow.transient.run_windowed_mcpat import run_windows
from workflow.transient.stats_windows import BEGIN, END, split_windows
from workflow.transient.validation import (
    summarize_power_windows,
    validate_power_triplet,
    validate_window_timeline,
)


class TransientStatisticsTests(unittest.TestCase):
    @staticmethod
    def timeline(windows: list[dict]) -> dict:
        return {
            "nominal_sample_interval_ticks": 10,
            "nominal_sample_interval_ms": 10.0,
            "measurement_start_tick": 100,
            "measurement_end_tick": 100 + sum(
                int(window["end_tick"]) - int(window["start_tick"])
                for window in windows
            ),
            "windows": windows,
        }

    def test_power_validation_rejects_invalid_power_values(self):
        with self.assertRaisesRegex(ValueError, "dynamic.*leakage.*total"):
            validate_power_triplet(
                {"dynamic_power_w": 2.0, "leakage_power_w": 1.0,
                 "total_power_w": 4.0}, "bad"
            )
        with self.assertRaises(ValueError):
            validate_power_triplet(
                {"dynamic_power_w": -0.001, "leakage_power_w": 1.0,
                 "total_power_w": 0.999}, "negative"
            )
        with self.assertRaises(ValueError):
            validate_power_triplet(
                {"dynamic_power_w": float("inf"), "leakage_power_w": 1.0,
                 "total_power_w": float("inf")}, "infinite"
            )

    def test_power_summary_is_duration_weighted(self):
        windows = [
            {"start_tick": 0, "end_tick": 10, "duration_s": 0.01,
             "totals": {"dynamic_power_w": 8.0, "leakage_power_w": 2.0,
                        "total_power_w": 10.0}},
            {"start_tick": 10, "end_tick": 15, "duration_s": 0.005,
             "totals": {"dynamic_power_w": 3.0, "leakage_power_w": 1.0,
                        "total_power_w": 4.0}},
        ]
        summary = summarize_power_windows(windows)
        self.assertAlmostEqual(summary["total_power_w"]["weighted_mean"], 8.0)

    def test_power_timeline_rejects_discontinuous_ticks(self):
        with self.assertRaisesRegex(ValueError, "window timeline gap"):
            validate_window_timeline(self.timeline([
                {"index": 0, "start_tick": 0, "end_tick": 10,
                 "duration_s": 0.01},
                {"index": 1, "start_tick": 11, "end_tick": 20,
                 "duration_s": 0.009},
            ]))

    def test_timeline_requires_at_least_two_windows(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_window_timeline(self.timeline([
                {"index": 0, "start_tick": 100, "end_tick": 105,
                 "duration_s": 0.005},
            ]))

    def test_timeline_rejects_internal_partial_window(self):
        with self.assertRaisesRegex(ValueError, "non-final.*nominal"):
            validate_window_timeline(self.timeline([
                {"index": 0, "start_tick": 100, "end_tick": 105,
                 "duration_s": 0.005},
                {"index": 1, "start_tick": 105, "end_tick": 115,
                 "duration_s": 0.01},
            ]))

    def test_timeline_rejects_overlong_final_window(self):
        with self.assertRaisesRegex(ValueError, "final.*nominal"):
            validate_window_timeline(self.timeline([
                {"index": 0, "start_tick": 100, "end_tick": 110,
                 "duration_s": 0.01},
                {"index": 1, "start_tick": 110, "end_tick": 121,
                 "duration_s": 0.011},
            ]))

    def test_timeline_uses_independent_roi_end_tick(self):
        manifest = self.timeline([
            {"index": 0, "start_tick": 100, "end_tick": 110,
             "duration_s": 0.01},
            {"index": 1, "start_tick": 110, "end_tick": 115,
             "duration_s": 0.005},
        ])
        manifest["measurement_end_tick"] = 116
        with self.assertRaisesRegex(ValueError, "measurement end tick"):
            validate_window_timeline(manifest)

    def test_cumulative_sections_become_delta_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "r1"
            source.mkdir()
            write_json(source / "r1_metadata.json", {
                "transient_statistics": True,
                "transient_stats_mode": "cumulative",
                "canonical_source_r1": str((root / "canonical-r1").resolve()),
                "sample_interval_ticks": 10,
                "sample_interval_s": 0.01,
                "measurement_start_tick": 100,
                "measurement_end_tick": 120,
            })
            sections = [
                {"finalTick": 90, "simFreq": 1000,
                 "system.cpu0.numCycles": 90,
                 "system.cpu0.commitStats0.numInsts": 45},
                {"finalTick": 110, "simFreq": 1000,
                 "system.cpu0.numCycles": 10,
                 "system.cpu0.commitStats0.numInsts": 5},
                {"finalTick": 120, "simFreq": 1000,
                 "system.cpu0.numCycles": 21,
                 "system.cpu0.commitStats0.numInsts": 12},
                {"finalTick": 120, "simFreq": 1000,
                 "system.cpu0.numCycles": 21,
                 "system.cpu0.commitStats0.numInsts": 12},
            ]
            text = []
            for section in sections:
                text.append(BEGIN)
                text.extend(f"{name} {value}" for name, value in section.items())
                text.append(END)
            (source / "stats.txt").write_text("\n".join(text) + "\n")
            result = split_windows(source, root / "windows")
            self.assertEqual(result["window_count"], 2)
            self.assertEqual(result["timeline_audit"]["window_count"], 2)
            self.assertEqual(result["timeline_audit"]["total_duration_s"], 0.02)
            self.assertEqual(result["timeline_audit"]["first_tick"], 100)
            self.assertEqual(result["timeline_audit"]["last_tick"], 120)
            self.assertEqual(
                result["timeline_audit"]["independent_roi_duration_ticks"], 20
            )
            self.assertEqual(
                read_json(root / "windows/windows_manifest.json")["timeline_audit"],
                result["timeline_audit"],
            )
            self.assertEqual(len(result["dropped_sections"]), 2)
            second = (root / "windows/window_0001/stats.txt").read_text()
            self.assertIn("system.cpu0.numCycles", second)
            self.assertIn(" 11\n", second)
            self.assertIn("system.cpu0.commitStats0.numInsts", second)
            self.assertIn(" 7\n", second)

    def test_transient_r1_command_uses_separate_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            write_json(source / "r1_metadata.json", {
                "workload": "matmul",
                "binary": "/tmp/matmul",
                "command": ["/tmp/matmul", "-n", "16", "-t", "4"],
                "l1i_size": "32kB", "l1d_size": "32kB", "l2_size": "512kB",
                "memory_size": "2GiB", "cpu_clock": "2GHz",
                "warmup_insts": 1, "measure_insts": 2,
                "instruction_window_scope": "cpu0",
                "latencies": {},
            })
            command = command_from_metadata(
                source, root / "output", Path("/tmp/gem5.opt"), 10.0
            )
            self.assertIn("clip_r1_transient.py", " ".join(command))
            self.assertIn("10.0", command)
            self.assertNotIn("--outdir=" + str(source), command)


class TransientTraceTests(unittest.TestCase):
    @staticmethod
    def model() -> dict:
        return {
            "schema_version": 1,
            "modules": [
                {"name": "core0_logic", "kind": "core_logic", "core": 0,
                 "area_mm2": 1.0, "dynamic_power_w": 1.0,
                 "leakage_power_w": 0.1, "total_power_w": 1.1},
                {"name": "core1_logic", "kind": "core_logic", "core": 1,
                 "area_mm2": 1.0, "dynamic_power_w": 1.0,
                 "leakage_power_w": 0.1, "total_power_w": 1.1},
                {"name": "core2_logic", "kind": "core_logic", "core": 2,
                 "area_mm2": 1.0, "dynamic_power_w": 1.0,
                 "leakage_power_w": 0.1, "total_power_w": 1.1},
                {"name": "core3_logic", "kind": "core_logic", "core": 3,
                 "area_mm2": 1.0, "dynamic_power_w": 1.0,
                 "leakage_power_w": 0.1, "total_power_w": 1.1},
                {"name": "shared_l2", "kind": "l2", "area_mm2": 1.0,
                 "dynamic_power_w": 0.2, "leakage_power_w": 0.1,
                 "total_power_w": 0.3},
                {"name": "noc", "kind": "interconnect", "area_mm2": 0.1,
                 "dynamic_power_w": 0.1, "leakage_power_w": 0.01,
                 "total_power_w": 0.11},
            ],
        }

    def test_power_windows_emit_one_row_each(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            windows = []
            for index, scale in enumerate((1.0, 2.0)):
                modules = []
                for source in model["modules"]:
                    module = dict(source)
                    module["dynamic_power_w"] *= scale
                    module["total_power_w"] = (
                        module["dynamic_power_w"] + module["leakage_power_w"]
                    )
                    modules.append(module)
                windows.append({
                    "index": index, "start_tick": index * 10,
                    "end_tick": (index + 1) * 10, "duration_s": 0.01,
                    "duration_ticks": 10,
                    "source_stats_sha256": f"sha256:stats-{index}",
                    "modules": modules,
                    "totals": {
                        field: sum(module[field] for module in modules)
                        for field in ("dynamic_power_w", "leakage_power_w",
                                      "total_power_w")
                    },
                })
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "power_provenance": {
                    "dynamic": "McPAT Runtime Dynamic",
                    "subthreshold_leakage": "McPAT Subthreshold Leakage",
                    "gate_leakage": "McPAT Gate Leakage",
                    "postprocessing": "none",
                },
                "run_settings": {
                    "dynamic_scale": 1.0, "leakage_scale": 1.0,
                },
                "windows": windows,
            })
            config = {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
            }
            from workflow.floorplan.generate_hotspot_inputs import baseline_layout
            write_json(root / "layout.json", baseline_layout(model))
            result = materialize_trace(
                root / "modules.json", root / "layout.json",
                root / "power_windows.json", root / "hotspot", config,
            )
            self.assertEqual(result["window_count"], 2)
            self.assertAlmostEqual(
                result["power_summary"]["total_power_w"]["weighted_mean"],
                6.96,
            )
            self.assertLessEqual(result["maximum_grid_residual_w"], 1e-10)
            self.assertEqual(
                len((root / "hotspot/power_transient.ptrace").read_text().splitlines()),
                3,
            )
            self.assertIn("-sampling_intvl 0.01",
                          (root / "hotspot/hotspot.config").read_text())
            self.assertEqual(result["mode"], "operational transient validation")
            self.assertTrue(result["non_formal"])
            self.assertFalse(result["paper_equivalent"])
            self.assertTrue(result["acceptance_checks"]["all_passed"])
            self.assertEqual(result["acceptance_checks"]["failure_reasons"], [])
            self.assertEqual(
                result["raw_power_evidence"]["power_provenance"],
                {
                    "dynamic": "McPAT Runtime Dynamic",
                    "subthreshold_leakage": "McPAT Subthreshold Leakage",
                    "gate_leakage": "McPAT Gate Leakage",
                    "postprocessing": "none",
                },
            )
            self.assertNotIn("dynamic_scale", result["raw_power_evidence"])
            self.assertNotIn("leakage_scale", result["raw_power_evidence"])
            self.assertIn("power_conservation", result["conservation_evidence"])

    @staticmethod
    def windowed_mcpat_fixture(root: Path) -> tuple[Path, Path, dict, Path, dict]:
        source_r1 = root / "source-r1"
        source_r1.mkdir()
        write_json(source_r1 / "r1_metadata.json", {"workload": "matmul"})
        windows = []
        for index in range(2):
            window_dir = root / f"window-{index}"
            window_dir.mkdir()
            (window_dir / "stats.txt").write_text("stats\n", encoding="utf-8")
            windows.append({
                "index": index,
                "directory": str(window_dir),
                "stats_sha256": f"sha256:stats-{index}",
                "start_tick": index * 10,
                "end_tick": (index + 1) * 10,
                "duration_ticks": 10,
                "duration_s": 0.01,
                "is_partial": False,
            })
        manifest = {
            "source_r1": str(source_r1),
            "canonical_source_r1": str(source_r1),
            "nominal_sample_interval_ms": 10.0,
            "nominal_sample_interval_ticks": 10,
            "measurement_start_tick": 0,
            "measurement_end_tick": 20,
            "windows": windows,
        }
        manifest_path = root / "windows_manifest.json"
        write_json(manifest_path, manifest)
        mcpat = root / "mcpat"
        mcpat.write_text("synthetic executable\n", encoding="utf-8")
        parsed = {
            "processor": {
                "area_mm2": 1.0,
                "dynamic_power_w": 1.0,
                "subthreshold_leakage_w": 0.15,
                "gate_leakage_w": 0.05,
                "leakage_power_w": 0.2,
                "total_power_w": 1.2,
            },
            "modules": [{
                "name": "chip",
                "area_mm2": 1.0,
                "dynamic_power_w": 1.0,
                "subthreshold_leakage_w": 0.15,
                "gate_leakage_w": 0.05,
                "leakage_power_w": 0.2,
                "total_power_w": 1.2,
            }],
            "module_totals": {
                "area_mm2": 1.0,
                "dynamic_power_w": 1.0,
                "leakage_power_w": 0.2,
                "total_power_w": 1.2,
            },
            "checks": {},
        }
        return manifest_path, root / "output", {"mcpat": {}}, mcpat, parsed

    @staticmethod
    def run_synthetic_mcpat(
        manifest_path: Path, output: Path, config: dict, mcpat: Path, parsed: dict,
    ) -> tuple[dict, int]:
        with patch(
            "workflow.transient.run_windowed_mcpat.convert"
        ), patch(
            "workflow.transient.run_windowed_mcpat.parse_mcpat_text",
            side_effect=lambda _text: copy.deepcopy(parsed),
        ), patch(
            "workflow.transient.run_windowed_mcpat.subprocess.run",
            return_value=type("Process", (), {
                "returncode": 0,
                "stdout": "McPAT (version 1.3 results",
            })(),
        ) as mcpat_run:
            result = run_windows(manifest_path, output, config, mcpat)
        return result, mcpat_run.call_count

    def test_windowed_mcpat_records_raw_power_without_calibration(self):
        """Transient window records must expose direct, unscaled McPAT power."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.windowed_mcpat_fixture(Path(temporary))
            result, mcpat_runs = self.run_synthetic_mcpat(*fixture)

            for window in result["windows"]:
                self.assertNotIn("power_calibration", window)
                self.assertEqual(
                    window["power_provenance"],
                    {
                        "dynamic": "McPAT Runtime Dynamic",
                        "subthreshold_leakage": "McPAT Subthreshold Leakage",
                        "gate_leakage": "McPAT Gate Leakage",
                        "postprocessing": "none",
                    },
                )
                self.assertNotIn(
                    "power_calibration", read_json(Path(window["mcpat_json"]))
                )
            self.assertEqual(mcpat_runs, 2)
            self.assertNotIn("dynamic_scale", result["run_settings"])
            self.assertNotIn("leakage_scale", result["run_settings"])

    def test_windowed_mcpat_reuses_current_raw_cache_without_mcpat(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.windowed_mcpat_fixture(Path(temporary))
            created, mcpat_runs = self.run_synthetic_mcpat(*fixture)
            self.assertEqual(mcpat_runs, 2)

            with patch(
                "workflow.transient.run_windowed_mcpat.convert",
                side_effect=AssertionError("cache reuse must not convert input"),
            ), patch(
                "workflow.transient.run_windowed_mcpat.parse_mcpat_text",
                side_effect=AssertionError("cache reuse must not parse McPAT"),
            ), patch(
                "workflow.transient.run_windowed_mcpat.subprocess.run",
                side_effect=AssertionError("cache reuse must not invoke McPAT"),
            ):
                reused = run_windows(*fixture[:4])

            self.assertEqual(reused["windows"], created["windows"])

    def test_windowed_mcpat_regenerates_cache_without_raw_provenance(self):
        for provenance in (None, {"postprocessing": "calibrated"}):
            with self.subTest(provenance=provenance), tempfile.TemporaryDirectory() as temporary:
                fixture = self.windowed_mcpat_fixture(Path(temporary))
                created, mcpat_runs = self.run_synthetic_mcpat(*fixture)
                self.assertEqual(mcpat_runs, 2)
                output = fixture[1]
                for window in created["windows"]:
                    cached_path = output / f"window_{window['index']:04d}/window_power.json"
                    cached = read_json(cached_path)
                    if provenance is None:
                        cached.pop("power_provenance")
                    else:
                        cached["power_provenance"] = provenance
                    write_json(cached_path, cached)

                regenerated, mcpat_runs = self.run_synthetic_mcpat(*fixture)

                self.assertEqual(mcpat_runs, 2)
                self.assertEqual(
                    [window["power_provenance"] for window in regenerated["windows"]],
                    [created["power_provenance"]] * 2,
                )

    def test_power_windows_reject_corrupted_module_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            corrupted = [dict(module) for module in model["modules"]]
            corrupted[0]["total_power_w"] = 99.0
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "run_settings": {
                    "dynamic_scale": 1.0, "leakage_scale": 1.0,
                },
                "windows": [
                    {
                        "index": index,
                        "start_tick": index * 10,
                        "end_tick": (index + 1) * 10,
                        "duration_ticks": 10,
                        "duration_s": 0.01,
                        "source_stats_sha256": f"sha256:stats-{index}",
                        "modules": copy.deepcopy(corrupted),
                        "totals": {
                            field: sum(module[field] for module in corrupted)
                            for field in ("dynamic_power_w", "leakage_power_w",
                                          "total_power_w")
                        },
                    }
                    for index in range(2)
                ],
            })
            config = {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
            }
            from workflow.floorplan.generate_hotspot_inputs import baseline_layout
            write_json(root / "layout.json", baseline_layout(model))
            with self.assertRaisesRegex(ValueError, "dynamic_power_w.*total_power_w"):
                materialize_trace(
                    root / "modules.json", root / "layout.json",
                    root / "power_windows.json", root / "hotspot", config,
                )

    def test_power_windows_reject_aggregate_mismatch_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            modules = [dict(module) for module in model["modules"]]
            windows = []
            for index in range(2):
                windows.append({
                    "index": index,
                    "start_tick": index * 10,
                    "end_tick": (index + 1) * 10,
                    "duration_ticks": 10,
                    "duration_s": 0.01,
                    "source_stats_sha256": f"sha256:stats-{index}",
                    "modules": copy.deepcopy(modules),
                    "totals": {
                        "dynamic_power_w": 1.0,
                        "leakage_power_w": 0.5,
                        "total_power_w": 1.5,
                    },
                })
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "run_settings": {
                    "dynamic_scale": 1.0,
                    "leakage_scale": 1.0,
                },
                "windows": windows,
            })
            config = {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
            }
            from workflow.floorplan.generate_hotspot_inputs import baseline_layout
            write_json(root / "layout.json", baseline_layout(model))
            output_dir = root / "hotspot"

            with self.assertRaisesRegex(ValueError, "aggregate module power"):
                materialize_trace(
                    root / "modules.json", root / "layout.json",
                    root / "power_windows.json", output_dir, config,
                )

            self.assertFalse(output_dir.exists())

    def test_module_name_mismatch_creates_no_trace_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            modules = [dict(module) for module in model["modules"]]
            modules[-1]["name"] = "wrong-name"
            totals = {
                field: sum(module[field] for module in modules)
                for field in ("dynamic_power_w", "leakage_power_w", "total_power_w")
            }
            windows = [
                {
                    "index": index,
                    "start_tick": index * 10,
                    "end_tick": (index + 1) * 10,
                    "duration_ticks": 10,
                    "duration_s": 0.01,
                    "source_stats_sha256": f"sha256:stats-{index}",
                    "modules": copy.deepcopy(modules),
                    "totals": totals,
                }
                for index in range(2)
            ]
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "run_settings": {
                    "dynamic_scale": 1.0,
                    "leakage_scale": 1.0,
                },
                "windows": windows,
            })
            config = {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
            }
            from workflow.floorplan.generate_hotspot_inputs import baseline_layout
            write_json(root / "layout.json", baseline_layout(model))
            output_dir = root / "hotspot"

            with self.assertRaisesRegex(ValueError, "module mismatch"):
                materialize_trace(
                    root / "modules.json", root / "layout.json",
                    root / "power_windows.json", output_dir, config,
                )

            self.assertFalse(output_dir.exists())

    def test_power_trace_serialization_round_trips_conserving_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "power.ptrace"
            values = [1.2345678901234567, 0.10000000000000002]
            write_trace(path, ["a", "b"], [values])
            round_trip = [float(value) for value in path.read_text().splitlines()[1].split()]
            self.assertEqual(round_trip, values)
            self.assertEqual(sum(round_trip), sum(values))

    def test_power_identity_ignores_elapsed_and_paths_but_detects_science(self):
        from workflow.transient.validation import power_trace_identity

        modules = [
            {
                "name": "core0",
                "dynamic_power_w": 2.0,
                "leakage_power_w": 1.0,
                "total_power_w": 3.0,
            }
        ]
        manifest = {
            "source_windows": "/machine-a/windows.json",
            "elapsed_seconds": 12.5,
            "nominal_sample_interval_ms": 10.0,
            "nominal_sample_interval_ticks": 10,
            "measurement_start_tick": 100,
            "measurement_end_tick": 120,
            "run_settings": {
                "mcpat_settings": {"temperature_k": 320},
                "opt_for_clk": 0,
                "dynamic_scale": 1.0,
                "leakage_scale": 1.0,
                "mcpat_binary_path": "/machine-a/tools/mcpat",
            },
            "windows": [
                {
                    "index": index,
                    "source_stats": f"/machine-a/window-{index}/stats.txt",
                    "source_stats_sha256": f"sha256:stats-{index}",
                    "mcpat_json": f"/machine-a/window-{index}/mcpat.json",
                    "start_tick": 100 + index * 10,
                    "end_tick": 110 + index * 10,
                    "duration_ticks": 10,
                    "duration_s": 0.01,
                    "is_partial": False,
                    "modules": copy.deepcopy(modules),
                    "totals": {
                        "dynamic_power_w": 2.0,
                        "leakage_power_w": 1.0,
                        "total_power_w": 3.0,
                    },
                }
                for index in range(2)
            ],
        }
        relocated = copy.deepcopy(manifest)
        relocated["source_windows"] = "/machine-b/windows.json"
        relocated["elapsed_seconds"] = 99.0
        relocated["run_settings"]["mcpat_binary_path"] = "/machine-b/tools/mcpat"
        for index, window in enumerate(relocated["windows"]):
            window["source_stats"] = f"/machine-b/window-{index}/stats.txt"
            window["mcpat_json"] = f"/machine-b/window-{index}/mcpat.json"

        self.assertEqual(
            power_trace_identity(manifest), power_trace_identity(relocated)
        )

        changed = copy.deepcopy(relocated)
        changed["windows"][0]["modules"][0].update({
            "dynamic_power_w": 2.5,
            "total_power_w": 3.5,
        })
        changed["windows"][0]["totals"].update({
            "dynamic_power_w": 2.5,
            "total_power_w": 3.5,
        })
        self.assertNotEqual(
            power_trace_identity(manifest), power_trace_identity(changed)
        )

    def test_discontinuous_power_window_timeline_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            windows = []
            for index, (start_tick, end_tick) in enumerate(((0, 10), (11, 20))):
                modules = [dict(module) for module in model["modules"]]
                windows.append({
                    "index": index, "start_tick": start_tick,
                    "end_tick": end_tick, "duration_s": 0.01,
                    "duration_ticks": end_tick - start_tick,
                    "source_stats_sha256": f"sha256:stats-{index}",
                    "modules": modules,
                    "totals": {
                        field: sum(module[field] for module in modules)
                        for field in ("dynamic_power_w", "leakage_power_w",
                                      "total_power_w")
                    },
                })
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "run_settings": {
                    "dynamic_scale": 1.0, "leakage_scale": 1.0,
                },
                "windows": windows,
            })
            config = {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
            }
            from workflow.floorplan.generate_hotspot_inputs import baseline_layout
            write_json(root / "layout.json", baseline_layout(model))
            output_dir = root / "hotspot"
            with self.assertRaisesRegex(ValueError, "window timeline gap"):
                materialize_trace(
                    root / "modules.json", root / "layout.json",
                    root / "power_windows.json", output_dir, config,
                )
            self.assertFalse(output_dir.exists())

    def test_temperature_trace_times_start_after_first_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.ttrace"
            path.write_text("a b\n300 310\n320 315\n")
            samples = parse_ttrace(path, 0.01)
            self.assertEqual(samples[0]["time_s"], 0.01)
            self.assertEqual(samples[1]["peak_unit"], "a")
            self.assertAlmostEqual(samples[1]["tmax_c"], 46.85)

    def test_temperature_summary_separates_initial_trace_and_final(self):
        samples = [
            {"index": 0, "time_s": 0.01, "peak_unit": "a",
             "tmax_c": 90.0, "tavg_c": 70.0},
            {"index": 1, "time_s": 0.02, "peak_unit": "b",
             "tmax_c": 95.0, "tavg_c": 72.0},
            {"index": 2, "time_s": 0.03, "peak_unit": "a",
             "tmax_c": 92.0, "tavg_c": 71.0},
        ]
        result = summarize_temperature_samples(
            samples, {"time_s": 0.0, "peak_unit": "initial", "tmax_c": 100.0}
        )
        self.assertEqual(result["trace_min_peak"]["time_s"], 0.01)
        self.assertEqual(result["trace_peak"]["time_s"], 0.02)
        self.assertEqual(result["final_peak"]["tmax_c"], 92.0)
        self.assertEqual(result["overall_peak"]["peak_unit"], "initial")
        self.assertEqual(result["trace_peak_minus_initial_c"], -5.0)
        self.assertEqual(result["final_minus_initial_c"], -8.0)

    def test_thermal_result_has_standard_classification_and_acceptance_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hotspot = root / "hotspot"
            hotspot.write_text("synthetic executable")
            write_json(root / "transient_trace_manifest.json", {
                "sample_interval_s": 0.01,
                "window_count": 2,
                "timeline_audit": {
                    "window_count": 2,
                    "total_duration_s": 0.02,
                },
                "raw_power_evidence": {
                    "dynamic_scale": 1.0,
                    "leakage_scale": 1.0,
                },
                "conservation_evidence": {
                    "maximum_grid_residual_w": 0.0,
                },
            })
            write_json(root / "hotspot_manifest.json", {"ambient_c": 25.0})
            steady = root / "steady.txt"
            steady.write_text("unit 300\n")

            def fake_hotspot(*_args, **kwargs):
                cwd = Path(kwargs["cwd"])
                (cwd / "transient.ttrace").write_text(
                    "unit\n301.000000\n302.000000\n"
                )
                return type("Process", (), {"returncode": 0, "stdout": "ok"})()

            with patch(
                "workflow.transient.run_hotspot_transient.subprocess.run",
                side_effect=fake_hotspot,
            ):
                result = run_hotspot_transient(
                    root, hotspot=hotspot, steady_source=steady
                )

            self.assertEqual(result["mode"], "operational transient validation")
            self.assertTrue(result["non_formal"])
            self.assertFalse(result["paper_equivalent"])
            self.assertTrue(result["acceptance_checks"]["all_passed"])
            self.assertEqual(result["acceptance_checks"]["failure_reasons"], [])
            for field in (
                "initial_peak", "trace_min_peak", "trace_peak", "final_peak",
                "overall_peak",
            ):
                self.assertIn(field, result)
            csv_text = (root / "transient_summary.csv").read_text(encoding="utf-8")
            self.assertIn("27.850000", csv_text)
            self.assertIn("28.850000", csv_text)

    def test_sampling_limitation_uses_actual_interval(self):
        from workflow.transient.validation import sampling_resolution_limitation

        message = sampling_resolution_limitation(2.0)
        self.assertIn("2 ms averaging", message)
        self.assertNotIn("10 ms averaging", message)

    def test_boolean_flag_is_explicit_and_defaults_can_remain_false(self):
        self.assertTrue(boolean_text("true"))
        self.assertFalse(boolean_text("false"))
        with self.assertRaises(Exception):
            boolean_text("maybe")


class TransientComparisonTests(unittest.TestCase):
    @staticmethod
    def identity(path: Path) -> str:
        from workflow.transient.validation import power_trace_identity
        return power_trace_identity(read_json(path))

    @staticmethod
    def file_hash(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def steady(path: Path, layout: str) -> None:
        write_json(path / "modules.json", {})
        write_json(path / "hotspot/layout.json", {})
        (path / "hotspot/steady.txt").write_text("unit 300\n")
        write_json(path / "pipeline_summary.json", {
            "layout_method": layout, "workload": "matmul", "tmax_c": 90.0,
        })

    @staticmethod
    def canonical_metadata() -> dict:
        return {
            "stage": "CLIP-3D R1",
            "workload": "matmul",
            "binary": "/tmp/matmul",
            "command": ["/tmp/matmul", "-n", "16", "-t", "4"],
            "environment": [],
            "stdin": None,
            "num_cores": 4,
            "cpu_type": "X86O3CPU",
            "cpu_clock": "2GHz",
            "issue_width": 4,
            "rob_entries": 192,
            "l1i_size": "32kB",
            "l1d_size": "32kB",
            "l1_associativity": 2,
            "l2_size": "512kB",
            "l2_associativity": 8,
            "cache_line_bytes": 64,
            "memory_size": "2GiB",
            "latencies": {},
            "warmup_insts_cpu0": 1,
            "measure_insts_cpu0": 2,
            "warmup_insts": 1,
            "measure_insts": 2,
            "instruction_window_scope": "cpu0",
            "stop_anchor": "CPU0 thread 0",
            "thread_mapping": "synthetic",
        }

    @staticmethod
    def operational_config() -> dict:
        return {
            "schema_version": 1,
            "name": "constrained_5p0_raw_power_p1_operational",
            "mcpat": {
                "temperature_k": 320,
                "device_type": 0,
                "longer_channel_device": 1,
                "interconnect_projection_type": 1,
                "opt_for_clk": 0,
            },
            "frequency": {"ambient_c": 25.0},
            "physical": {
                "tiers": 2,
                "grid_size": 4,
                "utilization": 0.7,
                "r_convec_k_per_w": 5.0,
                "thermal_stack": {
                    "silicon_resistivity_mk_per_w": 0.01,
                    "tim_resistivity_mk_per_w": 0.25,
                    "interposer_thickness_m": 0.0001,
                    "active_silicon_thickness_m": 0.00005,
                    "tim_thickness_m": 0.00002,
                    "local_resistance_scale": 8.72,
                },
            },
        }

    @classmethod
    def provenance_inputs(cls, root: Path) -> tuple[Path, Path, Path, Path]:
        source = root / "source"
        fixed = root / "fixed"
        clip3d = root / "clip3d"
        config_path = root / "config.json"
        source.mkdir(parents=True)
        metadata = cls.canonical_metadata()
        write_json(source / "r1_metadata.json", metadata)
        write_json(source / "status.json", {"state": "success"})
        (source / "stats.txt").write_text("canonical stats\n")
        config = cls.operational_config()
        write_json(config_path, config)
        raw_provenance = {
            "dynamic": "McPAT Runtime Dynamic",
            "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
            "postprocessing": "none",
        }
        modules = TransientTraceTests.model()
        modules.update({
            "source_r1": str(source.resolve()),
            "architecture": metadata,
            "power_provenance": raw_provenance,
            "power_calibration": None,
        })
        from workflow.floorplan.generate_hotspot_inputs import baseline_layout
        layout = baseline_layout(modules)
        stack = {
            **config["physical"]["thermal_stack"],
            "silicon_resistance_scale": 1.0,
            "tim_resistance_scale": 1.0,
            "effective_silicon_resistivity_mk_per_w": 0.0872,
            "effective_tim_resistivity_mk_per_w": 2.18,
        }
        total_power = sum(
            module["total_power_w"] for module in modules["modules"]
        )
        for path, layout_method in ((fixed, "fixed-bin"), (clip3d, "clip3d")):
            (path / "hotspot").mkdir(parents=True)
            (path / "mcpat").mkdir(parents=True)
            pilot_modules = copy.deepcopy(modules)
            pilot_modules["source_mcpat"] = str((path / "mcpat/mcpat.json").resolve())
            write_json(path / "modules.json", pilot_modules)
            write_json(path / "hotspot/layout.json", layout)
            (path / "hotspot/steady.txt").write_text("unit 300\n")
            write_json(path / "hotspot/hotspot_manifest.json", {
                "schema_version": 1,
                "ambient_c": 25.0,
                "r_convec_k_per_w": 5.0,
                "grid_size": 4,
                "thermal_stack": stack,
            })
            write_json(path / "mcpat/mcpat.json", {
                "schema_version": 1,
                "power_provenance": raw_provenance,
                "power_calibration": None,
                "modules": copy.deepcopy(modules["modules"]),
            })
            write_json(path / "run_config.json", {
                "schema_version": 1,
                "source": str(config_path.resolve()),
                "layout_method": layout_method,
                "config": config,
            })
            write_json(path / "pipeline_summary.json", {
                "schema_version": 2,
                "r1": str(source.resolve()),
                "output": str(path.resolve()),
                "experiment": config["name"],
                "workload": "matmul",
                "l1d_size": "32kB",
                "l2_size": "512kB",
                "layout_method": layout_method,
                "layout_mode": layout_method,
                "cooling": {"r_convec_k_per_w": 5.0, "ambient_c": 25.0},
                "module_count": len(modules["modules"]),
                "total_power_w": total_power,
                "power_provenance": raw_provenance,
                "tmax_c": 90.0,
                "artifacts": {
                    "config": str((path / "run_config.json").resolve()),
                    "modules": str((path / "modules.json").resolve()),
                    "layout": str((path / "hotspot/layout.json").resolve()),
                    "hotspot_manifest": str(
                        (path / "hotspot/hotspot_manifest.json").resolve()
                    ),
                    "mcpat_json": str((path / "mcpat/mcpat.json").resolve()),
                },
            })
        return source, fixed, clip3d, config_path

    @staticmethod
    def summary(root: Path, layout: str, steady: float, trace: float,
                final: float, peak_time: float, sample_ms: float = 10.0) -> dict:
        sample_s = sample_ms / 1000.0
        total_s = 2.0 * sample_s
        power_windows = root / "shared/windows/mcpat/power_windows.json"
        if not power_windows.is_file():
            write_json(power_windows, {
                "nominal_sample_interval_ms": sample_ms,
                "nominal_sample_interval_ticks": 10,
                "measurement_start_tick": 0,
                "measurement_end_tick": 20,
                "run_settings": {
                    "mcpat_settings": {"temperature_k": 320},
                    "opt_for_clk": 0,
                    "dynamic_scale": 1.0,
                    "leakage_scale": 1.0,
                },
                "windows": [
                    {
                        "index": 0, "start_tick": 0, "end_tick": 10,
                        "duration_ticks": 10, "duration_s": sample_s,
                        "source_stats_sha256": "sha256:stats-0",
                        "modules": [{
                            "name": "chip",
                            "dynamic_power_w": 8.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 10.0,
                        }],
                        "totals": {
                            "dynamic_power_w": 8.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 10.0,
                        },
                    },
                    {
                        "index": 1, "start_tick": 10, "end_tick": 20,
                        "duration_ticks": 10, "duration_s": sample_s,
                        "source_stats_sha256": "sha256:stats-1",
                        "modules": [{
                            "name": "chip",
                            "dynamic_power_w": 11.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 13.0,
                        }],
                        "totals": {
                            "dynamic_power_w": 11.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 13.0,
                        },
                    },
                ],
            })
        return {
            "mode": "operational transient validation",
            "non_formal": True,
            "paper_equivalent": False,
            "layout_method": layout,
            "source_r1": str((root / "source_r1").resolve()),
            "transient_r1": str((root / "shared_r1").resolve()),
            "sample_interval_ms": sample_ms,
            "window_count": 2,
            "actual_gem5_duration_s": total_s,
            "hotspot_trace_duration_s": total_s,
            "padded_final_duration_s": 0.0,
            "power_trace_identity": TransientComparisonTests.identity(power_windows),
            "power_summary": {
                "total_power_w": {
                    "minimum": 10.0,
                    "maximum": 13.0,
                    "weighted_mean": 11.5,
                    "peak_window_index": 1,
                },
            },
            "steady_tmax_c": steady,
            "temperature": {
                "initial_peak": {
                    "peak_unit": "initial", "tmax_c": steady, "time_s": 0.0,
                },
                "trace_min_peak": {
                    "peak_unit": "cell0", "tmax_c": trace - 1.0,
                    "time_s": sample_s,
                },
                "trace_peak": {"tmax_c": trace, "time_s": peak_time},
                "final_peak": {"tmax_c": final, "time_s": total_s},
                "overall_peak": {
                    "peak_unit": "cell1" if trace >= steady else "initial",
                    "tmax_c": max(trace, steady),
                    "time_s": peak_time if trace >= steady else 0.0,
                },
                "samples": [
                    {"index": 0, "time_s": sample_s, "tmax_c": trace - 1.0,
                     "tavg_c": trace - 4.0},
                    {"index": 1, "time_s": total_s, "tmax_c": final,
                     "tavg_c": final - 4.0},
                ],
            },
            "raw_power_evidence": {
                "dynamic_scale": 1.0,
                "leakage_scale": 1.0,
                "power_provenance": {
                    "dynamic": "McPAT Runtime Dynamic",
                    "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
                    "postprocessing": "none",
                },
            },
            "conservation_evidence": {
                "maximum_grid_residual_w": 0.0,
                "module_to_window_totals": True,
                "grid_conservation": True,
            },
            "acceptance_checks": {
                "checks": {"synthetic_branch": True},
                "all_passed": True,
                "failure_reasons": [],
            },
            "artifacts": {"power_windows": str(power_windows.resolve())},
        }

    def test_comparison_writes_deltas_and_deterministic_csv_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self.summary(root, "fixed-bin", 91.0, 95.0, 93.0, 0.02)
            clip3d = self.summary(root, "clip3d", 89.0, 93.5, 91.0, 0.01)

            result = compare_layout_results(fixed, clip3d, root / "comparison")

            self.assertAlmostEqual(
                result["temperature_c"]["trace_peak_clip_minus_fixed"], -1.5
            )
            self.assertAlmostEqual(
                result["temperature_c"]["steady_peak_clip_minus_fixed"], -2.0
            )
            self.assertAlmostEqual(
                result["temperature_c"]["final_peak_clip_minus_fixed"], -2.0
            )
            self.assertEqual(result["shared_input"]["window_count"], 2)
            self.assertAlmostEqual(
                result["timing_s"]["trace_peak_clip_minus_fixed"], -0.01
            )
            self.assertTrue(
                (root / "comparison/transient_comparison.json").is_file()
            )
            comparison_header = (
                root / "comparison/transient_comparison.csv"
            ).read_text().splitlines()[0]
            self.assertEqual(
                comparison_header,
                "layout,steady_peak_c,trace_peak_c,final_peak_c,"
                "trace_peak_time_s,power_peak_time_s,"
                "power_peak_to_temperature_peak_lag_s",
            )
            timeseries_header = (
                root / "comparison/power_temperature_timeseries.csv"
            ).read_text().splitlines()[0]
            self.assertEqual(
                timeseries_header,
                "index,time_s,total_power_w,fixed_peak_c,fixed_average_c,"
                "clip3d_peak_c,clip3d_average_c",
            )

    def test_comparison_carries_complete_audit_classification_and_limitations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self.summary(
                root, "fixed-bin", 91.0, 95.0, 93.0, 0.004, sample_ms=2.0
            )
            clip3d = self.summary(
                root, "clip3d", 89.0, 93.5, 91.0, 0.002, sample_ms=2.0
            )

            result = compare_layout_results(fixed, clip3d, root / "comparison")

            self.assertEqual(result["mode"], "operational transient validation")
            self.assertTrue(result["non_formal"])
            self.assertFalse(result["paper_equivalent"])
            self.assertTrue(result["acceptance_checks"]["all_passed"])
            self.assertEqual(result["acceptance_checks"]["failure_reasons"], [])
            for layout in ("fixed", "clip3d"):
                values = result["temperature_c"][layout]
                for field in (
                    "initial_peak", "trace_min_peak", "trace_peak", "final_peak",
                    "overall_peak", "trace_peak_minus_steady_c",
                ):
                    self.assertIn(field, values)
            self.assertIn("fixed", result["raw_power_evidence"])
            self.assertIn("clip3d", result["conservation_evidence"])
            limitations = " ".join(result["model_limitations"])
            self.assertIn("2 ms averaging", limitations)
            self.assertNotIn("10 ms averaging", limitations)
            self.assertIn("startup history", limitations)

    def test_comparison_rejects_mismatched_shared_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self.summary(root, "fixed-bin", 91.0, 95.0, 93.0, 0.02)
            mutations = {
                "source R1": ("source_r1", str(root / "other-source-r1")),
                "transient R1": ("transient_r1", str(root / "other-r1")),
                "sample interval": ("sample_interval_ms", 5.0),
                "window count": ("window_count", 3),
                "actual duration": ("actual_gem5_duration_s", 0.019),
                "power trace": ("power_trace_identity", "sha256:different"),
            }
            for message, (field, value) in mutations.items():
                with self.subTest(field=field):
                    clip3d = self.summary(
                        root, "clip3d", 89.0, 93.5, 91.0, 0.01
                    )
                    clip3d[field] = value
                    with self.assertRaisesRegex(ValueError, message):
                        compare_layout_results(fixed, clip3d, root / field)

    def test_comparison_rejects_power_trace_changed_after_layout_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self.summary(root, "fixed-bin", 91.0, 95.0, 93.0, 0.02)
            clip3d = self.summary(root, "clip3d", 89.0, 93.5, 91.0, 0.01)
            power_path = Path(fixed["artifacts"]["power_windows"])
            changed = read_json(power_path)
            changed["windows"][0]["totals"] = {
                "dynamic_power_w": 9.0,
                "leakage_power_w": 2.0,
                "total_power_w": 11.0,
            }
            changed["windows"][0]["modules"][0].update({
                "dynamic_power_w": 9.0,
                "total_power_w": 11.0,
            })
            write_json(power_path, changed)

            with self.assertRaisesRegex(ValueError, "power trace identity"):
                compare_layout_results(fixed, clip3d, root / "comparison")

    def test_partial_power_peak_uses_hotspot_sample_time_for_lag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = self.summary(root, "fixed-bin", 91.0, 95.0, 93.0, 0.02)
            clip3d = self.summary(root, "clip3d", 89.0, 93.5, 91.0, 0.02)
            power_path = Path(fixed["artifacts"]["power_windows"])
            power = read_json(power_path)
            power["measurement_end_tick"] = 15
            power["windows"][1]["end_tick"] = 15
            power["windows"][1]["duration_ticks"] = 5
            power["windows"][1]["duration_s"] = 0.005
            write_json(power_path, power)
            identity = self.identity(power_path)
            for summary in (fixed, clip3d):
                summary["power_trace_identity"] = identity
                summary["actual_gem5_duration_s"] = 0.015
                summary["padded_final_duration_s"] = 0.005

            result = compare_layout_results(fixed, clip3d, root / "comparison")

            self.assertAlmostEqual(result["timing_s"]["power_peak_time"], 0.02)
            self.assertAlmostEqual(
                result["timing_s"]["power_peak_gem5_end_time"], 0.015
            )
            self.assertAlmostEqual(
                result["timing_s"]["fixed_power_peak_to_temperature_peak_lag"],
                0.0,
            )

    def test_prepare_power_windows_materializes_shared_stages_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_r1 = root / "source-r1"
            transient_r1 = root / "transient-r1"
            source_r1.mkdir()
            transient_r1.mkdir()
            metadata = {
                "workload": "matmul", "binary": "/tmp/matmul",
                "command": ["/tmp/matmul"], "num_cores": 4,
                "cpu_clock": "2GHz", "l1i_size": "32kB",
                "l1d_size": "32kB", "l2_size": "512kB",
                "memory_size": "2GiB", "warmup_insts": 1,
                "measure_insts": 2, "instruction_window_scope": "cpu0",
                "latencies": {},
            }
            write_json(source_r1 / "r1_metadata.json", metadata)
            write_json(transient_r1 / "r1_metadata.json", {
                **metadata, "sample_interval_ms": 10.0,
            })

            def fake_split(_source: Path, output: Path) -> dict:
                result = {
                    "window_count": 2,
                    "canonical_source_r1": str(source_r1.resolve()),
                    "transient_r1": str(transient_r1.resolve()),
                    "nominal_sample_interval_ms": 10.0,
                    "windows": [],
                }
                write_json(output / "windows_manifest.json", result)
                return result

            def fake_mcpat(_manifest: Path, output: Path, _config: dict) -> dict:
                result = {
                    "window_count": 2,
                    "canonical_source_r1": str(source_r1.resolve()),
                    "transient_r1": str(transient_r1.resolve()),
                    "nominal_sample_interval_ms": 10.0,
                    "nominal_sample_interval_ticks": 10,
                    "measurement_start_tick": 100,
                    "measurement_end_tick": 115,
                    "power_provenance": {
                        "dynamic": "McPAT Runtime Dynamic",
                        "subthreshold_leakage": "McPAT Subthreshold Leakage",
                        "gate_leakage": "McPAT Gate Leakage",
                        "postprocessing": "none",
                    },
                    "run_settings": {
                        "mcpat_settings": {},
                        "opt_for_clk": 0,
                        "power_provenance": {
                            "dynamic": "McPAT Runtime Dynamic",
                            "subthreshold_leakage": "McPAT Subthreshold Leakage",
                            "gate_leakage": "McPAT Gate Leakage",
                            "postprocessing": "none",
                        },
                    },
                    "windows": [
                        {
                            "index": 0, "start_tick": 100, "end_tick": 110,
                            "duration_ticks": 10, "duration_s": 0.01,
                            "source_stats_sha256": "sha256:stats-0",
                            "modules": [{
                                "name": "chip", "dynamic_power_w": 2.0,
                                "leakage_power_w": 1.0, "total_power_w": 3.0,
                            }],
                            "totals": {
                                "dynamic_power_w": 2.0, "leakage_power_w": 1.0,
                                "total_power_w": 3.0,
                            },
                        },
                        {
                            "index": 1, "start_tick": 110, "end_tick": 115,
                            "duration_ticks": 5, "duration_s": 0.005,
                            "source_stats_sha256": "sha256:stats-1",
                            "modules": [{
                                "name": "chip", "dynamic_power_w": 1.0,
                                "leakage_power_w": 1.0, "total_power_w": 2.0,
                            }],
                            "totals": {
                                "dynamic_power_w": 1.0, "leakage_power_w": 1.0,
                                "total_power_w": 2.0,
                            },
                        },
                    ],
                }
                write_json(output / "power_windows.json", result)
                return result

            with patch(
                "workflow.transient.run_transient_pipeline.split_windows",
                side_effect=fake_split,
            ), patch(
                "workflow.transient.run_transient_pipeline.run_windows",
                side_effect=fake_mcpat,
            ):
                result = prepare_power_windows(
                    source_r1, transient_r1, root / "shared/windows", {}, 10.0
                )

            self.assertEqual(result["window_count"], 2)
            self.assertAlmostEqual(result["actual_gem5_duration_s"], 0.015)
            self.assertEqual(
                Path(result["power_windows"]),
                (root / "shared/windows/mcpat/power_windows.json").resolve(),
            )

    def test_dual_runner_records_status_and_prepares_shared_power_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed_steady, clip_steady, config = self.provenance_inputs(root)
            output = root / "output"
            stage_calls = []

            def fake_r1(source_dir: Path, output_dir: Path, sample_ms: float) -> dict:
                self.assertEqual(read_json(output / "status.json")["state"], "running")
                output_dir.mkdir(parents=True)
                write_json(output_dir / "status.json", {
                    "state": "success", "source_r1": str(source_dir.resolve()),
                    "sample_interval_ms": sample_ms,
                })
                write_json(output_dir / "r1_metadata.json", {
                    "workload": "matmul", "sample_interval_ms": sample_ms,
                })
                return read_json(output_dir / "status.json")

            def fake_prepare(source_dir: Path, transient_dir: Path,
                             windows_dir: Path, _config: dict,
                             sample_ms: float) -> dict:
                if stage_calls:
                    raise AssertionError("shared preprocessing ran more than once")
                stage_calls.append("prepare")
                power_windows = windows_dir / "mcpat/power_windows.json"
                write_json(power_windows, {
                    "nominal_sample_interval_ms": sample_ms,
                    "windows": [],
                })
                return {
                    "source_r1": str(source_dir.resolve()),
                    "transient_r1": str(transient_dir.resolve()),
                    "sample_interval_ms": sample_ms,
                    "window_count": 2,
                    "actual_gem5_duration_s": 0.02,
                    "hotspot_trace_duration_s": 0.02,
                    "padded_final_duration_s": 0.0,
                    "power_windows": str(power_windows.resolve()),
                    "power_trace_identity": "sha256:shared",
                    "stage_seconds": {},
                }

            def fake_layout(_source: Path, steady: Path, branch: Path,
                            _config: dict, power: Path,
                            _initial_temperature: str = "steady") -> dict:
                layout = "fixed-bin" if steady == fixed_steady.resolve() else "clip3d"
                result = self.summary(
                    root, layout, 91.0 if layout == "fixed-bin" else 89.0,
                    95.0 if layout == "fixed-bin" else 93.5,
                    93.0 if layout == "fixed-bin" else 91.0,
                    0.02 if layout == "fixed-bin" else 0.01,
                )
                result["output"] = str(branch)
                result["artifacts"]["power_windows"] = str(power.resolve())
                result["power_trace_identity"] = "sha256:shared"
                return result

            def fake_compare(fixed: dict, clip3d: dict, destination: Path) -> dict:
                self.assertEqual(destination, output.resolve() / "comparison")
                self.assertEqual(
                    fixed["artifacts"]["power_windows"],
                    clip3d["artifacts"]["power_windows"],
                )
                result = {"mode": "operational transient validation"}
                write_json(destination / "transient_comparison.json", result)
                return result

            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=fake_r1,
            ), patch(
                "workflow.transient.run_dual_layout_validation.prepare_power_windows",
                side_effect=fake_prepare,
            ), patch(
                "workflow.transient.run_dual_layout_validation.run_layout_thermal",
                side_effect=fake_layout,
            ), patch(
                "workflow.transient.run_dual_layout_validation.compare_layout_results",
                side_effect=fake_compare,
            ):
                result = run_dual_layout_validation(
                    source, fixed_steady, clip_steady, output, config, 10.0
                )

            self.assertEqual(stage_calls, ["prepare"])
            self.assertEqual(result["state"], "success")
            self.assertEqual(read_json(output / "status.json")["state"], "success")
            self.assertTrue((output / "experiment_summary.json").is_file())
            self.assertTrue(
                (output / "comparison/transient_comparison.json").is_file()
            )

    def test_dual_runner_failure_status_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed_steady, clip_steady, config = self.provenance_inputs(root)
            output = root / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=RuntimeError("synthetic R1 failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic R1 failure"):
                    run_dual_layout_validation(
                        source, fixed_steady, clip_steady, output, config, 10.0
                    )

            status = read_json(output / "status.json")
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["exception_type"], "RuntimeError")
            self.assertEqual(status["exception_message"], "synthetic R1 failure")
            self.assertEqual(sentinel.read_text(), "keep")

    def test_dual_runner_rejects_wrong_pilot_source_before_r1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed, clip3d, config = self.provenance_inputs(root)
            summary = read_json(fixed / "pipeline_summary.json")
            summary["r1"] = str((root / "different-r1").resolve())
            write_json(fixed / "pipeline_summary.json", summary)

            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=AssertionError("R1 must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "source R1"):
                    run_dual_layout_validation(
                        source, fixed, clip3d, root / "output", config, 10.0
                    )

    def test_dual_runner_rejects_non_raw_pilot_power_before_r1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed, clip3d, config = self.provenance_inputs(root)
            modules = read_json(clip3d / "modules.json")
            modules["power_calibration"] = {
                "dynamic_scale": 1.2,
                "leakage_scale": 1.0,
            }
            write_json(clip3d / "modules.json", modules)

            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=AssertionError("R1 must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "raw-power.*scale"):
                    run_dual_layout_validation(
                        source, fixed, clip3d, root / "output", config, 10.0
                    )

    def test_dual_runner_rejects_non_raw_summary_power_before_r1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed, clip3d, config = self.provenance_inputs(root)
            summary = read_json(clip3d / "pipeline_summary.json")
            summary["power_calibration"] = {
                "dynamic_scale": 1.2,
                "leakage_scale": 1.0,
            }
            write_json(clip3d / "pipeline_summary.json", summary)

            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=AssertionError("R1 must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "raw-power.*scale"):
                    run_dual_layout_validation(
                        source, fixed, clip3d, root / "output", config, 10.0
                    )

    def test_dual_runner_rejects_config_calibration_before_r1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, fixed, clip3d, config_path = self.provenance_inputs(root)
            config = read_json(config_path)
            config["mcpat"]["power_calibration"] = {
                "dynamic_scale": 1.0,
                "leakage_scale": 1.0,
            }
            write_json(config_path, config)
            for steady in (fixed, clip3d):
                run_config = read_json(steady / "run_config.json")
                run_config["config"] = config
                write_json(steady / "run_config.json", run_config)

            with patch(
                "workflow.transient.run_dual_layout_validation.run_transient_r1",
                side_effect=AssertionError("R1 must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "power_calibration"):
                    run_dual_layout_validation(
                        source, fixed, clip3d, root / "output", config_path, 10.0
                    )

    def test_dual_runner_rejects_output_overlapping_read_only_inputs_without_writes(self):
        relationships = ("equal", "ancestor", "descendant")
        for relationship in relationships:
            with self.subTest(relationship=relationship), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, fixed, clip3d, config = self.provenance_inputs(root / "inputs")
                sentinels = [
                    source / "r1_metadata.json",
                    fixed / "pipeline_summary.json",
                    clip3d / "pipeline_summary.json",
                ]
                before = {path: self.file_hash(path) for path in sentinels}
                if relationship == "equal":
                    output = fixed
                elif relationship == "ancestor":
                    output = root / "inputs"
                else:
                    output = source / "transient-output"

                with patch(
                    "workflow.transient.run_dual_layout_validation.run_transient_r1",
                    side_effect=AssertionError("R1 must not start"),
                ):
                    with self.assertRaisesRegex(ValueError, "read-only input"):
                        run_dual_layout_validation(
                            source, fixed, clip3d, output, config, 10.0
                        )

                self.assertEqual(
                    {path: self.file_hash(path) for path in sentinels}, before
                )
                if relationship == "descendant":
                    self.assertFalse(output.exists())
                elif relationship == "ancestor":
                    self.assertFalse((output / "status.json").exists())

    def test_single_runner_rejects_missing_steady_input_before_r1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            write_json(config, {})
            with patch(
                "workflow.transient.run_transient_pipeline.run_transient_r1",
                side_effect=AssertionError("R1 must not start"),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "steady pipeline"):
                    run_transient_pipeline(
                        root / "source", root / "missing-steady",
                        root / "output", config,
                    )


class TransientR1CacheTests(unittest.TestCase):
    @staticmethod
    def source(path: Path, workload: str = "matmul") -> dict:
        path.mkdir(parents=True)
        metadata = {
            **TransientComparisonTests.canonical_metadata(),
            "workload": workload,
        }
        write_json(path / "r1_metadata.json", metadata)
        write_json(path / "status.json", {"state": "success"})
        (path / "stats.txt").write_text("canonical stats\n")
        return metadata

    @staticmethod
    def cache(path: Path, source: Path, metadata: dict) -> None:
        path.mkdir(parents=True)
        write_json(path / "status.json", {
            "state": "success",
            "source_r1": str(source.resolve()),
            "sample_interval_ms": 10.0,
        })
        write_json(path / "r1_metadata.json", {
            **metadata,
            "transient_statistics": True,
            "transient_stats_mode": "cumulative",
            "sample_interval_ms": 10.0,
            "sample_interval_s": 0.01,
            "sample_interval_ticks": 10,
            "measurement_start_tick": 100,
            "measurement_end_tick": 120,
            "canonical_source_r1": str(source.resolve()),
        })
        (path / "stats.txt").write_text("transient stats\n")

    def test_wrapper_reuses_compatible_successful_shared_r1(self):
        from workflow.transient.run_transient_r1 import run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            metadata = self.source(source)
            cache = root / "shared-r1"
            self.cache(cache, source, metadata)
            gem5 = root / "gem5.opt"
            gem5.write_text("synthetic executable")

            with patch(
                "workflow.transient.run_transient_r1.subprocess.run",
                side_effect=AssertionError("compatible cache must be reused"),
            ):
                result = run(source, cache, 10.0, gem5=gem5)

            self.assertEqual(result["state"], "success")
            self.assertEqual(Path(result["source_r1"]), source.resolve())

    def test_wrapper_rejects_successful_cache_from_another_source(self):
        from workflow.transient.run_transient_r1 import run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            metadata = self.source(source)
            other_source = root / "other-source"
            self.source(other_source, workload="stencil")
            cache = root / "shared-r1"
            self.cache(cache, other_source, metadata)
            gem5 = root / "gem5.opt"
            gem5.write_text("synthetic executable")

            with patch(
                "workflow.transient.run_transient_r1.subprocess.run",
                side_effect=AssertionError("incompatible cache must not run"),
            ):
                with self.assertRaisesRegex(RuntimeError, "different source R1"):
                    run(source, cache, 10.0, gem5=gem5)


class HotSpotPrecisionPatchTests(unittest.TestCase):
    def test_hotspot_patch_tracks_six_decimal_temperature_outputs(self):
        patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches/hotspot/0001-six-decimal-temperature-output.patch"
        )
        text = patch_path.read_text(encoding="utf-8")
        self.assertNotIn('+    fprintf(fp, "%.2f', text)
        self.assertGreaterEqual(text.count("%.6f"), 11)
        self.assertIn("hotspot.c", text)
        self.assertIn("temperature_grid.c", text)
        self.assertIn("temperature_block.c", text)

        source_root = Path(__file__).resolve().parents[1] / "tools/src/hotspot"
        if not source_root.exists():
            return
        source_text = "\n".join(
            (source_root / filename).read_text(encoding="utf-8")
            for filename in (
                "hotspot.c",
                "temperature_grid.c",
                "temperature_block.c",
            )
        )
        machine_readable_lines = "\n".join(
            line for line in source_text.splitlines()
            if "fprintf(" in line and ("fp," in line or "grid_transient_fp," in line)
        )
        self.assertNotIn("%.2f", machine_readable_lines)


class TransientDocumentationTests(unittest.TestCase):
    def test_dual_layout_experiment_documents_operational_inputs(self):
        """Prevent losing the reproducible operational validation command."""
        document = (
            Path(__file__).resolve().parents[1] / "docs/transient_thermal_zh.md"
        ).read_text()

        for required in (
            "clip3d_constrained_5p0_raw_power_p1_operational.json",
            "runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB",
            "runs/operational_raw_power_p1/pilot_direct_20260731/fixed-bin",
            "runs/operational_raw_power_p1/pilot_direct_20260731/clip3d",
            "runs/transient_validation/matmul_32kB_512kB_10ms_20260803",
            "operational",
            "non-formal",
        ):
            self.assertIn(required, document)


if __name__ == "__main__":
    unittest.main()
