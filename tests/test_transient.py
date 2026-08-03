from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow.common import read_json, write_json
from workflow.run_lifting_pipeline import boolean_text
from workflow.transient.compare_layouts import compare_layout_results
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.run_hotspot_transient import (
    parse_ttrace,
    summarize_temperature_samples,
)
from workflow.transient.run_dual_layout_validation import run_dual_layout_validation
from workflow.transient.run_transient_r1 import command_from_metadata
from workflow.transient.run_transient_pipeline import prepare_power_windows
from workflow.transient.run_transient_pipeline import run_transient_pipeline
from workflow.transient.stats_windows import BEGIN, END, split_windows
from workflow.transient.validation import (
    summarize_power_windows,
    validate_power_triplet,
    validate_window_timeline,
)


class TransientStatisticsTests(unittest.TestCase):
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
            validate_window_timeline({"windows": [
                {"index": 0, "start_tick": 0, "end_tick": 10,
                 "duration_s": 0.01},
                {"index": 1, "start_tick": 11, "end_tick": 20,
                 "duration_s": 0.009},
            ]})

    def test_cumulative_sections_become_delta_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "r1"
            source.mkdir()
            write_json(source / "r1_metadata.json", {
                "transient_statistics": True,
                "transient_stats_mode": "cumulative",
                "sample_interval_ticks": 10,
                "sample_interval_s": 0.01,
                "measurement_start_tick": 100,
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
            self.assertEqual(result["timeline_audit"], {
                "window_count": 2,
                "total_duration_s": 0.02,
                "first_tick": 100,
                "last_tick": 120,
            })
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
                    "modules": modules,
                    "totals": {
                        field: sum(module[field] for module in modules)
                        for field in ("dynamic_power_w", "leakage_power_w",
                                      "total_power_w")
                    },
                })
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
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

    def test_power_windows_reject_corrupted_module_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            corrupted = [dict(module) for module in model["modules"]]
            corrupted[0]["total_power_w"] = 99.0
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
                "windows": [{
                    "index": 0, "start_tick": 0, "end_tick": 10,
                    "duration_s": 0.01, "modules": corrupted,
                    "totals": {
                        field: sum(module[field] for module in corrupted)
                        for field in ("dynamic_power_w", "leakage_power_w",
                                      "total_power_w")
                    },
                }],
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
                    "modules": modules,
                    "totals": {
                        field: sum(module[field] for module in modules)
                        for field in ("dynamic_power_w", "leakage_power_w",
                                      "total_power_w")
                    },
                })
            write_json(root / "power_windows.json", {
                "nominal_sample_interval_ms": 10.0,
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

    def test_boolean_flag_is_explicit_and_defaults_can_remain_false(self):
        self.assertTrue(boolean_text("true"))
        self.assertFalse(boolean_text("false"))
        with self.assertRaises(Exception):
            boolean_text("maybe")


class TransientComparisonTests(unittest.TestCase):
    @staticmethod
    def identity(path: Path) -> str:
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
    def summary(root: Path, layout: str, steady: float, trace: float,
                final: float, peak_time: float) -> dict:
        power_windows = root / "shared/windows/mcpat/power_windows.json"
        if not power_windows.is_file():
            write_json(power_windows, {
                "nominal_sample_interval_ms": 10.0,
                "windows": [
                    {
                        "index": 0, "start_tick": 0, "end_tick": 10,
                        "duration_s": 0.01,
                        "totals": {
                            "dynamic_power_w": 8.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 10.0,
                        },
                    },
                    {
                        "index": 1, "start_tick": 10, "end_tick": 20,
                        "duration_s": 0.01,
                        "totals": {
                            "dynamic_power_w": 11.0,
                            "leakage_power_w": 2.0,
                            "total_power_w": 13.0,
                        },
                    },
                ],
            })
        return {
            "layout_method": layout,
            "source_r1": str((root / "source_r1").resolve()),
            "transient_r1": str((root / "shared_r1").resolve()),
            "sample_interval_ms": 10.0,
            "window_count": 2,
            "actual_gem5_duration_s": 0.02,
            "hotspot_trace_duration_s": 0.02,
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
                "trace_peak": {"tmax_c": trace, "time_s": peak_time},
                "final_peak": {"tmax_c": final, "time_s": 0.02},
                "samples": [
                    {"index": 0, "time_s": 0.01, "tmax_c": trace - 1.0,
                     "tavg_c": trace - 4.0},
                    {"index": 1, "time_s": 0.02, "tmax_c": final,
                     "tavg_c": final - 4.0},
                ],
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
                    "nominal_sample_interval_ms": 10.0,
                    "windows": [],
                }
                write_json(output / "windows_manifest.json", result)
                return result

            def fake_mcpat(_manifest: Path, output: Path, _config: dict) -> dict:
                result = {
                    "window_count": 2,
                    "nominal_sample_interval_ms": 10.0,
                    "windows": [
                        {"duration_s": 0.01}, {"duration_s": 0.005},
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
            source = root / "source"
            fixed_steady = root / "fixed-steady"
            clip_steady = root / "clip-steady"
            output = root / "output"
            source.mkdir()
            fixed_steady.mkdir()
            clip_steady.mkdir()
            self.steady(fixed_steady, "fixed-bin")
            self.steady(clip_steady, "clip3d")
            config = root / "config.json"
            write_json(config, {})
            write_json(source / "r1_metadata.json", {"workload": "matmul"})
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
            source = root / "source"
            source.mkdir()
            output = root / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            config = root / "config.json"
            write_json(config, {})
            fixed_steady = root / "fixed"
            clip_steady = root / "clip"
            self.steady(fixed_steady, "fixed-bin")
            self.steady(clip_steady, "clip3d")

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
