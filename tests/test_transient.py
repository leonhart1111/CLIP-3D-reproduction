from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.run_lifting_pipeline import boolean_text
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.run_hotspot_transient import parse_ttrace
from workflow.transient.run_transient_r1 import command_from_metadata
from workflow.transient.stats_windows import BEGIN, END, split_windows


class TransientStatisticsTests(unittest.TestCase):
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
                    "index": index, "duration_s": 0.01,
                    "modules": modules,
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
            self.assertEqual(
                len((root / "hotspot/power_transient.ptrace").read_text().splitlines()),
                3,
            )
            self.assertIn("-sampling_intvl 0.01",
                          (root / "hotspot/hotspot.config").read_text())

    def test_temperature_trace_times_start_after_first_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.ttrace"
            path.write_text("a b\n300 310\n320 315\n")
            samples = parse_ttrace(path, 0.01)
            self.assertEqual(samples[0]["time_s"], 0.01)
            self.assertEqual(samples[1]["peak_unit"], "a")
            self.assertAlmostEqual(samples[1]["tmax_c"], 46.85)

    def test_boolean_flag_is_explicit_and_defaults_can_remain_false(self):
        self.assertTrue(boolean_text("true"))
        self.assertFalse(boolean_text("false"))
        with self.assertRaises(Exception):
            boolean_text("maybe")


if __name__ == "__main__":
    unittest.main()
