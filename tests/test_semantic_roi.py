from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from workflow.common import read_json, write_json
from workflow.analysis.summarize_sweep import summarize
from workflow.r1_catalog import build_catalogue
from workflow.r1_protocol import (
    SEMANTIC_SCOPE,
    canonical_protocol,
    protocol_id,
)
from workflow.r2.run_r2 import (
    _command_tail,
    _semantic_metrics,
    _validate_measurement,
    _validated_measurement,
)
from workflow.r2.semantic_comparison import compare_semantic_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_CONFIG = (
    PROJECT_ROOT / "configs/experiments/r1_semantic_cache_sweep.json"
)
LEGACY_CONFIG = PROJECT_ROOT / "configs/experiments/r1_cache_sweep.json"


def load_runner():
    path = PROJECT_ROOT / "scripts/run_r1_sweep.py"
    spec = importlib.util.spec_from_file_location("semantic_r1_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic_protocol(workload: str = "matmul") -> dict:
    declarations = {
        "matmul": (2, 4, "balanced-row-batch"),
        "stencil": (2, 4, "jacobi-iteration"),
    }
    warmup, measure, unit_type = declarations[workload]
    return canonical_protocol({
        "profile": "semantic",
        "instruction_window_scope": SEMANTIC_SCOPE,
        "warmup_work_units": warmup,
        "measure_work_units": measure,
        "work_unit_type": unit_type,
    })


def latency_vector() -> dict:
    overrides = {
        "l1i_tag_latency": 1,
        "l1i_data_latency": 1,
        "l1i_response_latency": 1,
        "l1d_tag_latency": 1,
        "l1d_data_latency": 1,
        "l1d_response_latency": 1,
        "l2_tag_latency": 8,
        "l2_data_latency": 8,
        "l2_response_latency": 8,
        "xbar_frontend_latency": 1,
        "xbar_forward_latency": 1,
        "xbar_response_latency": 1,
        "xbar_snoop_response_latency": 1,
    }
    arguments = []
    for key, value in overrides.items():
        arguments.extend((f"--{key.replace('_', '-')}", str(value)))
    return {"gem5_overrides": overrides, "gem5_args": arguments}


class SemanticProtocolTests(unittest.TestCase):
    def test_semantic_profile_is_isolated_from_historical_config(self):
        legacy = read_json(LEGACY_CONFIG)
        semantic = read_json(SEMANTIC_CONFIG)
        self.assertNotIn("semantic", legacy["profiles"])
        self.assertNotEqual(legacy.get("default_profile"), "semantic")
        self.assertEqual(semantic["default_profile"], "semantic")
        self.assertEqual(set(semantic["profiles"]), {"semantic"})

    def test_canonical_semantic_protocol_rejects_instruction_fields(self):
        canonical = semantic_protocol()
        self.assertEqual(canonical["marker_sequence"], ["workbegin", "workend"])
        self.assertEqual(canonical["marker_work_id"], 1)
        self.assertNotIn("measure_insts", canonical)
        with self.assertRaisesRegex(ValueError, "work_unit_type"):
            canonical_protocol({
                "profile": "semantic",
                "instruction_window_scope": SEMANTIC_SCOPE,
                "warmup_work_units": 2,
                "measure_work_units": 4,
            })

    def test_protocol_identity_binds_workload_binary(self):
        protocol = semantic_protocol()
        self.assertNotEqual(
            protocol_id(protocol, workload_binary_sha256="a" * 64),
            protocol_id(protocol, workload_binary_sha256="b" * 64),
        )

    def test_runner_emits_work_units_and_requires_binary_digest(self):
        runner = load_runner()
        experiment = read_json(SEMANTIC_CONFIG)
        arguments = SimpleNamespace(
            workloads=["matmul"], l1d_sizes=["32kB"], l2_sizes=["512kB"],
            profile="semantic", output_root=Path("/tmp/semantic-test"),
            gem5=Path("/tmp/gem5.opt"), r1_config=Path("/tmp/clip_r1.py"),
        )
        with self.assertRaisesRegex(ValueError, "hashed matmul binary"):
            runner.make_jobs(arguments, experiment, "c" * 64, "d" * 64, {})
        jobs = runner.make_jobs(
            arguments, experiment, "c" * 64, "d" * 64,
            {"matmul": "e" * 64},
        )
        self.assertEqual(len(jobs), 1)
        command = runner.command_for(jobs[0], arguments)
        self.assertIn("--warmup-work-units", command)
        self.assertIn("--measure-work-units", command)
        self.assertIn("--workload-binary-sha256", command)
        self.assertNotIn("--measure-insts", command)

    def test_r2_semantic_command_requires_binary_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "matmul"
            binary.write_bytes(b"semantic workload")
            metadata = {
                "workload": "matmul", "binary": str(binary),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "l1i_size": "32kB", "l1d_size": "32kB", "l2_size": "512kB",
                "instruction_window_scope": SEMANTIC_SCOPE,
                "warmup_work_units": 2, "measure_work_units": 4,
                "work_unit_type": "balanced-row-batch", "command": [str(binary)],
                "r1_protocol": semantic_protocol(),
                "r1_protocol_id": "sha256:protocol",
            }
            command = _command_tail(metadata, latency_vector())
            self.assertIn("--binary", command)
            self.assertIn("--measure-work-units", command)
            broken = dict(metadata)
            broken.pop("binary_sha256")
            with self.assertRaisesRegex(ValueError, "binary_sha256"):
                _command_tail(broken, latency_vector())
            binary.write_bytes(b"replaced workload")
            with self.assertRaisesRegex(ValueError, "differs from binary_sha256"):
                _command_tail(metadata, latency_vector())

    def test_r2_legacy_command_does_not_acquire_semantic_identity_flags(self):
        metadata = {
            "workload": "matmul", "binary": "/tmp/matmul",
            "binary_sha256": "a" * 64,
            "l1i_size": "32kB", "l1d_size": "32kB", "l2_size": "512kB",
            "instruction_window_scope": "cpu0",
            "warmup_insts_cpu0": 100, "measure_insts_cpu0": 200,
            "r1_protocol": {"family": "clip3d-r1", "profile": "paper"},
            "r1_protocol_id": "sha256:legacy", "command": ["/tmp/matmul"],
        }
        command = _command_tail(metadata, latency_vector())
        self.assertNotIn("--binary", command)
        self.assertNotIn("--workload-binary-sha256", command)
        self.assertNotIn("--r1-protocol-id", command)
        self.assertLess(command.index("--measure-insts"),
                        command.index("--instruction-window-scope"))


class SemanticEvidenceTests(unittest.TestCase):
    def write_roi(self, directory: Path, *, completion_ticks: int = 1000) -> None:
        write_json(directory / "roi_events.json", {
            "schema_version": 1,
            "scope": SEMANTIC_SCOPE,
            "work_unit_type": "balanced-row-batch",
            "warmup_work_units": 2,
            "measure_work_units": 4,
            "marker_work_id": 1,
            "events": [
                {"cause": "workbegin", "work_id": 1, "tick": 500},
                {"cause": "workend", "work_id": 1,
                 "tick": 500 + completion_ticks},
            ],
            "completion_ticks": completion_ticks,
        })

    def test_global_completion_cycles_come_from_marker_ticks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_roi(output)
            result = _semantic_metrics(
                {"simTicks": 1000, "simFreq": 1_000_000_000_000,
                 "system.cpu0.numCycles": 999999},
                {"cpu_clock": "2GHz", "warmup_work_units": 2,
                 "measure_work_units": 4,
                 "work_unit_type": "balanced-row-batch"},
                output,
            )
        self.assertEqual(result["completion_cycles"], 2.0)
        self.assertEqual(result["work_units_per_cycle"], 2.0)

    def test_catalogue_requires_untampered_semantic_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_path = root / "experiment.json"
            write_json(experiment_path, {
                "schema_version": 1,
                "workloads": ["matmul"],
                "l1d_sizes": ["32kB"],
                "l2_sizes": ["512kB"],
                "profiles": {"semantic": {
                    "instruction_window_scope": SEMANTIC_SCOPE,
                    "semantic_work_units": {"matmul": {
                        "warmup": 2, "measure": 4,
                        "type": "balanced-row-batch",
                    }},
                }},
            })
            point = root / "r1/matmul/l1d_32kB/l2_512kB"
            point.mkdir(parents=True)
            binary = root / "matmul"
            binary.write_bytes(b"semantic binary")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            protocol = semantic_protocol()
            identity = protocol_id(protocol, workload_binary_sha256=digest)
            write_json(point / "r1_metadata.json", {
                "workload": "matmul", "l1d_size": "32kB", "l2_size": "512kB",
                "instruction_window_scope": SEMANTIC_SCOPE,
                "warmup_work_units": 2, "measure_work_units": 4,
                "work_unit_type": "balanced-row-batch", "cpu_clock": "2GHz",
                "binary": str(binary), "binary_sha256": digest,
                "r1_protocol": protocol, "r1_protocol_id": identity,
            })
            write_json(point / "status.json", {
                "state": "success", "r1_protocol_id": identity,
            })
            stats = []
            for core in range(4):
                stats.extend((
                    f"system.cpu{core}.commitStats0.numInsts {100 + core}",
                    f"system.cpu{core}.numCycles {200 + core}",
                ))
            stats.extend(("simTicks 1000", "simFreq 1000000000000"))
            (point / "stats.txt").write_text("\n".join(stats) + "\n")
            self.write_roi(point)
            status = read_json(point / "status.json")
            status.update({
                "stats_sha256": hashlib.sha256(
                    (point / "stats.txt").read_bytes()
                ).hexdigest(),
                "r1_metadata_sha256": hashlib.sha256(
                    (point / "r1_metadata.json").read_bytes()
                ).hexdigest(),
                "roi_events_sha256": hashlib.sha256(
                    (point / "roi_events.json").read_bytes()
                ).hexdigest(),
            })
            write_json(point / "status.json", status)

            accepted = build_catalogue(
                root / "r1", experiment_path, profile="semantic"
            )
            self.assertTrue(accepted["complete"], accepted)
            record = accepted["canonical_records"][0]
            self.assertEqual(record["completion_cycles"], 2.0)

            evidence = read_json(point / "roi_events.json")
            evidence["events"][1]["tick"] += 1
            write_json(point / "roi_events.json", evidence)
            rejected = build_catalogue(
                root / "r1", experiment_path, profile="semantic"
            )
            self.assertFalse(rejected["complete"])
            self.assertTrue(any("marker ticks" in error
                                for error in rejected["canonical_records"][0]["errors"]))

    def test_cached_r2_instruction_vector_is_recomputed_from_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_roi(output)
            stats = {"simTicks": 1000, "simFreq": 1_000_000_000_000}
            lines = ["simTicks 1000", "simFreq 1000000000000"]
            for core in range(4):
                stats[f"system.cpu{core}.commitStats0.numInsts"] = 100 + core
                stats[f"system.cpu{core}.numCycles"] = 200 + core
                lines.extend((
                    f"system.cpu{core}.commitStats0.numInsts {100 + core}",
                    f"system.cpu{core}.numCycles {200 + core}",
                ))
            stats_path = output / "stats.txt"
            stats_path.write_text("\n".join(lines) + "\n")
            digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
            metadata = {
                "num_cores": 4,
                "cpu_clock": "2GHz",
                "instruction_window_scope": SEMANTIC_SCOPE,
                "warmup_work_units": 2,
                "measure_work_units": 4,
                "work_unit_type": "balanced-row-batch",
            }
            per_core, ipc2 = _validated_measurement(
                stats, metadata, SEMANTIC_SCOPE
            )
            semantic = _semantic_metrics(stats, metadata, output)
            result = {
                "stats": str(stats_path.resolve()),
                "stats_sha256": digest,
                "per_core": per_core,
                "ipc2": ipc2,
                "instruction_vector": [100, 101, 999, 103],
                **semantic,
            }
            status = {
                "stats": str(stats_path.resolve()),
                "stats_sha256": digest,
                "ipc2": ipc2,
                **semantic,
            }
            reasons = []
            _validate_measurement(
                result, status, metadata, output, reasons, SEMANTIC_SCOPE
            )
            self.assertIn(
                "R2 result instruction vector differs from cached stats", reasons
            )


class SemanticComparisonTests(unittest.TestCase):
    def result(self, vector: list[int], *, ipc: float, rate: float) -> dict:
        return {
            "instruction_window_scope": SEMANTIC_SCOPE,
            "r1_protocol_id": "sha256:protocol",
            "workload_binary_sha256": "a" * 64,
            "work_unit_type": "balanced-row-batch",
            "measure_work_units": 4,
            "instruction_vector": vector,
            "ipc2": ipc,
            "work_units_per_cycle": rate,
        }

    def test_ipc_is_allowed_only_for_an_exact_instruction_vector(self):
        fixed = self.result([100, 101, 102, 103], ipc=2.0, rate=0.1)
        same = compare_semantic_results(
            fixed, self.result([100, 101, 102, 103], ipc=2.2, rate=0.11)
        )
        self.assertTrue(same["same_trace"])
        self.assertAlmostEqual(same["ipc_improvement_percent"], 10.0)

        different = compare_semantic_results(
            fixed, self.result([100, 101, 103, 103], ipc=2.2, rate=0.11),
            2.0, 1.5,
        )
        self.assertFalse(different["ipc_comparison_allowed"])
        self.assertIsNone(different["ipc_improvement_percent"])
        self.assertEqual(different["primary_metric"], "work_units_per_ns")
        self.assertAlmostEqual(different["fixed_score"], 0.2)
        self.assertAlmostEqual(different["clip3d_score"], 0.165)

    def test_homogeneous_sweep_ranks_semantic_fixed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            point = {
                "workload": "matmul", "l1d_size": "32kB", "l2_size": "512kB",
                "layout_method": "fixed-bin",
                "cooling": {"r_convec_k_per_w": 5.0},
                "ipc1": 2.0, "tmax_c": 80.0,
                "sustainable_frequency_ghz": 1.5,
                "bips1_thermal": 3.0, "ipc2": 2.1, "bips2": 3.15,
                "primary_performance_metric": "work_units_per_ns",
                "work_units_per_cycle": 0.1, "work_units_per_ns": 0.15,
                "r2_critical_path_cycles": 12, "total_pipeline_seconds": 1.0,
            }
            write_json(root / "point/pipeline_summary.json", point)
            result = summarize(
                root, root / "summary.csv", root / "summary.json",
                expected_points=1,
            )
        self.assertEqual(result["primary_performance_metric"],
                         "work_units_per_ns")
        self.assertEqual(result["score_definition"],
                         "fixed semantic work throughput*f_sus")


if __name__ == "__main__":
    unittest.main()
