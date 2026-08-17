from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from workflow.analysis.select_short_roi import (
    CANDIDATE_TARGETS,
    ConvergenceLimits,
    collect_candidate,
    compare_candidate,
    select_measurement_target,
)
from workflow.common import read_json, write_json
from workflow.r1_protocol import (
    canonical_protocol,
    classify_legacy_protocol,
    protocol_id,
    require_protocol,
)
from workflow.transient.sampling import (
    measured_roi_duration,
    resolve_sampling_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHORT_CONFIG = (
    PROJECT_ROOT / "configs/experiments/r1_short_convergence.json"
)


def load_runner():
    path = PROJECT_ROOT / "scripts/run_r1_sweep.py"
    spec = importlib.util.spec_from_file_location("run_r1_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_short(profile: str, measure: int, scope: str = "all-cores") -> dict:
    return canonical_protocol({
        "profile": profile,
        "warmup_insts": 2_000_000,
        "measure_insts": measure,
        "instruction_window_scope": scope,
    })


class ShortProfileProtocolTests(unittest.TestCase):
    def test_short_profiles_are_all_core_and_isolated(self):
        experiment = json.loads(SHORT_CONFIG.read_text())
        expected = {
            "short_conv_1m": 1_000_000,
            "short_conv_2m": 2_000_000,
            "short_conv_5m": 5_000_000,
            "short_conv_10m": 10_000_000,
        }
        self.assertEqual(
            experiment.get("default_profile"), "short_conv_1m"
        )
        for name, target in expected.items():
            profile = experiment["profiles"][name]
            self.assertEqual(profile["warmup_insts"], 2_000_000)
            self.assertEqual(profile["measure_insts"], target)
            self.assertEqual(
                profile["instruction_window_scope"], "all-cores"
            )

    def test_canonical_round_trip_and_rejects_malformed(self):
        canonical = canonical_short("short_conv_1m", 1_000_000)
        self.assertEqual(
            canonical["instruction_window_scope"], "all-cores"
        )
        for broken in (
            {"profile": "p", "warmup_insts": 0,
             "measure_insts": 1, "instruction_window_scope": "all-cores"},
            {"profile": "p", "warmup_insts": 1,
             "measure_insts": 1, "instruction_window_scope": "cpu"},
        ):
            with self.assertRaises(ValueError):
                canonical_protocol(broken)

    def test_protocol_id_changes_with_scope_target_and_hashes(self):
        base = canonical_short("short_conv_1m", 1_000_000)
        self.assertNotEqual(
            protocol_id(base),
            protocol_id(canonical_short("short_conv_1m", 2_000_000)),
        )
        self.assertNotEqual(
            protocol_id(base),
            protocol_id(
                canonical_short("short_conv_1m", 1_000_000, "cpu0")
            ),
        )
        self.assertNotEqual(
            protocol_id(base),
            protocol_id(base, gem5_config_sha256="a" * 64),
        )
        self.assertNotEqual(
            protocol_id(base),
            protocol_id(base, gem5_binary_sha256="b" * 64),
        )
        self.assertTrue(protocol_id(base).startswith("sha256:"))

    def test_require_protocol_rejects_missing_and_wrong_family(self):
        with self.assertRaises(ValueError):
            require_protocol({})
        with self.assertRaisesRegex(ValueError, "family"):
            require_protocol(
                {"r1_protocol": canonical_short("short_conv_1m", 1_000_000)},
                family="other-family",
            )
        canonical = require_protocol(
            {"r1_protocol": canonical_short("short_conv_1m", 1_000_000)}
        )
        self.assertEqual(canonical["measure_insts"], 1_000_000)

    def test_legacy_classification_reads_historical_metadata(self):
        legacy = {
            "warmup_insts": 100_000_000,
            "measure_insts": 500_000_000,
        }
        self.assertEqual(
            classify_legacy_protocol(legacy), "legacy-paper-500m"
        )
        corrected = {
            "r1_protocol": canonical_short("short_conv_1m", 1_000_000),
        }
        self.assertEqual(classify_legacy_protocol(corrected), "corrected")

    def test_runner_reuse_requires_exact_protocol_id(self):
        runner = load_runner()
        job = runner.Job(
            workload="matmul", l1d_size="32kB", l2_size="512kB",
            profile="short_conv_1m", family="clip3d-r1",
            warmup_insts=2_000_000, measure_insts=1_000_000,
            instruction_window_scope="all-cores", options=None,
            protocol_id="sha256:expected",
            output_dir=Path("/tmp/example"),
        )
        self.assertTrue(runner.is_reusable(
            {"state": "success", "r1_protocol_id": "sha256:expected"}, job
        ))
        self.assertFalse(runner.is_reusable(
            {"state": "success", "r1_protocol_id": "sha256:old"}, job
        ))
        self.assertFalse(runner.is_reusable(
            {"state": "success"}, job
        ))
        self.assertFalse(runner.is_reusable(
            {"state": "failed", "r1_protocol_id": "sha256:expected"}, job
        ))


def write_candidate_r1(root: Path, measure: int,
                       *, scope: str = "all-cores",
                       warmup: int = 2_000_000) -> Path:
    r1 = root / f"r1_{measure}"
    r1.mkdir(parents=True, exist_ok=True)
    canonical = canonical_protocol({
        "profile": f"short_conv_{measure // 1_000_000}m",
        "warmup_insts": warmup,
        "measure_insts": measure,
        "instruction_window_scope": scope,
    })
    identity = protocol_id(canonical)
    write_json(r1 / "r1_metadata.json", {
        "workload": "matmul",
        "binary": "/tmp/matmul",
        "command": ["/tmp/matmul", "-n", "128", "-t", "4"],
        "l1i_size": "32kB", "l1d_size": "32kB", "l2_size": "512kB",
        "memory_size": "2GiB", "cpu_clock": "2GHz",
        "warmup_insts": warmup,
        "measure_insts": measure,
        "instruction_window_scope": scope,
        "r1_protocol": canonical,
        "r1_protocol_id": identity,
    })
    lines = []
    for core in range(4):
        lines.append(f"system.cpu{core}.commitStats0.numInsts {measure}")
        lines.append(f"system.cpu{core}.numCycles {measure * 2}")
        lines.append(
            f"system.l2.demandAccesses::cpu{core}.data {(core + 1) * 100}"
        )
    lines.append("simSeconds 0.0005")
    lines.append("simTicks 1000000")
    lines.append("simFreq 2000000000.0")
    (r1 / "stats.txt").write_text("\n".join(lines) + "\n")
    write_json(r1 / "status.json", {"state": "success"})
    return r1


def write_modules(path: Path) -> Path:
    modules = {
        "schema_version": 1,
        "modules": [
            {"name": f"core{core}_exec", "kind": "core_exec", "core": core,
             "area_mm2": 1.0, "dynamic_power_w": 0.8,
             "leakage_power_w": 0.2, "total_power_w": 1.0}
            for core in range(4)
        ] + [
            {"name": "shared_l2", "kind": "l2", "area_mm2": 1.0,
             "dynamic_power_w": 0.4, "leakage_power_w": 0.1,
             "total_power_w": 0.5},
        ],
    }
    write_json(path, modules)
    return path


class ConvergenceSelectionTests(unittest.TestCase):
    def test_collect_candidate_reads_convergence_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1 = write_candidate_r1(root, 1_000_000)
            modules = write_modules(root / "modules.json")
            candidate = collect_candidate(r1, modules)
        self.assertEqual(candidate["measure_insts"], 1_000_000)
        self.assertAlmostEqual(candidate["aggregate_ipc"], 2.0)
        self.assertAlmostEqual(candidate["total_power_w"], 4.5)
        self.assertEqual(candidate["per_core_l2_traffic"], [100, 200, 300, 400])

    def test_collect_candidate_rejects_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1 = write_candidate_r1(
                root, 1_000_000, scope="cpu0"
            )
            modules = write_modules(root / "modules.json")
            with self.assertRaisesRegex(ValueError, "all-cores"):
                collect_candidate(r1, modules)
            r1_cpu0 = write_candidate_r1(root, 1_000_000, scope="all-cores")
            metadata = read_json(r1_cpu0 / "r1_metadata.json")
            metadata["r1_protocol"]["warmup_insts"] = 100_000
            write_json(r1_cpu0 / "r1_metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "warmup"):
                collect_candidate(r1_cpu0, modules)

    def test_compare_uses_relative_and_tv_metrics(self):
        candidate = {
            "measure_insts": 1_000_000,
            "aggregate_ipc": 2.01,
            "total_power_w": 4.59,
            "module_total_power_w": [1.0, 1.0, 1.0, 1.0, 0.5],
            "per_core_l2_traffic": [100, 200, 300, 400],
        }
        reference = {
            "measure_insts": 10_000_000,
            "aggregate_ipc": 2.0,
            "total_power_w": 4.5,
            "module_names": ["a", "b", "c", "d", "e"],
            "module_total_power_w": [1.0, 1.0, 1.0, 1.0, 0.5],
            "per_core_l2_traffic": [100, 200, 300, 400],
        }
        candidate["module_names"] = reference["module_names"]
        comparison = compare_candidate(candidate, reference)
        self.assertTrue(comparison["passes"]["ipc"])
        self.assertTrue(comparison["passes"]["total_power"])
        self.assertAlmostEqual(
            comparison["metrics"]["module_distribution"], 0.0
        )

    def test_selects_shortest_only_when_suffix_passes(self):
        def passed(target):
            return {
                "passes": {
                    "ipc": True, "total_power": True,
                    "module_distribution": True,
                    "l2_traffic_distribution": True,
                }
            }

        def failed(target):
            return {
                "passes": {
                    "ipc": True, "total_power": True,
                    "module_distribution": False,
                    "l2_traffic_distribution": True,
                }
            }

        comparisons = {
            1_000_000: passed(1_000_000),
            2_000_000: passed(2_000_000),
            5_000_000: passed(5_000_000),
        }
        self.assertEqual(select_measurement_target(comparisons), 1_000_000)
        comparisons[2_000_000] = failed(2_000_000)
        self.assertEqual(select_measurement_target(comparisons), 5_000_000)

    def test_falls_back_to_10m(self):
        comparisons = {
            1_000_000: {"passes": {
                "ipc": True, "total_power": True,
                "module_distribution": True,
                "l2_traffic_distribution": True}},
            2_000_000: {"passes": {
                "ipc": True, "total_power": True,
                "module_distribution": True,
                "l2_traffic_distribution": True}},
            5_000_000: {"passes": {
                "ipc": False, "total_power": True,
                "module_distribution": True,
                "l2_traffic_distribution": True}},
        }
        self.assertEqual(select_measurement_target(comparisons), 10_000_000)


class TransientDerivationTests(unittest.TestCase):
    def write_roi_r1(self, root: Path, roi_ms: float) -> Path:
        r1 = root / "r1"
        r1.mkdir(parents=True)
        canonical = canonical_short("short_conv_1m", 1_000_000)
        write_json(r1 / "r1_metadata.json", {
            "warmup_insts": 2_000_000,
            "measure_insts": 1_000_000,
            "instruction_window_scope": "all-cores",
            "r1_protocol": canonical,
            "r1_protocol_id": protocol_id(canonical),
        })
        sim_seconds = roi_ms / 1000.0
        sim_ticks = int(round(sim_seconds * 2e9))
        (r1 / "stats.txt").write_text(
            "\n".join((
                "system.cpu0.commitStats0.numInsts 1000000",
                "system.cpu0.numCycles 1000000",
                f"simSeconds {sim_seconds!r}",
                f"simTicks {sim_ticks}",
                "simFreq 2000000000.0",
            )) + "\n"
        )
        return r1

    def test_derives_sub_2ms_interval_from_roi(self):
        with tempfile.TemporaryDirectory() as temporary:
            r1 = self.write_roi_r1(Path(temporary), 4.8)
            policy = resolve_sampling_policy(r1)
        self.assertEqual(policy["mode"], "roi-derived")
        self.assertAlmostEqual(policy["requested_interval_ms"], 0.096)
        self.assertEqual(policy["target_window_count"], 50)
        self.assertTrue(policy["policy_id"].startswith("sha256:"))

    def test_caps_at_max_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            r1 = self.write_roi_r1(Path(temporary), 100.0)
            policy = resolve_sampling_policy(r1)
        self.assertAlmostEqual(policy["requested_interval_ms"], 0.5)

    def test_explicit_equal_number_has_different_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            r1 = self.write_roi_r1(Path(temporary), 25.0)
            derived = resolve_sampling_policy(r1)
            explicit = resolve_sampling_policy(r1, override_ms=0.5)
        self.assertNotEqual(derived["policy_id"], explicit["policy_id"])
        self.assertEqual(explicit["mode"], "explicit-override")
        self.assertAlmostEqual(explicit["requested_interval_ms"], 0.5)

    def test_derived_mode_rejects_unidentified_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            r1 = Path(temporary) / "r1"
            r1.mkdir()
            write_json(r1 / "r1_metadata.json", {"measure_insts": 1_000_000})
            (r1 / "stats.txt").write_text("simSeconds 0.001\n")
            with self.assertRaises(ValueError):
                resolve_sampling_policy(r1)
            policy = resolve_sampling_policy(r1, override_ms=0.2)
        self.assertEqual(policy["mode"], "explicit-override")
        self.assertEqual(
            policy["r1_protocol_id"], "legacy-unidentified"
        )
        self.assertIn("source_roi_unavailable", policy)

    def test_measured_roi_cross_checks_ticks_and_frequency(self):
        with tempfile.TemporaryDirectory() as temporary:
            r1 = self.write_roi_r1(Path(temporary), 4.8)
            roi = measured_roi_duration(r1)
            self.assertAlmostEqual(roi["roi_duration_ms"], 4.8)
            self.assertEqual(
                roi["stats_sha256"],
                hashlib.sha256(
                    (r1 / "stats.txt").read_bytes()
                ).hexdigest(),
            )
