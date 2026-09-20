import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflow.thermal.linear_response import (
    LinearThermalResponse,
    fit_central_difference,
    parse_ptrace,
    soft_peak,
)
from workflow.thermal.build_linear_response import _module_source_map
from workflow.thermal.validate_linear_response import _candidate_document, _candidate_trace
from workflow.thermal.linear_search import (
    evaluate_candidates,
    pareto_front,
    rank_for_pruning,
)


class LinearResponseTests(unittest.TestCase):
    def setUp(self):
        self.power = np.array([1.0, 2.0])
        self.matrix = np.array([[2.0, 0.5], [0.25, 1.5]])
        self.intercept = np.array([25.0, 30.0])
        self.temperature = self.intercept + self.matrix @ self.power

    def test_central_difference_recovers_affine_operator(self):
        delta = 0.1
        records = []
        for index in range(2):
            plus = self.power.copy()
            minus = self.power.copy()
            plus[index] += delta
            minus[index] -= delta
            records.append({
                "source_index": index,
                "delta_w": delta,
                "positive_temperature_c": (self.intercept + self.matrix @ plus).tolist(),
                "negative_temperature_c": (self.intercept + self.matrix @ minus).tolist(),
            })
        response = fit_central_difference(
            ("p0", "p1"), ("t0", "t1"), self.power, self.temperature,
            records, {"contract": "synthetic"},
        )
        np.testing.assert_allclose(response.response_k_per_w, self.matrix)
        np.testing.assert_allclose(response.intercept_c, self.intercept)
        np.testing.assert_allclose(response.predict([2.0, 3.0]),
                                   self.intercept + self.matrix @ [2.0, 3.0])
        self.assertLess(response.metadata["positive_reconstruction_max_error_c"], 1e-12)

    def test_sensitivity_is_soft_peak_weighted_response(self):
        response = LinearThermalResponse(
            ("p0", "p1"), ("t0", "t1"), self.power, self.temperature,
            self.matrix, self.intercept, {},
        )
        weights = np.exp((self.temperature - max(self.temperature)))
        weights /= weights.sum()
        np.testing.assert_allclose(response.sensitivity(tau=1.0), weights @ self.matrix)

    def test_save_and_load_preserve_hash_bound_response(self):
        response = LinearThermalResponse(
            ("p0", "p1"), ("t0", "t1"), self.power, self.temperature,
            self.matrix, self.intercept, {"contract": "synthetic"},
        )
        with tempfile.TemporaryDirectory() as directory:
            arrays, manifest = response.save(Path(directory) / "response")
            loaded = LinearThermalResponse.load(manifest)
            np.testing.assert_allclose(loaded.response_k_per_w, self.matrix)
            self.assertTrue(arrays.is_file())

    def test_ptrace_parser_rejects_misaligned_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.ptrace"
            path.write_text("p0 p1\n1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different lengths"):
                parse_ptrace(path)

    def test_module_input_source_map_uses_trace_module_names(self):
        """Module-input HotSpot cases must not be interpreted as grid cells."""
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "layout.json").write_text(
                '{"modules": ['
                '{"name": "pe0", "total_power_w": 2.5},'
                '{"name": "sram0", "total_power_w": 0.0},'
                '{"name": "pe1", "total_power_w": 1.25}'
                '], "die_width_mm": 10.0}',
                encoding="utf-8",
            )
            (case / "power_grid.json").write_text(
                '{"grid_size": 32, "tiers": []}', encoding="utf-8"
            )
            (case / "hotspot_manifest.json").write_text(
                '{"input_granularity": "module"}', encoding="utf-8"
            )
            (case / "power.ptrace").write_text(
                "pe0 pe1\n2.5 1.25\n", encoding="utf-8"
            )
            names, power, mappings = _module_source_map(case)
            self.assertEqual(names, ["pe0", "pe1"])
            np.testing.assert_allclose(power, [2.5, 1.25])
            self.assertEqual(mappings[0]["source_cells"],
                             [{"index": 0, "weight": 1.0}])
            self.assertEqual(mappings[1]["source_cells"],
                             [{"index": 1, "weight": 1.0}])

    def test_module_input_source_map_rejects_power_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "layout.json").write_text(
                '{"modules": [{"name": "pe0", "total_power_w": 2.5}]}',
                encoding="utf-8",
            )
            (case / "power_grid.json").write_text(
                '{"grid_size": 32, "tiers": []}', encoding="utf-8"
            )
            (case / "hotspot_manifest.json").write_text(
                '{"input_granularity": "module"}', encoding="utf-8"
            )
            (case / "power.ptrace").write_text(
                "pe0\n2.4\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "power mismatch"):
                _module_source_map(case)

    def test_candidate_trace_preserves_grid_mapping_and_total_delta(self):
        response = LinearThermalResponse(
            ("m0", "m1"), ("t0",), np.array([2.0, 1.0]),
            np.array([30.0]), np.array([[1.0, 2.0]]), np.array([26.0]),
            {"source_mappings": [
                {"module": "m0", "source_cells": [
                    {"index": 0, "weight": 0.25},
                    {"index": 1, "weight": 0.75},
                ]},
                {"module": "m1", "source_cells": [
                    {"index": 2, "weight": 1.0},
                ]},
            ]},
        )
        actual = _candidate_trace(
            response, np.array([1.0, 2.0, 3.0]), np.array([3.0, 0.5])
        )
        np.testing.assert_allclose(actual, [1.25, 2.75, 2.5])

    def test_validation_selects_only_full_evaluation_candidates_from_search_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.json"
            path.write_text(
                '{"schema_version": 1, '
                '"selected_for_full_evaluation_ids": ["keep"], '
                '"candidates": ['
                '{"id": "pruned"}, {"id": "keep"}]}',
                encoding="utf-8",
            )
            self.assertEqual([item["id"] for item in _candidate_document(path)],
                             ["keep"])
            self.assertEqual(
                [item["id"] for item in _candidate_document(path, selected_only=False)],
                ["pruned", "keep"],
            )


class LinearSearchTests(unittest.TestCase):
    def test_candidate_estimate_and_pareto_pruning(self):
        matrix = np.array([[2.0, 0.1], [0.2, 1.5]])
        power = np.array([1.0, 1.0])
        temperature = np.array([50.0, 52.0])
        response = LinearThermalResponse(
            ("p0", "p1"), ("t0", "t1"),
            power, temperature, matrix, temperature - matrix @ power, {},
        )
        records = evaluate_candidates(response, [
            {"id": "fast", "power_w": [1.2, 1.1], "performance": 10,
             "energy": 4, "area": 5, "communication": 2},
            {"id": "slow", "power_w": [1.0, 1.0], "performance": 8,
             "energy": 3, "area": 4, "communication": 1},
        ], thermal_limit_c=100.0)
        self.assertEqual(len(records), 2)
        self.assertIn("predicted_delta_tsoft_c", records[0]["thermal"])
        front = pareto_front(records)
        self.assertEqual({record["id"] for record in front}, {"fast", "slow"})
        self.assertEqual(rank_for_pruning(records)[0]["id"], "slow")

    def test_overheated_candidate_is_not_preferred_when_feasible_exists(self):
        matrix = np.array([[10.0]])
        power = np.array([1.0])
        temperature = np.array([50.0])
        response = LinearThermalResponse(
            ("p0",), ("t0",), power, temperature,
            matrix, temperature - matrix @ power, {},
        )
        records = evaluate_candidates(response, [
            {"id": "hot", "power_w": [10.0], "performance": 100,
             "energy": 1, "area": 1, "communication": 1},
            {"id": "safe", "power_w": [1.0], "performance": 5,
             "energy": 5, "area": 5, "communication": 5},
        ], thermal_limit_c=100.0)
        front = pareto_front(records)
        self.assertEqual([record["id"] for record in front], ["safe"])


if __name__ == "__main__":
    unittest.main()
