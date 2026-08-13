#!/usr/bin/env python3

import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow.floorplan.optimize_layout import spatial_coupling
from workflow.thermal.identify_alpha_lc import (
    case_identity,
    evaluate_grid_convergence,
    estimate_cross_tier_weight,
    placement_design,
    run_hotspot_case,
    unit_power_model,
)


class AlphaLcKernelTests(unittest.TestCase):
    @staticmethod
    def modules(distance_mm: float = 4.0) -> list[dict]:
        return [
            {
                "name": "receiver", "tier": 0,
                "x_mm": 0.0, "y_mm": 0.0,
                "width_mm": 1.0, "height_mm": 1.0,
                "total_power_w": 0.0,
            },
            {
                "name": "source", "tier": 0,
                "x_mm": distance_mm, "y_mm": 0.0,
                "width_mm": 1.0, "height_mm": 1.0,
                "total_power_w": 1.0,
            },
        ]

    def test_larger_characteristic_length_increases_nonlocal_coupling(self):
        modules = self.modules()
        short = spatial_coupling(
            modules, 8.0, 0.7, spatial_model="area-quadrature",
            quadrature_order=2, lc_mm=0.5,
        )
        long = spatial_coupling(
            modules, 8.0, 0.7, spatial_model="area-quadrature",
            quadrature_order=2, lc_mm=8.0,
        )
        self.assertGreater(long, short)

    def test_self_coupling_is_independent_of_length_for_point_samples(self):
        module = [{
            "name": "source", "tier": 0,
            "x_mm": 1.0, "y_mm": 1.0,
            "width_mm": 1.0, "height_mm": 1.0,
            "total_power_w": 2.0,
        }]
        short = spatial_coupling(
            module, 8.0, 0.7, spatial_model="center", lc_mm=0.1,
        )
        long = spatial_coupling(
            module, 8.0, 0.7, spatial_model="center", lc_mm=10.0,
        )
        self.assertAlmostEqual(short, 2.0)
        self.assertAlmostEqual(long, 2.0)

    def test_area_quadrature_differs_from_center_collapse(self):
        modules = self.modules(distance_mm=2.0)
        center = spatial_coupling(
            modules, 8.0, 0.7, spatial_model="center", lc_mm=1.0,
        )
        quadrature = spatial_coupling(
            modules, 8.0, 0.7, spatial_model="area-quadrature",
            quadrature_order=2, lc_mm=1.0,
        )
        self.assertNotAlmostEqual(center, quadrature, places=8)

    def test_rejects_invalid_characteristic_length(self):
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spatial_coupling(self.modules(), 8.0, 0.7, lc_mm=value)


class PlacementDesignTests(unittest.TestCase):
    def write_model(self, root: Path, l2_area: float = 1.0,
                    preferred_width_mm: float | None = None) -> Path:
        modules = []
        for core in range(4):
            modules.append({
                "name": f"core{core}", "kind": "core", "core": core,
                "area_mm2": 4.0, "dynamic_power_w": 2.0,
                "leakage_power_w": 0.2, "total_power_w": 2.2,
            })
        modules.append({
            "name": "l2", "kind": "l2", "area_mm2": l2_area,
            "preferred_width_mm": (
                math.sqrt(l2_area) if preferred_width_mm is None
                else preferred_width_mm
            ),
            "dynamic_power_w": 0.5, "leakage_power_w": 0.05,
            "total_power_w": 0.55,
        })
        path = root / "modules.json"
        path.write_text(json.dumps({"schema_version": 1, "modules": modules}))
        return path

    def test_returns_thirteen_distinct_named_legal_placements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_model(Path(directory))
            first = placement_design(path, 0.70)
            second = placement_design(path, 0.70)
        self.assertEqual([p["label"] for p in first], [p["label"] for p in second])
        self.assertEqual(len(first), 13)
        self.assertEqual(len({(p["x_mm"], p["y_mm"]) for p in first}), 13)
        expected = {
            "center", "corner_ll", "corner_lr", "corner_ul", "corner_ur",
            "edge_bottom", "edge_top", "edge_left", "edge_right",
            "near_core0", "near_core1", "near_core2", "near_core3",
        }
        self.assertEqual({p["label"] for p in first}, expected)
        for placement in first:
            l2 = next(m for m in placement["layout"]["modules"] if m["kind"] == "l2")
            self.assertAlmostEqual(placement["x_mm"], l2["x_mm"])
            self.assertAlmostEqual(placement["y_mm"], l2["y_mm"])
            self.assertGreaterEqual(placement["fx"], 0.0)
            self.assertLessEqual(placement["fx"], 1.0)

    def test_rejects_geometry_without_eleven_distinct_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_model(
                Path(directory), preferred_width_mm=math.sqrt(16.0 / 0.70),
            )
            with self.assertRaisesRegex(RuntimeError, "at least 11 distinct"):
                placement_design(path, 0.70)


class GridConvergenceTests(unittest.TestCase):
    def records(self, grid64_offset: float = 0.04,
                invert64: bool = False) -> list[dict]:
        records = []
        truth = {"a": 80.0, "b": 81.0, "c": 82.0}
        for label, value in truth.items():
            records.append({"label": label, "grid_size": 128, "tmax_c": value})
            adjusted = value + grid64_offset
            if invert64 and label == "a":
                adjusted = 83.0
            records.append({"label": label, "grid_size": 64, "tmax_c": adjusted})
            records.append({"label": label, "grid_size": 32, "tmax_c": value + 0.4})
        return records

    def test_selects_smallest_grid_with_temperature_and_rank_convergence(self):
        report = evaluate_grid_convergence(self.records())
        self.assertTrue(report["accepted"])
        self.assertEqual(report["selected_grid_size"], 64)
        self.assertFalse(report["grids"]["32"]["accepted"])
        self.assertTrue(report["grids"]["64"]["rank_order_matches_reference"])

    def test_falls_back_to_reference_grid_when_coarser_grid_fails(self):
        report = evaluate_grid_convergence(self.records(grid64_offset=0.2))
        self.assertTrue(report["accepted"])
        self.assertEqual(report["selected_grid_size"], 128)

    def test_rank_inversion_rejects_coarser_grid(self):
        report = evaluate_grid_convergence(self.records(invert64=True))
        self.assertEqual(report["selected_grid_size"], 128)
        self.assertFalse(report["grids"]["64"]["rank_order_matches_reference"])

    def test_missing_reference_layout_is_rejected(self):
        records = [r for r in self.records() if not (
            r["label"] == "c" and r["grid_size"] == 128
        )]
        with self.assertRaisesRegex(ValueError, "same layout labels"):
            evaluate_grid_convergence(records)

    def test_case_identity_changes_with_model_stack_grid_or_stimulus(self):
        base = case_identity(
            {"modules": [{"name": "l2", "total_power_w": 1.0}]},
            {"modules": [{"name": "l2", "x_mm": 0.0}]},
            {"grid_size": 64, "thermal_stack": {"local_resistance_scale": 1.0}},
            {"kind": "real-power"},
        )
        variants = [
            ({"modules": [{"name": "l2", "total_power_w": 2.0}]},
             {"modules": [{"name": "l2", "x_mm": 0.0}]},
             {"grid_size": 64, "thermal_stack": {"local_resistance_scale": 1.0}},
             {"kind": "real-power"}),
            ({"modules": [{"name": "l2", "total_power_w": 1.0}]},
             {"modules": [{"name": "l2", "x_mm": 1.0}]},
             {"grid_size": 64, "thermal_stack": {"local_resistance_scale": 1.0}},
             {"kind": "real-power"}),
            ({"modules": [{"name": "l2", "total_power_w": 1.0}]},
             {"modules": [{"name": "l2", "x_mm": 0.0}]},
             {"grid_size": 128, "thermal_stack": {"local_resistance_scale": 1.0}},
             {"kind": "real-power"}),
            ({"modules": [{"name": "l2", "total_power_w": 1.0}]},
             {"modules": [{"name": "l2", "x_mm": 0.0}]},
             {"grid_size": 64, "thermal_stack": {"local_resistance_scale": 1.0}},
             {"kind": "unit-power"}),
        ]
        self.assertTrue(all(case_identity(*variant) != base for variant in variants))

    def test_hotspot_case_reuses_only_matching_identity(self):
        model = {"modules": [{"name": "l2", "total_power_w": 1.0}]}
        layout = {"modules": [{"name": "l2", "x_mm": 0.0}]}
        physical = {
            "grid_size": 64, "utilization": 0.7, "ambient_c": 25.0,
            "r_convec_k_per_w": 1.042,
            "thermal_stack": {"local_resistance_scale": 1.0},
        }
        stimulus = {"kind": "real-power"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "modules.json"
            model_path.write_text(json.dumps(model))

            def fake_materialize(_model, case_dir, *_args):
                (case_dir / "hotspot_manifest.json").write_text(
                    json.dumps({"ambient_c": 25.0, "r_convec_k_per_w": 1.042})
                )

            def fake_hotspot(case_dir, _hotspot):
                result = {"tmax_c": 80.125, "peak_unit": "cell0"}
                (case_dir / "thermal_result.json").write_text(json.dumps(result))
                return result

            with patch(
                "workflow.thermal.identify_alpha_lc.materialize",
                side_effect=fake_materialize,
            ) as materialize_mock, patch(
                "workflow.thermal.identify_alpha_lc.run_hotspot",
                side_effect=fake_hotspot,
            ) as hotspot_mock:
                first = run_hotspot_case(
                    model_path, layout, physical, stimulus, root / "cases",
                    hotspot=Path("/fake/hotspot"),
                )
                second = run_hotspot_case(
                    model_path, layout, physical, stimulus, root / "cases",
                    hotspot=Path("/fake/hotspot"),
                )
                changed = run_hotspot_case(
                    model_path, layout, dict(physical, grid_size=128), stimulus,
                    root / "cases", hotspot=Path("/fake/hotspot"),
                )
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertNotEqual(first["case_dir"], changed["case_dir"])
            self.assertEqual(materialize_mock.call_count, 2)
            self.assertEqual(hotspot_mock.call_count, 2)


class UnitResponseTests(unittest.TestCase):
    def test_unit_power_model_changes_only_copied_power_fields(self):
        model = {
            "schema_version": 3,
            "modules": [
                {"name": "core0", "dynamic_power_w": 2.0,
                 "leakage_power_w": 0.2, "total_power_w": 2.2,
                 "area_mm2": 4.0},
                {"name": "l2", "dynamic_power_w": 0.5,
                 "leakage_power_w": 0.1, "total_power_w": 0.6,
                 "area_mm2": 1.0},
            ],
        }
        original = json.loads(json.dumps(model))
        stimulated = unit_power_model(model, "l2")
        self.assertEqual(model, original)
        core, l2 = stimulated["modules"]
        self.assertEqual(
            (core["dynamic_power_w"], core["leakage_power_w"], core["total_power_w"]),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            (l2["dynamic_power_w"], l2["leakage_power_w"], l2["total_power_w"]),
            (1.0, 0.0, 1.0),
        )
        self.assertEqual(stimulated["unit_power_stimulus"]["source_name"], "l2")

    def test_unit_power_model_rejects_missing_or_duplicate_source(self):
        duplicate = {"modules": [
            {"name": "x", "dynamic_power_w": 1, "leakage_power_w": 0,
             "total_power_w": 1},
            {"name": "x", "dynamic_power_w": 1, "leakage_power_w": 0,
             "total_power_w": 1},
        ]}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            unit_power_model(duplicate, "x")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            unit_power_model(duplicate, "missing")

    def test_cross_tier_estimator_uses_rise_ratio_and_is_reproducible(self):
        cases = [
            {"label": f"case{i}", "ambient_c": 25.0,
             "same_tier_tmax_c": 25.0 + same,
             "cross_tier_tmax_c": 25.0 + cross}
            for i, (same, cross) in enumerate(
                ((10.0, 7.0), (20.0, 14.0), (5.0, 3.5), (8.0, 5.6))
            )
        ]
        first = estimate_cross_tier_weight(cases, bootstrap_samples=200, seed=7)
        second = estimate_cross_tier_weight(cases, bootstrap_samples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.7)
        self.assertTrue(first["accepted"])
        self.assertEqual(len(first["per_case_ratios"]), 4)

    def test_cross_tier_estimator_rejects_invalid_denominator_and_instability(self):
        with self.assertRaisesRegex(ValueError, "same-tier temperature rise"):
            estimate_cross_tier_weight([{
                "label": "bad", "ambient_c": 25.0,
                "same_tier_tmax_c": 25.0, "cross_tier_tmax_c": 26.0,
            }], bootstrap_samples=20)
        unstable = [
            {"label": "a", "ambient_c": 25.0,
             "same_tier_tmax_c": 35.0, "cross_tier_tmax_c": 26.0},
            {"label": "b", "ambient_c": 25.0,
             "same_tier_tmax_c": 35.0, "cross_tier_tmax_c": 34.0},
            {"label": "c", "ambient_c": 25.0,
             "same_tier_tmax_c": 35.0, "cross_tier_tmax_c": 30.0},
        ]
        report = estimate_cross_tier_weight(
            unstable, bootstrap_samples=500, seed=11,
        )
        self.assertFalse(report["accepted"])
        self.assertIn("relative interval width", report["rejection_reasons"][0])


if __name__ == "__main__":
    unittest.main()
