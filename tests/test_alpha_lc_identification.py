#!/usr/bin/env python3

import math
import json
import tempfile
import unittest
from pathlib import Path

from workflow.floorplan.optimize_layout import spatial_coupling
from workflow.thermal.identify_alpha_lc import placement_design


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


if __name__ == "__main__":
    unittest.main()
