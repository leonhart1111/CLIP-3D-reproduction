from pathlib import Path
import unittest

from workflow.common import read_json
from workflow.floorplan.discrete_partition import search_discrete_partitions
from workflow.run_lifting_pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]
STEADY = (
    ROOT
    / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_"
    "lambda0020119_traffic_weighted_discrete_partition_exploratory.json"
)
ROM = (
    ROOT
    / "configs/experiments/clip3d_transient_rom_lambda0020119_"
    "traffic_weighted_discrete_partition_exploratory.json"
)
PROVENANCE = (
    ROOT / "manifests/parameter_provenance/lambda_wire_fft_rejected.json"
)


class TransientROMParityConfigTests(unittest.TestCase):
    """Catch untracked λ evidence or drift from the matched steady controls."""

    def test_tracked_lambda_evidence_preserves_rejection(self):
        report = read_json(PROVENANCE)

        self.assertEqual(report["lambda_wire"], 0.0020119160767721133)
        self.assertFalse(
            report["recommendation"]["accepted_for_this_workload"]
        )
        self.assertFalse(
            report["recommendation"]["cross_workload_transfer_validated"]
        )

    def test_rom_profile_matches_all_nonthermal_discrete_controls(self):
        steady = read_json(STEADY)
        rom = read_json(ROM)
        paths = (
            ("frequency", "f0_ghz"),
            ("frequency", "fmin_ghz"),
            ("frequency", "tsafe_c"),
            ("frequency", "ambient_c"),
            ("physical", "grid_size"),
            ("physical", "utilization"),
            ("physical", "r_convec_k_per_w"),
            ("layout_optimizer", "r_convec_k_per_w"),
            ("layout_optimizer", "alpha"),
            ("layout_optimizer", "beta"),
            ("layout_optimizer", "cross_tier_weight"),
            ("layout_optimizer", "lambda_wire"),
            ("layout_optimizer", "allowed_l2_tiers"),
            ("layout_optimizer", "wire_objective"),
            ("layout_optimizer", "partition_grid_steps"),
            ("layout_optimizer", "include_fixed_baseline"),
            ("delay", "wire_rounding"),
            ("delay", "wire_aggregation"),
        )

        for section, field in paths:
            with self.subTest(section=section, field=field):
                self.assertEqual(rom[section][field], steady[section][field])
        expected_source = str(PROVENANCE.relative_to(ROOT))
        self.assertEqual(
            steady["layout_optimizer"]["parameter_provenance"]["lambda_wire"][
                "source"
            ],
            expected_source,
        )
        self.assertEqual(
            rom["layout_optimizer"]["parameter_provenance"]["lambda_wire"][
                "source"
            ],
            expected_source,
        )
        self.assertTrue(rom["transient_rom"]["enabled"])
        self.assertFalse(rom["formal_validation"]["strict_p1"])
        self.assertFalse(rom["formal_validation"]["accepted"])
        validate_config(rom, "clip3d")


class DiscretePartitionEngineTests(unittest.TestCase):
    """Catch grid, geometry, fixed-baseline, partition, or tie-break drift."""

    @staticmethod
    def base_layout():
        return {
            "die_width_mm": 2.0,
            "die_height_mm": 2.0,
            "modules": [
                {
                    "name": "HOT",
                    "kind": "core",
                    "tier": 1,
                    "x_mm": 0.0,
                    "y_mm": 0.0,
                    "width_mm": 1.0,
                    "height_mm": 1.0,
                },
                {
                    "name": "L2",
                    "kind": "l2",
                    "tier": 1,
                    "x_mm": 1.0,
                    "y_mm": 1.0,
                    "width_mm": 1.0,
                    "height_mm": 1.0,
                },
            ],
        }

    @staticmethod
    def evaluator(layout, origin, start):
        l2 = next(module for module in layout["modules"] if module["name"] == "L2")
        x_mm = float(l2["x_mm"])
        y_mm = float(l2["y_mm"])
        tier = int(l2["tier"])
        cycle = int(x_mm + y_mm + 0.5)
        return {
            "objective_loss": abs(x_mm - 1.0) + cycle,
            "continuous_selected_wire_cycles": float(cycle) + x_mm / 100.0,
            "r2_wire_cycles": cycle,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "tier": tier,
            "origin": origin,
            "start": start,
        }

    def test_search_includes_fixed_and_one_deterministic_best_per_cycle(self):
        first = search_discrete_partitions(
            self.base_layout(), "L2", [1], 3, True, self.evaluator
        )
        second = search_discrete_partitions(
            self.base_layout(), "L2", [1], 3, True, self.evaluator
        )

        self.assertEqual(first, second)
        self.assertEqual(first["total_grid_candidates"], 9)
        self.assertTrue(first["fixed_baseline_included"])
        self.assertEqual(first["fixed_baseline"]["origin"], "fixed-bin")
        self.assertEqual(len(first["geometry_rejections"]), 4)
        grid_cycles = {
            item["r2_wire_cycles"]
            for item in first["legal_grid_candidates_detail"]
        }
        self.assertEqual(
            [item["r2_wire_cycles"] for item in first["partitions"]],
            sorted(grid_cycles),
        )
        self.assertIn(first["selected"], first["candidates"])

    def test_search_rejects_invalid_partition_controls(self):
        for steps in (2, 4):
            with self.subTest(steps=steps), self.assertRaisesRegex(
                ValueError, "odd integer"
            ):
                search_discrete_partitions(
                    self.base_layout(), "L2", [1], steps, True,
                    self.evaluator,
                )
        with self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
            search_discrete_partitions(
                self.base_layout(), "L2", [1], 3, False, self.evaluator
            )


if __name__ == "__main__":
    unittest.main()
