from pathlib import Path
import tempfile
import unittest

from workflow.common import read_json
from workflow.floorplan.discrete_partition import search_discrete_partitions
from workflow.run_lifting_pipeline import validate_config
from workflow.transient.rom.paired_validation import (
    branch_metrics,
    publish_paired_comparison,
    require_selected_cycle_identity,
)


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


class PairedValidationUnitTests(unittest.TestCase):
    """Catch cycle drift and publication of predicted or incomplete BIPS."""

    @staticmethod
    def vector(cycle=7, aggregation="traffic-weighted"):
        return {
            "components_cycles": {"layout_wire": cycle},
            "wire_cycle_aggregation_for_r2": aggregation,
            "layout_delays": {"traffic_weighted_wire_cycles": cycle},
        }

    @staticmethod
    def branch(name, ipc2, frequency):
        metrics = branch_metrics(ipc2, frequency)
        return {
            "branch": name,
            **metrics,
            "controls": {
                "lambda_wire": 0.0020119160767721133,
                "wire_aggregation": "traffic-weighted",
                "wire_rounding": "nearest",
            },
            "artifacts": {
                "hotspot": f"/{name}/hotspot/result.json",
                "r2_result": f"/{name}/gem5_r2/r2_result.json",
            },
        }

    def test_selected_cycle_must_equal_every_r2_representation(self):
        self.assertEqual(
            require_selected_cycle_identity(
                {"r2_wire_cycles": 7}, self.vector(), "traffic-weighted"
            ),
            7,
        )
        with self.assertRaisesRegex(ValueError, "integer wire cycle"):
            require_selected_cycle_identity(
                {"r2_wire_cycles": 7}, self.vector(cycle=6),
                "traffic-weighted",
            )
        with self.assertRaisesRegex(ValueError, "aggregation"):
            require_selected_cycle_identity(
                {"r2_wire_cycles": 7}, self.vector(aggregation="mean"),
                "traffic-weighted",
            )

    def test_branch_metrics_require_both_real_measurements(self):
        self.assertEqual(branch_metrics(1.5, 1.8), {
            "validated_f_sus_trans_hotspot_ghz": 1.8,
            "measured_ipc2": 1.5,
            "measured_bips2_trans": 2.7,
        })
        self.assertIsNone(branch_metrics(None, 1.8)["measured_bips2_trans"])
        self.assertIsNone(branch_metrics(1.5, None)["measured_bips2_trans"])

    def test_paired_report_is_published_only_for_two_measured_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixed = self.branch("fixed-bin", 1.4, 1.8)
            clip = self.branch("clip3d", None, 1.9)

            self.assertIsNone(
                publish_paired_comparison(
                    fixed, clip, output, r2_requested=False
                )
            )
            self.assertFalse((output / "paired_comparison.json").exists())
            self.assertFalse((output / "paired_comparison.csv").exists())
            self.assertIsNone(
                publish_paired_comparison(fixed, clip, output, r2_requested=True)
            )
            self.assertFalse((output / "paired_comparison.json").exists())
            self.assertFalse((output / "paired_comparison.csv").exists())

            clip = self.branch("clip3d", 1.5, 1.9)
            report = publish_paired_comparison(
                fixed, clip, output, r2_requested=True
            )
            fixed_bips = 1.4 * 1.8
            clip_bips = 1.5 * 1.9
            expected = (clip_bips - fixed_bips) / fixed_bips * 100.0
            self.assertEqual(
                report["bips2_trans_improvement_percent"], expected
            )
            self.assertTrue((output / "paired_comparison.json").is_file())
            self.assertTrue((output / "paired_comparison.csv").is_file())
            self.assertTrue(report["non_formal"])
            self.assertFalse(report["paper_equivalent"])


if __name__ == "__main__":
    unittest.main()
