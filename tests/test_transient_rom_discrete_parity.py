from pathlib import Path
import unittest

from workflow.common import read_json
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


if __name__ == "__main__":
    unittest.main()
