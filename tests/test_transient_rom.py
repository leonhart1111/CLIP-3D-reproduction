from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflow.common import write_json
from workflow.transient.rom.contracts import (
    parse_settings,
    require_accepted_package,
    rom_input_identity,
)


class ROMContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = Path(self.temporary.name) / "accepted-rom"
        self.package.mkdir()

    @staticmethod
    def identity() -> dict:
        return rom_input_identity(
            canonical_r1_metadata_hash="sha256:r1",
            power_trace="sha256:power",
            modules_geometry_hash="sha256:modules",
            layout_geometry_hash="sha256:layout",
            configuration_hash="sha256:config",
            hotspot_hash="sha256:hotspot",
            grid={"rows": 64, "columns": 64},
            stack={"layers": ["silicon", "tim"]},
            cooling={"ambient_c": 25.0},
            allowed_l2_tiers=[1],
        )

    def test_parse_settings_uses_strict_rom_defaults(self):
        settings = parse_settings({})

        self.assertEqual(settings.sample_interval_ms, 2.0)
        self.assertEqual(settings.calibration_windows, 64)
        self.assertEqual(settings.prbs_seed, 20260807)
        self.assertEqual(settings.prbs_fraction, 0.20)
        self.assertEqual(settings.pod_energy_threshold, 0.999)
        self.assertEqual(settings.max_pod_rank, 16)
        self.assertEqual(settings.ridge, 1e-8)
        self.assertEqual(settings.max_condition_number, 1e10)
        self.assertEqual(settings.pss_period_repeats, 20)
        self.assertEqual(settings.pss_tolerance_c, 0.01)
        self.assertEqual(settings.frequency_tolerance_ghz, 0.01)
        self.assertEqual(settings.max_holdout_peak_error_c, 1.0)
        self.assertEqual(settings.max_holdout_grid_rmse_c, 0.75)
        self.assertEqual(settings.search_grid_points_per_axis, 25)
        self.assertEqual(settings.refinement_starts, 5)

    def test_parse_settings_requires_exact_8_plus_2_and_positive_thresholds(self):
        with self.assertRaisesRegex(ValueError, "calibration_runs must equal 8"):
            parse_settings({"transient_rom": {"calibration_runs": 7}})
        with self.assertRaisesRegex(ValueError, "validation_runs must equal 2"):
            parse_settings({"transient_rom": {"validation_runs": 3}})
        with self.assertRaisesRegex(ValueError, "calibration_runs must equal 8"):
            parse_settings({"transient_rom": {"calibration_runs": 8.0}})
        with self.assertRaisesRegex(ValueError, "pod_energy_threshold"):
            parse_settings({"transient_rom": {"pod_energy_threshold": 1.0}})
        with self.assertRaisesRegex(ValueError, "sample_interval_ms"):
            parse_settings({"transient_rom": {"sample_interval_ms": float("inf")}})

    def test_rom_input_identity_records_all_scientific_provenance(self):
        identity = self.identity()

        self.assertEqual(identity["canonical_r1_metadata_hash"], "sha256:r1")
        self.assertEqual(identity["power_trace"], "sha256:power")
        self.assertEqual(identity["modules_geometry_hash"], "sha256:modules")
        self.assertEqual(identity["layout_geometry_hash"], "sha256:layout")
        self.assertEqual(identity["configuration_hash"], "sha256:config")
        self.assertEqual(identity["hotspot_hash"], "sha256:hotspot")
        self.assertEqual(identity["grid"], {"rows": 64, "columns": 64})
        self.assertEqual(identity["stack"], {"layers": ["silicon", "tim"]})
        self.assertEqual(identity["cooling"], {"ambient_c": 25.0})
        self.assertEqual(identity["allowed_l2_tiers"], [1])

    def test_rejects_accepted_package_with_changed_power_identity(self):
        write_json(self.package / "rom_acceptance.json", {
            "accepted": True,
            "identity": self.identity(),
        })

        with self.assertRaisesRegex(ValueError, "power trace identity"):
            require_accepted_package(
                self.package, {**self.identity(), "power_trace": "changed"}
            )

    def test_require_accepted_package_rejects_partial_matching_identity(self):
        write_json(self.package / "rom_acceptance.json", {
            "accepted": True,
            "identity": self.identity(),
        })

        with self.assertRaisesRegex(ValueError, "canonical R1 metadata hash identity"):
            require_accepted_package(self.package, {"power_trace": "sha256:power"})

    def test_require_accepted_package_rejects_unaccepted_or_incomplete_identity(self):
        write_json(self.package / "rom_acceptance.json", {"accepted": False})
        with self.assertRaisesRegex(ValueError, "ROM package is not accepted"):
            require_accepted_package(self.package, self.identity())

        write_json(self.package / "rom_acceptance.json", {
            "accepted": True,
            "identity": {"power_trace": "sha256:power"},
        })
        with self.assertRaisesRegex(ValueError, "canonical R1 metadata hash identity"):
            require_accepted_package(self.package, self.identity())


if __name__ == "__main__":
    unittest.main()
