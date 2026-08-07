from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from workflow.common import write_json
from workflow.transient.rom.contracts import (
    parse_settings,
    require_accepted_package,
    rom_input_identity,
)
from workflow.transient.rom.calibration_design import (
    _find_rectangle_shrink,
    _rectangle_coordinates,
    _rectangle_is_legal,
    build_design,
    interpolation_domain,
    layout_for_point,
    make_prbs_input,
)
from workflow.transient.rom.materialize_calibration import (
    build_prbs_power_windows,
    execute_calibration_cases,
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

    def test_parse_settings_rejects_prbs_fraction_that_can_zero_power(self):
        with self.assertRaisesRegex(ValueError, "prbs_fraction must be less than 1"):
            parse_settings({"transient_rom": {"prbs_fraction": 1.0}})

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


class CalibrationDesignTests(unittest.TestCase):
    @staticmethod
    def model() -> dict:
        modules = []
        for core in range(4):
            modules.append({
                "name": f"core{core}_logic", "kind": "core_logic", "core": core,
                "area_mm2": 1.0, "dynamic_power_w": 0.8,
                "leakage_power_w": 0.2, "total_power_w": 1.0,
            })
        modules.extend((
            {"name": "shared_l2", "kind": "l2", "area_mm2": 0.0025,
             "dynamic_power_w": 0.4, "leakage_power_w": 0.1, "total_power_w": 0.5},
            {"name": "noc", "kind": "interconnect", "area_mm2": 0.1,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.01, "total_power_w": 0.11},
        ))
        return {"schema_version": 1, "modules": modules}

    @staticmethod
    def settings():
        return parse_settings({})

    def test_two_tier_design_has_four_legal_anchors_per_tier(self):
        design = build_design(self.model(), [0, 1], self.settings())

        self.assertEqual(len(design["training"]), 8)
        self.assertEqual(design["allowed_l2_tiers"], [0, 1])
        self.assertEqual(
            {point["tier"] for point in design["training"]}, {0, 1}
        )
        self.assertEqual({point["tier"] for point in design["holdout"]}, {0, 1})
        self.assertEqual(interpolation_domain(design, 0)["kind"], "bilinear")
        self.assertEqual(interpolation_domain(design, 1)["kind"], "bilinear")
        for point in design["training"] + design["holdout"]:
            layout_for_point(design["base_layout"], point)

    def test_single_tier_design_requires_scipy_delaunay(self):
        with patch.dict("sys.modules", {"scipy": None, "scipy.spatial": None}):
            with self.assertRaisesRegex(ValueError, "SciPy is required for single-tier Delaunay"):
                build_design(self.model(), [1], self.settings())

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("scipy") is not None,
        "SciPy is unavailable in this test environment",
    )
    def test_single_tier_design_has_eight_anchors_and_two_holdouts(self):
        design = build_design(self.model(), [1], self.settings())

        self.assertEqual(len(design["training"]), 8)
        self.assertEqual(len(design["holdout"]), 2)
        self.assertEqual(design["allowed_l2_tiers"], [1])
        self.assertEqual({point["tier"] for point in design["training"]}, {1})
        self.assertEqual(interpolation_domain(design, 1)["kind"], "delaunay")
        self.assertEqual(len(interpolation_domain(design, 1)["simplices"]), 6)
        for point in design["training"] + design["holdout"]:
            layout_for_point(design["base_layout"], point)

    def test_prbs_is_repeatable_positive_and_independent(self):
        a = make_prbs_input(["core0", "shared_l2"], "shared_l2", self.settings())
        b = make_prbs_input(["core0", "shared_l2"], "shared_l2", self.settings())

        self.assertEqual(a, b)
        self.assertNotEqual(a["multipliers"]["core0"], a["multipliers"]["shared_l2"])
        self.assertEqual(set(a["multipliers"]), {"core0", "shared_l2"})
        for values in a["multipliers"].values():
            self.assertEqual(len(values), self.settings().calibration_windows)
            self.assertTrue(all(value > 0.0 for value in values))
            self.assertAlmostEqual(sum(values) / len(values), 1.0, places=12)

    def test_prbs_rejects_settings_that_can_generate_zero_multiplier(self):
        unsafe_settings = replace(self.settings(), prbs_fraction=1.0)

        with self.assertRaisesRegex(ValueError, "positive PRBS multipliers"):
            make_prbs_input(["core0", "shared_l2"], "shared_l2", unsafe_settings)

    def test_two_tier_design_repairs_obstructed_corners_as_one_rectangle(self):
        model = self.model()
        noc = next(module for module in model["modules"] if module["name"] == "noc")
        noc["preferred_width_mm"] = 2.25

        design = build_design(model, [0, 1], self.settings())
        anchors = [point for point in design["training"] if point["tier"] == 1]
        x_values = {point["x_mm"] for point in anchors}
        y_values = {point["y_mm"] for point in anchors}

        self.assertEqual(len(x_values), 2)
        self.assertEqual(len(y_values), 2)
        self.assertEqual(
            {(point["x_mm"], point["y_mm"]) for point in anchors},
            {(x_mm, y_mm) for x_mm in x_values for y_mm in y_values},
        )
        for point in anchors:
            layout_for_point(design["base_layout"], point)

    def test_rectangle_search_reaches_a_legal_scale_above_one_half(self):
        base_layout = {
            "die_width_mm": 10.0,
            "modules": [
                {"name": "shared_l2", "kind": "l2", "tier": 1,
                 "x_mm": 9.0, "y_mm": 9.0, "width_mm": 1.0, "height_mm": 1.0},
                {"name": "blocker", "kind": "interconnect", "tier": 1,
                 "x_mm": 0.0, "y_mm": 0.0, "width_mm": 3.0, "height_mm": 3.0},
            ],
        }

        scale = _find_rectangle_shrink(base_layout, 1, 9.0, 9.0, [])

        self.assertGreater(scale, 0.5)
        self.assertLess(scale, 0.7)
        self.assertTrue(_rectangle_is_legal(
            base_layout, 1, _rectangle_coordinates(9.0, 9.0, scale), []
        ))


class ROMCalibrationCaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.modules_path = self.root / "modules.json"
        self.power_path = self.root / "raw_power_windows.json"
        self.config_path = self.root / "config.json"
        self.hotspot = self.root / "hotspot"
        self.hotspot.write_text("mock executable", encoding="utf-8")
        write_json(self.modules_path, CalibrationDesignTests.model())
        write_json(self.power_path, self.raw_windows())
        write_json(self.config_path, {
            "physical": {
                "grid_size": 1,
                "utilization": 0.70,
                "r_convec_k_per_w": 0.1,
            },
            "frequency": {
                "ambient_c": 25.0,
                "f0_ghz": 2.0,
                "fmin_ghz": 1.0,
            },
        })
        self.settings = parse_settings({})
        self.design = build_design(
            CalibrationDesignTests.model(), [0, 1], self.settings
        )

    @staticmethod
    def raw_windows() -> dict:
        modules = CalibrationDesignTests.model()["modules"]
        records = []
        for index in range(2):
            samples = [dict(module) for module in modules]
            records.append({
                "schema_version": 1,
                "index": index,
                "source_stats_sha256": f"sha256:stats-{index}",
                "start_tick": 2 * index,
                "end_tick": 2 * (index + 1),
                "duration_ticks": 2,
                "duration_s": 0.002,
                "is_partial": False,
                "modules": samples,
                "totals": {
                    field: sum(module[field] for module in samples)
                    for field in ("dynamic_power_w", "leakage_power_w", "total_power_w")
                },
            })
        return {
            "schema_version": 1,
            "canonical_source_r1": "/source/r1",
            "transient_r1": "/source/transient-r1",
            "window_count": 2,
            "nominal_sample_interval_ms": 2.0,
            "nominal_sample_interval_ticks": 2,
            "measurement_start_tick": 0,
            "measurement_end_tick": 4,
            "module_names": [module["name"] for module in modules],
            "run_settings": {"mcpat_settings": {}},
            "power_provenance": {
                "dynamic": "McPAT Runtime Dynamic",
                "subthreshold_leakage": "McPAT Subthreshold Leakage",
                "gate_leakage": "McPAT Gate Leakage",
                "postprocessing": "none",
            },
            "windows": records,
        }

    def multipliers(self) -> dict[str, list[float]]:
        return self.design["prbs"]["multipliers"]

    @staticmethod
    def _mock_hotspot(case_dir: Path, hotspot: Path, initial_temperature: str) -> dict:
        manifest = __import__("json").loads(
            (case_dir / "transient_trace_manifest.json").read_text(encoding="utf-8")
        )
        names = (case_dir / "power_transient.ptrace").read_text(
            encoding="utf-8"
        ).splitlines()[0].split()
        rows = ["\t".join(names)]
        rows.extend("\t".join("300.0" for _ in names)
                    for _ in range(manifest["window_count"]))
        (case_dir / "transient.ttrace").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        return {
            "command": [str(hotspot), "-c", "hotspot.config"],
            "elapsed_seconds": 0.125,
            "initial_temperature": initial_temperature,
        }

    def test_calibration_emits_exactly_eight_training_and_two_holdout_cases(self):
        # Break caught: omitting an anchor/holdout or routing either through a
        # non-ambient HotSpot invocation changes the case-artifact contract.
        with patch(
            "workflow.transient.rom.materialize_calibration.run_hotspot_transient",
            side_effect=self._mock_hotspot,
        ) as hotspot_run:
            report = execute_calibration_cases(
                self.modules_path, self.power_path, self.config_path, self.design,
                self.root / "calibration", self.settings, hotspot=self.hotspot,
            )

        self.assertEqual(report["training_hotspot_calls"], 8)
        self.assertEqual(report["holdout_hotspot_calls"], 2)
        self.assertEqual(hotspot_run.call_count, 10)
        self.assertEqual(len(report["training_cases"]), 8)
        self.assertEqual(len(report["holdout_cases"]), 2)
        self.assertTrue(all(
            case["initial_temperature"] == "ambient"
            for case in report["training_cases"] + report["holdout_cases"]
        ))
        self.assertTrue(all(
            case["artifacts"]["sha256"]["power_windows"].startswith("sha256:")
            and case["artifacts"]["sha256"]["power_trace"].startswith("sha256:")
            and case["artifacts"]["sha256"]["temperature_trace"].startswith("sha256:")
            for case in report["training_cases"] + report["holdout_cases"]
        ))
        self.assertEqual(
            [case["frequency_ghz"] for case in report["holdout_cases"]],
            [2.0, 1.6],
        )
        self.assertTrue(all(
            case["periodic_steady_state"]["full_grid_converged"]
            for case in report["holdout_cases"]
        ))

    def test_prbs_windows_preserve_power_triplets(self):
        # Break caught: scaling only one power component or retaining a stale
        # total makes a materialized PRBS power record physically inconsistent.
        result = build_prbs_power_windows(
            self.raw_windows(), self.multipliers(), 64
        )

        self.assertEqual(result["window_count"], 64)
        self.assertEqual(result["power_provenance"], self.raw_windows()["power_provenance"])
        self.assertTrue(all(
            module["total_power_w"]
            == module["dynamic_power_w"] + module["leakage_power_w"]
            for window in result["windows"] for module in window["modules"]
        ))

    def test_calibration_rejects_a_nonempty_output_directory(self):
        # Break caught: publishing cases into an existing directory can mix
        # provenance from different calibration inputs.
        output = self.root / "occupied"
        output.mkdir()
        (output / "prior.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "not empty"):
            execute_calibration_cases(
                self.modules_path, self.power_path, self.config_path, self.design,
                output, self.settings, hotspot=self.hotspot,
            )


if __name__ == "__main__":
    unittest.main()
