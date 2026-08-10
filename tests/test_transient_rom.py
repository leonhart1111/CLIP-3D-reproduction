from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy

from workflow.common import read_json, write_json
from workflow.transient.generate_hotspot_trace import materialize_trace, trace_input_identity
from workflow.transient.rom.contracts import (
    parse_settings,
    require_accepted_package,
    require_package_calibration_evidence,
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
from workflow.transient.rom.evidence import ROM_CLASSIFICATION
from workflow.transient.rom.calibrate_rom import validate_calibration_holdouts
from workflow.transient.rom.materialize_calibration import (
    build_prbs_power_windows,
    execute_calibration_cases,
)
from workflow.transient.rom.layout_rom import (
    _average_power_steady_state,
    evaluate_layout_rom,
    find_rom_sustainable_frequency,
    interpolate_l2_input,
    validate_holdouts,
)
from workflow.transient.rom.optimize_layout import (
    canonical_frequency_grid,
    optimize_transient_layout,
)
from workflow.transient.rom.pod_state_space import (
    StateSpaceModel,
    discretize,
    fit_state_space,
    load_model,
    save_model,
)
from workflow.transient.run_hotspot_transient import (
    parse_ttrace_grid,
    summarize_period_end_convergence,
)
from workflow.transient.validation import power_trace_identity, validate_power_windows
from workflow.transient.verify_sustainable_frequency import (
    find_sustainable_frequency,
    last_period_peak,
)


_USE_DERIVED_FINAL_VALIDATION = object()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_test_artifact_manifest(package: Path) -> None:
    classification = {
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
    }
    artifacts = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "rom_artifact_manifest.json":
            continue
        artifacts.append({
            "path": path.relative_to(package).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "classification": classification,
        })
    write_json(package / "rom_artifact_manifest.json", {
        "schema_version": 1,
        **classification,
        "scope": str(package.resolve()),
        "artifacts": artifacts,
    })


def write_synthetic_accepted_package(
    package: Path, design: dict, identity: dict, settings: object,
    model: StateSpaceModel, modules_source: Path, config_source: Path,
    source_power_windows: dict,
) -> None:
    """Write one internally consistent, fully audited synthetic ROM package."""
    classification = {
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
    }
    package.mkdir(parents=True, exist_ok=True)
    package_modules = package / "modules.json"
    package_modules.write_bytes(Path(modules_source).read_bytes())
    modules_sha256 = "sha256:" + hashlib.sha256(
        package_modules.read_bytes()
    ).hexdigest()
    source_power_path = package / "source_power_windows.json"
    write_json(source_power_path, source_power_windows)
    source_power_sha256 = "sha256:" + hashlib.sha256(
        source_power_path.read_bytes()
    ).hexdigest()
    prbs_power_windows = build_prbs_power_windows(
        source_power_windows, design["prbs"]["multipliers"],
        settings.calibration_windows,
    )
    package_config = package / "config.json"
    package_config.write_bytes(Path(config_source).read_bytes())
    config = read_json(package_config)
    f0_ghz = float(config["frequency"]["f0_ghz"])
    fmin_ghz = float(config["frequency"]["fmin_ghz"])
    holdout_frequencies = (f0_ghz, 0.6 * f0_ghz + 0.4 * fmin_ghz)
    write_json(package / "anchors.json", design)
    write_json(package / "calibration_manifest.json", {
        "schema_version": 1,
        **classification,
        "calibration_runs": 8,
        "validation_runs": 2,
        "initialization_runs": 2,
        "calibration_hotspot_calls": 12,
        "calibration_design_hash": canonical_json_sha256(design),
        "identity": identity,
        "settings": {
            **vars(settings),
            "calibration_runs": 8,
            "validation_runs": 2,
            "initialization_runs": 2,
            "calibration_hotspot_calls": 12,
        },
        "sources": {
            "modules": "modules.json",
            "modules_sha256": modules_sha256,
            "power_trace_identity": identity["power_trace"],
            "power_windows": "source_power_windows.json",
            "power_windows_sha256": source_power_sha256,
            "config": "config.json",
            "config_sha256": identity["configuration_hash"],
            "hotspot_sha256": identity["hotspot_hash"],
        },
        "training_ids": [point["id"] for point in design["training"]],
        "holdout_ids": [point["id"] for point in design["holdout"]],
    })

    def evidence_case(point: dict, kind: str, index: int) -> dict:
        case_dir = package / f"{kind}_{point['id']}"
        case_dir.mkdir(exist_ok=True)
        artifact_paths = {
            "modules": package_modules,
            "layout": case_dir / "layout.json",
            "power_windows": case_dir / "power_windows.json",
            "power_trace": case_dir / "power_transient.ptrace",
            "temperature_trace": case_dir / "transient.ttrace",
        }
        if kind == "holdout":
            artifact_paths.update({
                "steady_initialization": case_dir / "steady_initialization.json",
                "initialization_steady": case_dir / "initialization.steady.txt",
                "initialization_grid_steady": (
                    case_dir / "initialization.grid.steady.txt"
                ),
            })
        layout = layout_for_point(design["base_layout"], point)
        case_power_windows = (
            prbs_power_windows if kind == "training" else source_power_windows
        )
        period_rows = (
            settings.calibration_windows if kind == "training"
            else len(source_power_windows["windows"]) * settings.pss_period_repeats
        )
        write_json(artifact_paths["layout"], layout)
        write_json(artifact_paths["power_windows"], case_power_windows)
        frequency_ghz = (
            f0_ghz if kind == "training" else holdout_frequencies[index]
        )
        materialize_trace(package_modules, artifact_paths["layout"],
                          artifact_paths["power_windows"], case_dir, config,
                          frequency_scale=frequency_ghz / f0_ghz,
                          period_repeats=period_rows // len(case_power_windows["windows"]))
        if kind == "holdout":
            artifact_paths["initialization_steady"].write_text(
                "shared_l2 300.0\n", encoding="utf-8"
            )
            artifact_paths["initialization_grid_steady"].write_text(
                "cell0 300.0\n", encoding="utf-8"
            )
            write_json(artifact_paths["steady_initialization"], {
                "command": ["hotspot", "-steady_file", "initialization.steady.txt"],
                "return_code": 0,
                "elapsed_seconds": 0.1,
                "input_sha256": {
                    "hotspot.config": "sha256:" + hashlib.sha256(
                        (case_dir / "hotspot.config").read_bytes()
                    ).hexdigest(),
                    "power_transient.ptrace": "sha256:" + hashlib.sha256(
                        artifact_paths["power_trace"].read_bytes()
                    ).hexdigest(),
                    "stack.lcf": "sha256:" + hashlib.sha256(
                        (case_dir / "stack.lcf").read_bytes()
                    ).hexdigest(),
                    "materials.txt": "sha256:" + hashlib.sha256(
                        (case_dir / "materials.txt").read_bytes()
                    ).hexdigest(),
                    "hotspot_binary": identity["hotspot_hash"],
                },
                "output_sha256": {
                    "initialization.steady.txt": "sha256:" + hashlib.sha256(
                        artifact_paths["initialization_steady"].read_bytes()
                    ).hexdigest(),
                    "initialization.grid.steady.txt": "sha256:" + hashlib.sha256(
                        artifact_paths["initialization_grid_steady"].read_bytes()
                    ).hexdigest(),
                },
            })
            rom = evaluate_layout_rom(
                model, design, case_power_windows, layout, frequency_ghz,
                settings, config,
            )
            period_grid_c = rom["final_period_grid_c"]
            temperature_rows_k = period_grid_c * settings.pss_period_repeats
            artifact_paths["temperature_trace"].write_text(
                "\t".join(model.grid_unit_names) + "\n"
                + "\n".join(
                    "\t".join(f"{value + 273.15:.17g}" for value in row)
                    for row in temperature_rows_k
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            artifact_paths["temperature_trace"].write_text(
                "cell0\n" + "300.0\n" * period_rows, encoding="utf-8"
            )
        trace_manifest = read_json(case_dir / "transient_trace_manifest.json")
        trace_manifest.update({
            "trace_input_identity": trace_input_identity(
                package_modules, artifact_paths["layout"], artifact_paths["power_windows"],
                package_config, frequency_ghz / f0_ghz,
            ),
            "hotspot_sha256": identity["hotspot_hash"],
            "window_count": period_rows,
            "temperature_grid_unit_names": ["cell0"],
        })
        write_json(case_dir / "transient_trace_manifest.json", trace_manifest)
        case = {
            "id": point["id"],
            "kind": kind,
            "point": point,
            "initial_temperature": (
                "ambient" if kind == "training" else "average_power_steady"
            ),
            "frequency_ghz": frequency_ghz,
            "frequency_scale": frequency_ghz / f0_ghz,
            "window_count": period_rows,
            "hotspot": {
                "command": (
                    ["hotspot"] if kind == "training"
                    else ["hotspot", "-init_file", "initial.steady.txt"]
                ),
                "elapsed_seconds": 0.1,
            },
            "artifacts": {
                **{
                    name: path.relative_to(package).as_posix()
                    for name, path in artifact_paths.items()
                },
                "sha256": {
                    name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in artifact_paths.items()
                },
            },
        }
        if kind == "holdout":
            case["steady_initialization"] = read_json(
                artifact_paths["steady_initialization"]
            )
            names, rows = parse_ttrace_grid(artifact_paths["temperature_trace"])
            windows_per_period = len(source_power_windows["windows"])
            convergence = summarize_period_end_convergence(
                rows, windows_per_period
            )
            case["periodic_steady_state"] = {
                "period_repeats": settings.pss_period_repeats,
                "pss_tolerance_c": settings.pss_tolerance_c,
                "full_grid_converged": True,
                "grid_unit_names": names,
                "evidence": convergence,
            }
        return case

    training = [
        evidence_case(point, "training", index)
        for index, point in enumerate(design["training"])
    ]
    holdouts = [
        evidence_case(point, "holdout", index)
        for index, point in enumerate(design["holdout"])
    ]
    write_json(package / "calibration_cases.json", {
        "schema_version": 1,
        **classification,
        "training_hotspot_calls": 8,
        "holdout_initialization_hotspot_calls": 2,
        "holdout_transient_hotspot_calls": 2,
        "calibration_hotspot_calls": 12,
        "training_cases": training,
        "holdout_cases": holdouts,
    })
    ordered_training = sorted(training, key=lambda case: case["id"])
    fit_report = {
        "schema_version": 1,
        **classification,
        "case_order": [case["id"] for case in ordered_training],
        "calibration_design_hash": canonical_json_sha256(design),
        "temperature_trace_sha256": {
            case["id"]: case["artifacts"]["sha256"]["temperature_trace"]
            for case in ordered_training
        },
        "power_windows_sha256": {
            case["id"]: case["artifacts"]["sha256"]["power_windows"]
            for case in ordered_training
        },
        "pod": {
            "rank": 1,
            "energy_threshold": settings.pod_energy_threshold,
        },
        "identification": {
            "ridge": settings.ridge,
            "gram_condition_number": 1.0,
            "max_condition_number": settings.max_condition_number,
        },
        "continuous_conversion": {
            "logm_input_condition_number": 1.0,
            "max_logm_condition_number": settings.max_logm_condition_number,
            "logm_error_estimate": 0.0,
            "max_logm_error_estimate": settings.max_logm_error_estimate,
            "exp_log_reconstruction_error": 0.0,
            "max_exp_log_reconstruction_error": (
                settings.max_exp_log_reconstruction_error
            ),
        },
    }
    write_json(package / "fit_report.json", fit_report)
    save_model(package / "pod_model.npz", model, fit_report)
    validation = validate_calibration_holdouts(
        model, design, holdouts, settings, config,
        output_dir=package, identity=None, publish=False,
    )
    if validation.get("accepted") is not True:
        raise AssertionError("synthetic accepted package failed canonical holdout gates")
    write_json(package / "validation_report.json", validation)
    write_json(package / "rom_acceptance.json", {
        "schema_version": 1,
        **classification,
        "accepted": True,
        "identity": identity,
        "validation_report": "validation_report.json",
    })
    write_test_artifact_manifest(package)


class ROMContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "accepted-rom"
        self.modules = self.root / "modules.json"
        self.power_windows = self.root / "power_windows.json"
        self.config_path = self.root / "config.json"
        self.hotspot = self.root / "hotspot"

        modules = CalibrationDesignTests.model()
        modules["ipc1"] = 2.0
        write_json(self.modules, modules)
        source_power = ROMCalibrationCaseTests.raw_windows()
        write_json(self.power_windows, source_power)
        self.hotspot.write_text("mock executable", encoding="utf-8")
        write_json(self.config_path, self.config())
        settings = self.settings()
        design = build_design(modules, [1], settings)
        fixed_names = tuple(
            module["name"] for module in modules["modules"]
            if module["kind"] != "l2"
        )
        model = StateSpaceModel(
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.zeros((1, len(fixed_names))),
            b_l2_anchors={
                point["id"]: numpy.ones((1, 1))
                for point in design["training"]
            },
            module_names=fixed_names,
            grid_unit_names=("cell0",),
        )
        physical = self.config()["physical"]
        self._identity = rom_input_identity(
            canonical_r1_metadata_hash="sha256:r1",
            power_trace=power_trace_identity(source_power),
            modules_geometry_hash="sha256:" + hashlib.sha256(
                self.modules.read_bytes()
            ).hexdigest(),
            layout_geometry_hash=canonical_json_sha256(design["base_layout"]),
            configuration_hash="sha256:" + hashlib.sha256(
                self.config_path.read_bytes()
            ).hexdigest(),
            hotspot_hash="sha256:" + hashlib.sha256(
                self.hotspot.read_bytes()
            ).hexdigest(),
            grid={"rows": physical["grid_size"], "columns": physical["grid_size"]},
            stack=physical["thermal_stack"],
            cooling={
                "ambient_c": self.config()["frequency"]["ambient_c"],
                "r_convec_k_per_w": physical["r_convec_k_per_w"],
            },
            allowed_l2_tiers=[1],
            calibration_design_hash=canonical_json_sha256(design),
        )
        write_synthetic_accepted_package(
            self.package, design, self._identity, settings, model,
            self.modules, self.config_path, source_power,
        )

    @staticmethod
    def config() -> dict:
        return {
            "frequency": {
                "ambient_c": 25.0,
                "f0_ghz": 2.0,
                "fmin_ghz": 1.0,
                "tsafe_c": 50.0,
            },
            "physical": {
                "utilization": 0.70,
                "grid_size": 1,
                "r_convec_k_per_w": 0.1,
                "thermal_stack": {"layers": ["silicon", "tim"]},
            },
            "layout_optimizer": {
                "lambda_wire": 0.25,
                "wire_objective": "continuous",
                "allowed_l2_tiers": [1],
            },
            "delay": {"wire_aggregation": "mean", "wire_rounding": "nearest"},
        }

    @staticmethod
    def settings():
        return parse_settings({})

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
            calibration_design_hash="sha256:design",
        )

    def reuse_identity(self) -> dict:
        return self._identity

    def rewrite_and_rehash(self, path: Path, **changes: object) -> None:
        value = read_json(path)
        value.update(changes)
        write_json(path, value)
        write_test_artifact_manifest(self.package)

    def replace_case_directory_with_symlink(self, case_name: str) -> None:
        case_dir = self.package / case_name
        outside = self.root / "outside-case"
        case_dir.rename(outside)
        case_dir.symlink_to(outside, target_is_directory=True)

    def remove_packaged_source(self, name: str) -> None:
        (self.package / name).unlink()

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
        self.assertEqual(settings.max_logm_condition_number, 1e12)
        self.assertEqual(settings.max_logm_error_estimate, 1e-8)
        self.assertEqual(settings.max_exp_log_reconstruction_error, 1e-8)
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
        with self.assertRaisesRegex(ValueError, "max_logm_condition_number"):
            parse_settings({"transient_rom": {"max_logm_condition_number": 0.5}})
        with self.assertRaisesRegex(ValueError, "max_exp_log_reconstruction_error"):
            parse_settings({
                "transient_rom": {"max_exp_log_reconstruction_error": 0.0}
            })

    def test_parse_settings_rejects_prbs_fraction_that_can_zero_power(self):
        with self.assertRaisesRegex(ValueError, "prbs_fraction must be less than 1"):
            parse_settings({"transient_rom": {"prbs_fraction": 1.0}})

    def test_rom_input_identity_records_all_scientific_provenance(self):
        identity = self.reuse_identity()

        self.assertEqual(identity["canonical_r1_metadata_hash"], "sha256:r1")
        self.assertTrue(identity["power_trace"].startswith("sha256:"))
        self.assertTrue(identity["modules_geometry_hash"].startswith("sha256:"))
        self.assertTrue(identity["layout_geometry_hash"].startswith("sha256:"))
        self.assertTrue(identity["configuration_hash"].startswith("sha256:"))
        self.assertTrue(identity["hotspot_hash"].startswith("sha256:"))
        self.assertEqual(identity["grid"], {"rows": 1, "columns": 1})
        self.assertEqual(identity["stack"], {"layers": ["silicon", "tim"]})
        self.assertEqual(
            identity["cooling"], {"ambient_c": 25.0, "r_convec_k_per_w": 0.1}
        )
        self.assertEqual(identity["allowed_l2_tiers"], [1])
        self.assertTrue(identity["calibration_design_hash"].startswith("sha256:"))

    def test_reuse_rejects_contradictory_classification_and_symlink_tree(self):
        original = read_json(self.package / "rom_acceptance.json")
        self.rewrite_and_rehash(
            self.package / "rom_acceptance.json",
            thermal_mode="steady", non_formal=False,
        )
        with self.assertRaisesRegex(ValueError, "classification"):
            require_accepted_package(self.package, self.reuse_identity())
        write_json(self.package / "rom_acceptance.json", original)
        write_test_artifact_manifest(self.package)

        case_id = read_json(self.package / "anchors.json")["training"][0]["id"]
        self.replace_case_directory_with_symlink(f"training_{case_id}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_rejects_missing_packaged_source_power(self):
        self.remove_packaged_source("source_power_windows.json")
        with self.assertRaisesRegex(ValueError, "source power"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_accepts_exact_eight_plus_two_plus_two_call_evidence(self):
        # Break caught: continuing to require the legacy ten-call ambient
        # package rejects scientifically matched holdout preconditioning.
        evidence = require_package_calibration_evidence(
            self.package, self.reuse_identity(), self.settings(),
        )

        self.assertEqual(evidence["cases"]["training_hotspot_calls"], 8)
        self.assertEqual(
            evidence["cases"]["holdout_initialization_hotspot_calls"], 2
        )
        self.assertEqual(evidence["cases"]["holdout_transient_hotspot_calls"], 2)
        self.assertEqual(evidence["cases"]["calibration_hotspot_calls"], 12)

    def test_reuse_rejects_legacy_ten_call_package(self):
        # Break caught: silently accepting old ambient-start holdouts would
        # preserve the exact PSS failure this change is intended to remove.
        cases_path = self.package / "calibration_cases.json"
        cases = read_json(cases_path)
        cases.pop("holdout_initialization_hotspot_calls")
        cases.pop("holdout_transient_hotspot_calls")
        cases.pop("calibration_hotspot_calls")
        cases["holdout_hotspot_calls"] = 2
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.package)

        with self.assertRaisesRegex(ValueError, r"exact 8\+2\+2"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_rejects_changed_holdout_initialization_hash(self):
        # Break caught: a rehashed package inventory must not hide a steady
        # state that differs from the one bound by its holdout case evidence.
        cases = read_json(self.package / "calibration_cases.json")
        holdout = cases["holdout_cases"][0]
        path = self.package / holdout["artifacts"]["initialization_steady"]
        path.write_text("shared_l2 450.0\n", encoding="utf-8")
        write_test_artifact_manifest(self.package)

        with self.assertRaisesRegex(ValueError, "initialization_steady hash differs"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_rejects_rehashed_malformed_training_trace(self):
        """Catch a rehashed trace whose rows cannot represent the PRBS training input."""
        cases = read_json(self.package / "calibration_cases.json")
        case = cases["training_cases"][0]
        trace = self.package / case["artifacts"]["power_trace"]
        trace.write_text("cell0\n1.0\n", encoding="utf-8")
        digest = "sha256:" + hashlib.sha256(trace.read_bytes()).hexdigest()
        case["artifacts"]["sha256"]["power_trace"] = digest
        write_json(self.package / "calibration_cases.json", cases)
        write_test_artifact_manifest(self.package)
        with self.assertRaisesRegex(ValueError, "training.*trace"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_rejects_rehashed_same_shaped_power_payload(self):
        """Catch a finite payload edit that preserves all trace dimensions."""
        cases = read_json(self.package / "calibration_cases.json")
        case = cases["training_cases"][0]
        trace = self.package / case["artifacts"]["power_trace"]
        lines = trace.read_text(encoding="utf-8").splitlines()
        cells = lines[1].split()
        cells[0] = str(float(cells[0]) + 0.125)
        lines[1] = "\t".join(cells)
        trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
        case["artifacts"]["sha256"]["power_trace"] = (
            "sha256:" + hashlib.sha256(trace.read_bytes()).hexdigest()
        )
        write_json(self.package / "calibration_cases.json", cases)
        write_test_artifact_manifest(self.package)
        with self.assertRaisesRegex(ValueError, "power trace payload"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_reuse_rejects_contradictory_fit_report_classification(self):
        # Break caught: a rehashed fit report must not reinterpret a transient
        # ROM model as formal steady-state evidence during package reuse.
        fit_path = self.package / "fit_report.json"
        fit_report = read_json(fit_path)
        fit_report.update({"thermal_mode": "steady", "non_formal": False})
        write_json(fit_path, fit_report)
        model, _ = load_model(self.package / "pod_model.npz")
        save_model(self.package / "pod_model.npz", model, fit_report)
        write_test_artifact_manifest(self.package)

        with self.assertRaisesRegex(ValueError, "classification"):
            require_package_calibration_evidence(
                self.package, self.reuse_identity(), self.settings(),
            )

    def test_rejects_accepted_package_with_changed_calibration_design(self):
        # Break caught: matching physics with changed anchors, holdouts, or
        # simplices must not reinterpret stored B_L2 columns as a new design.
        write_json(self.package / "rom_acceptance.json", {
            **ROM_CLASSIFICATION,
            "accepted": True,
            "identity": self.reuse_identity(),
        })

        with self.assertRaisesRegex(ValueError, "calibration design hash identity"):
            require_accepted_package(
                self.package,
                {**self.reuse_identity(), "calibration_design_hash": "sha256:changed"},
            )

    def test_rejects_accepted_package_with_changed_power_identity(self):
        write_json(self.package / "rom_acceptance.json", {
            **ROM_CLASSIFICATION,
            "accepted": True,
            "identity": self.reuse_identity(),
        })

        with self.assertRaisesRegex(ValueError, "power trace identity"):
            require_accepted_package(
                self.package, {**self.reuse_identity(), "power_trace": "changed"}
            )

    def test_require_accepted_package_rejects_partial_matching_identity(self):
        write_json(self.package / "rom_acceptance.json", {
            **ROM_CLASSIFICATION,
            "accepted": True,
            "identity": self.reuse_identity(),
        })

        with self.assertRaisesRegex(ValueError, "canonical R1 metadata hash identity"):
            require_accepted_package(self.package, {"power_trace": "sha256:power"})

    def test_require_accepted_package_rejects_unaccepted_or_incomplete_identity(self):
        write_json(self.package / "rom_acceptance.json", {
            **ROM_CLASSIFICATION, "accepted": False,
        })
        with self.assertRaisesRegex(ValueError, "ROM package is not accepted"):
            require_accepted_package(self.package, self.reuse_identity())

        write_json(self.package / "rom_acceptance.json", {
            **ROM_CLASSIFICATION,
            "accepted": True,
            "identity": {"power_trace": "sha256:power"},
        })
        with self.assertRaisesRegex(ValueError, "canonical R1 metadata hash identity"):
            require_accepted_package(self.package, self.reuse_identity())


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
    def _mock_steady(case_dir: Path, hotspot: Path) -> dict:
        steady = case_dir / "initialization.steady.txt"
        grid = case_dir / "initialization.grid.steady.txt"
        steady.write_text("shared_l2 390.000000\n", encoding="utf-8")
        grid.write_text("t0_r00_c00 390.000000\n", encoding="utf-8")
        record = {
            "command": [str(hotspot), "-steady_file", steady.name],
            "return_code": 0,
            "elapsed_seconds": 0.0625,
            "input_sha256": {"power_transient.ptrace": "sha256:power"},
            "output_sha256": {
                steady.name: "sha256:" + hashlib.sha256(
                    steady.read_bytes()
                ).hexdigest(),
                grid.name: "sha256:" + hashlib.sha256(grid.read_bytes()).hexdigest(),
            },
        }
        write_json(case_dir / "steady_initialization.json", record)
        return record

    @staticmethod
    def _mock_hotspot(case_dir: Path, hotspot: Path, initial_temperature: str,
                      steady_source: Path | None = None) -> dict:
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
        command = [str(hotspot), "-c", "hotspot.config"]
        if initial_temperature == "steady":
            if steady_source != case_dir / "initialization.steady.txt":
                raise AssertionError("holdout did not use its matched steady state")
            command.extend(["-init_file", "initial.steady.txt"])
        return {
            "command": command,
            "elapsed_seconds": 0.125,
            "initial_temperature": initial_temperature,
        }

    def test_calibration_emits_exactly_eight_training_and_two_holdout_cases(self):
        # Break caught: omitting an anchor/holdout or routing either through a
        # non-ambient HotSpot invocation changes the case-artifact contract.
        with patch(
            "workflow.transient.rom.materialize_calibration.run_hotspot_steady",
            side_effect=self._mock_steady,
        ) as steady_run, patch(
            "workflow.transient.rom.materialize_calibration.run_hotspot_transient",
            side_effect=self._mock_hotspot,
        ) as hotspot_run:
            report = execute_calibration_cases(
                self.modules_path, self.power_path, self.config_path, self.design,
                self.root / "calibration", self.settings, hotspot=self.hotspot,
                identity={
                    **ROMContractTests.identity(),
                    "allowed_l2_tiers": [0, 1],
                    "calibration_design_hash": canonical_json_sha256(self.design),
                },
            )

        self.assertEqual(report["training_hotspot_calls"], 8)
        self.assertEqual(report["holdout_initialization_hotspot_calls"], 2)
        self.assertEqual(report["holdout_transient_hotspot_calls"], 2)
        self.assertEqual(report["calibration_hotspot_calls"], 12)
        self.assertEqual(steady_run.call_count, 2)
        self.assertEqual(hotspot_run.call_count, 10)
        self.assertEqual(len(report["training_cases"]), 8)
        self.assertEqual(len(report["holdout_cases"]), 2)
        self.assertTrue(all(
            case["initial_temperature"] == "ambient"
            for case in report["training_cases"]
        ))
        self.assertTrue(all(
            case["initial_temperature"] == "average_power_steady"
            for case in report["holdout_cases"]
        ))
        self.assertTrue(all(
            case["artifacts"]["sha256"]["modules"].startswith("sha256:")
            and case["artifacts"]["sha256"]["layout"].startswith("sha256:")
            and case["artifacts"]["sha256"]["power_windows"].startswith("sha256:")
            and case["artifacts"]["sha256"]["power_trace"].startswith("sha256:")
            and case["artifacts"]["sha256"]["temperature_trace"].startswith("sha256:")
            for case in report["training_cases"] + report["holdout_cases"]
        ))
        self.assertTrue(all(
            case["hotspot"] == {
                "command": [str(self.hotspot.resolve()), "-c", "hotspot.config"],
                "elapsed_seconds": 0.125,
            }
            for case in report["training_cases"]
        ))
        self.assertTrue(all(
            case["hotspot"]["command"][-2:] == [
                "-init_file", "initial.steady.txt"
            ]
            and case["steady_initialization"]["return_code"] == 0
            for case in report["holdout_cases"]
        ))
        self.assertEqual(
            [case["frequency_ghz"] for case in report["holdout_cases"]],
            [2.0, 1.6],
        )
        for case in report["holdout_cases"]:
            for name in (
                "steady_initialization", "initialization_steady",
                "initialization_grid_steady",
            ):
                self.assertTrue(case["artifacts"]["sha256"][name].startswith(
                    "sha256:"
                ))
            pss = case["periodic_steady_state"]
            self.assertEqual(pss["period_repeats"], 20)
            self.assertEqual(pss["grid_unit_names"], ["t0_r00_c00", "t1_r00_c00"])
            self.assertTrue(pss["full_grid_converged"])
            self.assertEqual(pss["evidence"]["period_count"], 20)
            self.assertEqual(pss["evidence"]["grid_cell_count"], 2)
            self.assertEqual(len(pss["evidence"]["period_end_deltas"]), 19)
            self.assertEqual(pss["evidence"]["last_delta_max_c"], 0.0)
        package = self.root / "calibration"
        self.assertEqual(read_json(package / "anchors.json"), self.design)
        manifest = read_json(package / "calibration_manifest.json")
        self.assertEqual(
            manifest["calibration_design_hash"], canonical_json_sha256(self.design)
        )
        self.assertEqual(manifest["settings"]["calibration_windows"], 64)
        self.assertEqual(manifest["settings"]["calibration_runs"], 8)
        self.assertEqual(manifest["settings"]["validation_runs"], 2)
        self.assertEqual(manifest["settings"]["initialization_runs"], 2)
        self.assertEqual(manifest["settings"]["calibration_hotspot_calls"], 12)
        self.assertEqual(
            manifest["identity"]["calibration_design_hash"],
            canonical_json_sha256(self.design),
        )
        self.assertEqual(
            (package / "modules.json").read_bytes(), self.modules_path.read_bytes()
        )
        for case in report["training_cases"] + report["holdout_cases"]:
            for name in (
                "modules", "layout", "power_windows", "power_trace",
                "temperature_trace",
            ):
                relative = Path(case["artifacts"][name])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertTrue((package / relative).is_file())
        for case in report["holdout_cases"]:
            for name in (
                "steady_initialization", "initialization_steady",
                "initialization_grid_steady",
            ):
                self.assertTrue((package / case["artifacts"][name]).is_file())

    def test_prbs_windows_preserve_power_triplets(self):
        # Break caught: scaling only one power component or retaining a stale
        # total makes a materialized PRBS power record physically inconsistent.
        result = build_prbs_power_windows(
            self.raw_windows(), self.multipliers(), 64
        )

        self.assertEqual(result["window_count"], 64)
        self.assertEqual(result["power_provenance"], self.raw_windows()["power_provenance"])
        self.assertEqual(validate_power_windows(result)["window_count"], 64)
        self.assertEqual(
            [window["source_stats_sha256"] for window in result["windows"]],
            ["sha256:stats-0", "sha256:stats-1"] * 32,
        )
        self.assertEqual(
            result["prbs_excitation"]["source_power_trace_identity"],
            power_trace_identity(self.raw_windows()),
        )
        self.assertTrue(all(
            module["total_power_w"]
            == module["dynamic_power_w"] + module["leakage_power_w"]
            for window in result["windows"] for module in window["modules"]
        ))
        self.assertTrue(all(
            power >= 0.0
            for window in result["windows"]
            for module in [*window["modules"], window["totals"]]
            for power in (
                module["dynamic_power_w"],
                module["leakage_power_w"],
                module["total_power_w"],
            )
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
                identity={
                    **ROMContractTests.identity(),
                    "allowed_l2_tiers": [0, 1],
                    "calibration_design_hash": canonical_json_sha256(self.design),
                },
            )

    def test_standalone_calibration_cli_publishes_reusable_package_manifest(self):
        # Break caught: a CLI calibration that writes acceptance without the
        # package inventory reports success but is rejected by normal reuse.
        from workflow.transient.rom import calibrate_rom

        output = self.root / "standalone-package"
        design_path = self.root / "design.json"
        write_json(design_path, self.design)
        identity = {
            **ROMContractTests.identity(),
            "allowed_l2_tiers": [0, 1],
            "calibration_design_hash": canonical_json_sha256(self.design),
        }

        def materialize(*_args, **_kwargs):
            output.mkdir()
            write_json(output / "anchors.json", self.design)
            write_json(output / "calibration_cases.json", {"synthetic": True})
            return {
                "training_hotspot_calls": 8,
                "holdout_initialization_hotspot_calls": 2,
                "holdout_transient_hotspot_calls": 2,
                "calibration_hotspot_calls": 12,
                "training_cases": [],
                "holdout_cases": [],
            }

        def save(path, *_args, **_kwargs):
            Path(path).write_bytes(b"synthetic model")

        def accept(*_args, **_kwargs):
            report = {"accepted": True, "failure_reasons": []}
            write_json(output / "validation_report.json", report)
            write_json(output / "rom_acceptance.json", {
                "accepted": True,
                "identity": identity,
                "validation_report": "validation_report.json",
            })
            return report

        argv = [
            "calibrate_rom", "--modules", str(self.modules_path),
            "--power-windows", str(self.power_path),
            "--config", str(self.config_path), "--design", str(design_path),
            "--output-dir", str(output), "--hotspot", str(self.hotspot),
        ]
        with patch("sys.argv", argv), patch.object(
            calibrate_rom, "_package_identity", return_value=identity,
        ), patch.object(
            calibrate_rom, "execute_calibration_cases", side_effect=materialize,
        ), patch.object(
            calibrate_rom, "fit_state_space",
            return_value=(object(), {"pod": {"rank": 1}}),
        ), patch.object(
            calibrate_rom, "save_model", side_effect=save,
        ), patch.object(
            calibrate_rom, "validate_calibration_holdouts", side_effect=accept,
        ), patch("builtins.print"):
            calibrate_rom.main()

        manifest = read_json(output / "rom_artifact_manifest.json")
        self.assertTrue(manifest["non_formal"])
        self.assertFalse(manifest["paper_equivalent"])
        paths = {record["path"] for record in manifest["artifacts"]}
        self.assertIn("rom_acceptance.json", paths)
        self.assertIn("pod_model.npz", paths)


class StateSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def first_order_model(a: float, b: float) -> StateSpaceModel:
        return StateSpaceModel(
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[a]], dtype=float),
            b_fixed=numpy.empty((1, 0), dtype=float),
            b_l2_anchors={"a0": numpy.array([[b]], dtype=float)},
            module_names=(),
            grid_unit_names=("cell0",),
        )

    def settings(self):
        return replace(parse_settings({}), calibration_windows=2)

    def unstable_training_cases(self) -> list[dict]:
        cases = []
        for index in range(8):
            case_dir = self.root / f"a{index}"
            case_dir.mkdir()
            write_json(case_dir / "hotspot_manifest.json", {"ambient_c": 25.0})
            (case_dir / "transient.ttrace").write_text(
                "t0_r00_c00\n299.15\n300.15\n", encoding="utf-8"
            )
            write_json(case_dir / "power_windows.json", {
                "nominal_sample_interval_ms": 2.0,
                "windows": [
                    {
                        "index": window,
                        "duration_s": 0.002,
                        "modules": [{
                            "name": "shared_l2", "kind": "l2",
                            "total_power_w": 0.0,
                        }],
                    }
                    for window in range(2)
                ],
            })
            cases.append({
                "id": f"a{index}",
                "point": {"id": f"a{index}"},
                "initial_temperature": "ambient",
                "artifacts": {
                    "temperature_trace": str(case_dir / "transient.ttrace"),
                    "power_windows": str(case_dir / "power_windows.json"),
                    "sha256": {
                        "temperature_trace": "sha256:" + hashlib.sha256(
                            (case_dir / "transient.ttrace").read_bytes()
                        ).hexdigest(),
                        "power_windows": "sha256:" + hashlib.sha256(
                            (case_dir / "power_windows.json").read_bytes()
                        ).hexdigest(),
                    },
                },
            })
        return cases

    def test_augmented_discretization_matches_first_order_rc(self):
        # Break caught: integrating B through A via an inverse or Euler step
        # gives the wrong exact zero-order-hold response.
        model = self.first_order_model(a=-2.0, b=3.0)

        a_d, b_d = discretize(model, 0.5, model.b_l2_anchors["a0"])

        self.assertAlmostEqual(a_d[0, 0], math.exp(-1.0), places=12)
        self.assertAlmostEqual(
            b_d[0, 0], 1.5 * (1.0 - math.exp(-1.0)), places=12
        )

    def test_fit_rejects_unstable_continuous_pole(self):
        # Break caught: accepting a positive continuous pole permits divergent
        # temperatures in PSS evaluation.
        with self.assertRaisesRegex(
            ValueError, "unstable continuous-time state matrix"
        ):
            fit_state_space(self.unstable_training_cases(), self.settings())

    def test_fit_requires_recorded_ambient_initial_state(self):
        # Break caught: prepending a zero state to a steady-start trace biases
        # every identified state and input coefficient.
        cases = self.unstable_training_cases()
        cases[0]["initial_temperature"] = "steady"

        with self.assertRaisesRegex(ValueError, "ambient initial temperature"):
            fit_state_space(cases, self.settings())

    def test_fit_rejects_truncated_training_case(self):
        # Break caught: mutually truncated temperature and power artifacts can
        # otherwise masquerade as the configured fixed calibration period.
        with self.assertRaisesRegex(ValueError, "calibration_windows"):
            fit_state_space(self.unstable_training_cases(), parse_settings({}))

    def test_fit_rejects_unreliable_logm_input_conditioning(self):
        # Break caught: merely recording an ill-conditioned augmented discrete
        # model allows a numerically unreliable continuous conversion to ship.
        settings = replace(
            self.settings(), max_logm_condition_number=1.0,
        )

        with self.assertRaisesRegex(ValueError, "logm input condition number"):
            fit_state_space(self.unstable_training_cases(), settings)

    def test_fit_rejects_exp_log_reconstruction_above_configured_limit(self):
        # Break caught: accepting logm output without checking exp(log(A))
        # permits conversion error to be amplified by every ROM recurrence.
        settings = replace(
            self.settings(),
            max_logm_condition_number=1e300,
            max_logm_error_estimate=1e300,
            max_exp_log_reconstruction_error=1e-8,
        )
        from scipy.linalg import expm as scipy_expm

        def inaccurate_expm(matrix):
            reconstructed = scipy_expm(matrix)
            reconstructed[0, 0] += 1e-4
            return reconstructed

        with patch(
            "workflow.transient.rom.pod_state_space.expm",
            side_effect=inaccurate_expm,
        ), self.assertRaisesRegex(ValueError, "exp/log reconstruction error"):
            fit_state_space(self.unstable_training_cases(), settings)

    def test_fit_rejects_recorded_training_artifact_hash_mismatch(self):
        # Break caught: fitting bytes that differ from the case record breaks the
        # audit link between calibration execution and identified coefficients.
        cases = self.unstable_training_cases()
        cases[0]["artifacts"]["sha256"]["power_windows"] = "sha256:wrong"

        with self.assertRaisesRegex(ValueError, "recorded power_windows hash"):
            fit_state_space(cases, self.settings())

    def test_save_rejects_unstable_continuous_matrix(self):
        # Break caught: an unstable matrix persisted in a package bypasses the
        # stability gate enforced during fitting.
        model = self.first_order_model(a=0.25, b=1.0)

        with self.assertRaisesRegex(
            ValueError, "unstable continuous-time state matrix"
        ):
            save_model(self.root / "unstable.npz", model, {})

    def test_load_rejects_duplicate_l2_anchor_ids(self):
        # Break caught: dictionary construction otherwise overwrites one
        # anchor's input matrix silently.
        path = self.root / "duplicate.npz"
        numpy.savez_compressed(
            path,
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.empty((1, 0)),
            b_l2_anchor_ids=numpy.array(["a0", "a0"]),
            b_l2_anchor_values=numpy.array([[1.0], [2.0]]),
            module_names=numpy.array([], dtype=str),
            grid_unit_names=numpy.array(["cell0"], dtype=str),
            metadata_json=numpy.asarray("{}"),
        )

        with self.assertRaisesRegex(ValueError, "anchor ids.*unique"):
            load_model(path)

    def test_load_rejects_complex_model_matrix(self):
        # Break caught: dtype conversion must not silently discard imaginary
        # continuous dynamics from a malformed archive.
        path = self.root / "complex.npz"
        numpy.savez_compressed(
            path,
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0 + 0.25j]]),
            b_fixed=numpy.empty((1, 0)),
            b_l2_anchor_ids=numpy.array(["a0"]),
            b_l2_anchor_values=numpy.array([[1.0]]),
            module_names=numpy.array([], dtype=str),
            grid_unit_names=numpy.array(["cell0"], dtype=str),
            metadata_json=numpy.asarray("{}"),
        )

        with self.assertRaisesRegex(ValueError, "a_continuous must be real"):
            load_model(path)

    def test_model_persistence_uses_exact_path_and_round_trips(self):
        # Break caught: numpy's filename overload appends .npz and makes the
        # same advertised Path impossible to load.
        path = self.root / "model"
        model = self.first_order_model(a=-2.0, b=3.0)
        metadata = {"rank": 1, "residual": 0.125}

        save_model(path, model, metadata)
        loaded, loaded_metadata = load_model(path)

        self.assertTrue(path.is_file())
        self.assertTrue(numpy.array_equal(
            loaded.temperature_basis, model.temperature_basis
        ))
        self.assertTrue(numpy.array_equal(
            loaded.a_continuous, model.a_continuous
        ))
        self.assertTrue(numpy.array_equal(
            loaded.b_l2_anchors["a0"], model.b_l2_anchors["a0"]
        ))
        self.assertEqual(loaded.module_names, model.module_names)
        self.assertEqual(loaded_metadata, metadata)

    def test_loaded_model_persists_temperature_grid_order(self):
        # Break caught: without ordered grid identities in the loaded model,
        # holdout cells can be compared positionally against the wrong POD rows.
        path = self.root / "grid-names.npz"
        numpy.savez_compressed(
            path,
            temperature_basis=numpy.ones((2, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.empty((1, 0)),
            b_l2_anchor_ids=numpy.array(["a0"]),
            b_l2_anchor_values=numpy.array([[1.0]]),
            module_names=numpy.array([], dtype=str),
            grid_unit_names=numpy.array(["cell0", "cell1"], dtype=str),
            metadata_json=numpy.asarray('{"grid_unit_names":["cell0","cell1"]}'),
        )

        loaded, _ = load_model(path)

        self.assertEqual(
            getattr(loaded, "grid_unit_names", None), ("cell0", "cell1")
        )


class ROMEvaluationTests(unittest.TestCase):
    @staticmethod
    def model() -> StateSpaceModel:
        return StateSpaceModel(
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.array([[1.0]]),
            b_l2_anchors={
                "a0": numpy.array([[2.0]]),
                "a1": numpy.array([[4.0]]),
                "a2": numpy.array([[6.0]]),
                "a3": numpy.array([[8.0]]),
            },
            module_names=("core0",),
            grid_unit_names=("cell0",),
        )

    @staticmethod
    def design() -> dict:
        return {
            "l2_name": "shared_l2",
            "allowed_l2_tiers": [0],
            "domains": {
                "0": {
                    "kind": "bilinear",
                    "anchor_ids": ["a0", "a1", "a2", "a3"],
                    "corners": [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
                },
            },
        }

    @staticmethod
    def layout(x_mm: float = 0.0, y_mm: float = 0.0) -> dict:
        return {
            "modules": [
                {"name": "core0", "kind": "core_logic", "tier": 0,
                 "x_mm": 0.0, "y_mm": 0.0, "width_mm": 1.0, "height_mm": 1.0},
                {"name": "shared_l2", "kind": "l2", "tier": 0,
                 "x_mm": x_mm, "y_mm": y_mm,
                 "width_mm": 0.25, "height_mm": 0.25},
            ],
        }

    @staticmethod
    def power_windows() -> dict:
        return {
            "nominal_sample_interval_ms": 1000.0,
            "windows": [{
                "duration_s": 1.0,
                "modules": [
                    {"name": "shared_l2", "kind": "l2",
                     "dynamic_power_w": 1.0, "leakage_power_w": 0.5,
                     "total_power_w": 1.5},
                    {"name": "core0", "kind": "core_logic",
                     "dynamic_power_w": 2.0, "leakage_power_w": 1.0,
                     "total_power_w": 3.0},
                ],
            }],
        }

    @staticmethod
    def settings():
        return replace(
            parse_settings({}), pss_period_repeats=2, pss_tolerance_c=100.0
        )

    @staticmethod
    def non_normal_model() -> StateSpaceModel:
        # Its period-to-period response dips below 0.11 C on the first
        # comparison, then rises above it on the final comparison.
        return StateSpaceModel(
            temperature_basis=numpy.eye(2),
            a_continuous=numpy.array([
                [math.log(0.9), 100.0 / 9.0],
                [0.0, math.log(0.9)],
            ]),
            b_fixed=numpy.zeros((2, 1)),
            b_l2_anchors={
                identifier: numpy.array([
                    [-0.03833730356990477], [0.007024033786918412],
                ])
                for identifier in ("a0", "a1", "a2", "a3")
            },
            module_names=("core0",),
            grid_unit_names=("cell0", "cell1"),
        )

    @staticmethod
    def config() -> dict:
        return {
            "frequency": {
                "f0_ghz": 2.0, "fmin_ghz": 1.0,
                "ambient_c": 25.0, "tsafe_c": 100.0,
            },
        }

    def config_with_grid(self, values: list[float]) -> dict:
        return {
            **self.config(),
            "transient_rom": {"frequencies_ghz": values},
        }

    def test_average_power_equilibrium_matches_first_order_rc(self):
        # Break caught: keeping the ROM at ambient while real HotSpot starts
        # from average-power steady state compares different thermal problems.
        model = StateSpaceTests.first_order_model(a=-2.0, b=3.0)

        state, audit = _average_power_steady_state(
            model,
            model.b_l2_anchors["a0"],
            [(0.25, numpy.array([4.0])), (0.75, numpy.array([4.0]))],
            max_condition_number=1.0e10,
        )

        self.assertAlmostEqual(state[0], 6.0, places=12)
        self.assertEqual(audit["method"], "average_power_continuous_equilibrium")
        self.assertLessEqual(audit["normalized_residual"], 1.0e-10)

    def test_average_power_equilibrium_weights_unequal_durations(self):
        # Break caught: an unweighted row mean disagrees with the physical
        # energy average when trace windows have unequal durations.
        model = StateSpaceTests.first_order_model(a=-2.0, b=3.0)

        state, audit = _average_power_steady_state(
            model,
            model.b_l2_anchors["a0"],
            [(0.25, numpy.array([2.0])), (0.75, numpy.array([6.0]))],
            max_condition_number=1.0e10,
        )

        self.assertAlmostEqual(audit["mean_power_w"][0], 5.0, places=12)
        self.assertAlmostEqual(state[0], 7.5, places=12)

    def test_average_power_equilibrium_rejects_ill_conditioned_state_matrix(self):
        # Break caught: solving a nearly singular A can create an arbitrarily
        # large initialization that nevertheless contains finite numbers.
        model = StateSpaceModel(
            temperature_basis=numpy.eye(2),
            a_continuous=numpy.diag([-1.0, -1.0e-12]),
            b_fixed=numpy.empty((2, 0)),
            b_l2_anchors={"a0": numpy.ones((2, 1))},
            module_names=(),
            grid_unit_names=("cell0", "cell1"),
        )

        with self.assertRaisesRegex(ValueError, "condition number"):
            _average_power_steady_state(
                model, model.b_l2_anchors["a0"],
                [(1.0, numpy.array([1.0]))],
                max_condition_number=1.0e10,
            )

    def test_average_power_equilibrium_rejects_large_solve_residual(self):
        # Break caught: a successful solver return is not sufficient evidence
        # that the returned state satisfies the thermal equilibrium equation.
        model = StateSpaceTests.first_order_model(a=-2.0, b=3.0)

        with patch(
            "workflow.transient.rom.layout_rom.numpy.linalg.solve",
            return_value=numpy.array([0.0]),
        ), self.assertRaisesRegex(ValueError, "normalized residual"):
            _average_power_steady_state(
                model, model.b_l2_anchors["a0"],
                [(1.0, numpy.array([4.0]))],
                max_condition_number=1.0e10,
            )

    def test_anchor_interpolation_is_exact_and_extrapolation_fails(self):
        # Break caught: blending tiers/anchors or silently extrapolating can
        # manufacture an input matrix unsupported by calibration evidence.
        actual = interpolate_l2_input(self.model(), self.design(), 0, 0.0, 0.0)

        self.assertTrue(numpy.array_equal(actual, numpy.array([[2.0]])))
        with self.assertRaisesRegex(ValueError, "outside ROM interpolation domain"):
            interpolate_l2_input(self.model(), self.design(), 0, -0.01, 0.5)

    def test_single_tier_interpolation_uses_persisted_barycentric_simplex(self):
        # Break caught: nearest-anchor selection or inverse-distance weights do
        # not implement the persisted single-tier Delaunay domain.
        model = replace(self.model(), b_l2_anchors={
            "a0": numpy.array([[1.0]]),
            "a1": numpy.array([[3.0]]),
            "a2": numpy.array([[5.0]]),
        })
        design = {
            "domains": {"1": {
                "kind": "delaunay", "anchor_ids": ["a0", "a1", "a2"],
                "points": [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                "simplices": [[0, 1, 2]],
            }},
        }

        actual = interpolate_l2_input(model, design, 1, 0.5, 0.5)

        self.assertTrue(numpy.allclose(actual, numpy.array([[2.5]]), atol=1e-15))
        with self.assertRaisesRegex(ValueError, "outside ROM interpolation domain"):
            interpolate_l2_input(model, design, 1, 1.5, 1.5)

    def test_evaluation_scales_dynamic_power_and_window_time_separately(self):
        # Break caught: scaling total power or retaining nominal duration loses
        # the specified leakage+s*dynamic, duration/s frequency transform.
        result = evaluate_layout_rom(
            self.model(), self.design(), self.power_windows(), self.layout(),
            4.0, self.settings(),
            {"frequency": {"f0_ghz": 2.0, "ambient_c": 25.0,
                           "tsafe_c": 100.0}},
        )

        expected_rise = 10.0
        self.assertAlmostEqual(result["frequency_scale"], 2.0)
        self.assertEqual(result["module_order"], ["core0", "shared_l2"])
        self.assertAlmostEqual(result["final_period_grid_c"][-1][0],
                               25.0 + expected_rise, places=12)
        self.assertAlmostEqual(result["last_period_peak_c"],
                               25.0 + expected_rise, places=12)
        self.assertTrue(result["last_period_peak"]["includes_period_initial_state"])
        self.assertTrue(result["converged"])
        self.assertEqual(
            result["thermal_initialization"]["method"],
            "average_power_continuous_equilibrium",
        )

    def test_rom_evaluates_fixed_horizon_after_early_tolerance_hit(self):
        # Break caught: stopping when an intermediate endpoint delta first
        # enters tolerance would make ROM evidence use fewer configured periods.
        settings = replace(
            self.settings(), pss_period_repeats=3, pss_tolerance_c=0.11,
        )
        power_windows = self.power_windows()
        second = {
            **power_windows["windows"][0],
            "modules": [
                dict(module) for module in power_windows["windows"][0]["modules"]
            ],
        }
        second["modules"][0].update(
            dynamic_power_w=4.0, total_power_w=4.5,
        )
        power_windows["windows"].append(second)

        result = evaluate_layout_rom(
            self.non_normal_model(), self.design(), power_windows,
            self.layout(), 2.0, settings, self.config(),
        )

        self.assertEqual(result["periods_evaluated"], settings.pss_period_repeats)
        self.assertEqual(len(result["period_end_deltas_max_c"]), 2)
        self.assertTrue(result["converged"])

    def test_partial_final_window_uses_full_hotspot_sampling_interval(self):
        # Break caught: integrating the final partial ROI duration makes the
        # ROM use less thermal time than HotSpot, whose last row is held for a
        # complete sampling interval before frequency scaling.
        power_windows = self.power_windows()
        partial = {
            **power_windows["windows"][0],
            "duration_s": 0.25,
            "modules": [dict(module) for module in power_windows["windows"][0]["modules"]],
        }
        power_windows["windows"].append(partial)

        result = evaluate_layout_rom(
            self.model(), self.design(), power_windows, self.layout(),
            4.0, self.settings(),
            {"frequency": {"f0_ghz": 2.0, "ambient_c": 25.0,
                           "tsafe_c": 100.0}},
        )

        expected_rise = 10.0
        self.assertAlmostEqual(
            result["final_period_grid_c"][-1][0], 25.0 + expected_rise,
            places=12,
        )
        self.assertAlmostEqual(
            result["thermal_initialization"]["period_duration_s"], 1.0,
            places=12,
        )

    def test_evaluation_rejects_changed_or_malformed_module_inputs(self):
        # Break caught: accepting an incomplete window shifts B columns and can
        # apply one module's power to another module's spatial input.
        malformed = self.power_windows()
        malformed["windows"][0]["modules"].pop()

        with self.assertRaisesRegex(ValueError, "module set"):
            evaluate_layout_rom(
                self.model(), self.design(), malformed, self.layout(), 2.0,
                self.settings(),
                {"frequency": {"f0_ghz": 2.0, "ambient_c": 25.0,
                               "tsafe_c": 100.0}},
            )

    def test_frequency_search_preserves_task1_local_bracket_semantics(self):
        # Break caught: a global monotonic binary search would miss the second
        # observed safe/unsafe transition in this deliberately nonmonotone grid.
        def evaluation(_model, _design, _powers, _layout, frequency, _settings, _config):
            safe = frequency < 1.25 or 1.5 <= frequency < 1.75
            return {
                "frequency_ghz": frequency, "converged": True,
                "last_period_peak_c": 40.0 if safe else 60.0,
            }

        with patch(
            "workflow.transient.rom.layout_rom.evaluate_layout_rom",
            side_effect=evaluation,
        ):
            result = find_rom_sustainable_frequency(
                self.model(), self.design(), self.power_windows(), self.layout(),
                [1.0, 1.25, 1.5], self.settings(),
                {"frequency": {"f0_ghz": 2.0, "ambient_c": 25.0,
                               "tsafe_c": 50.0}},
            )

        self.assertEqual(len(result["safe_unsafe_brackets"]), 2)
        self.assertTrue(all(
            bracket["local_boundary_assumption"]
            for bracket in result["safe_unsafe_brackets"]
        ))

    @staticmethod
    def failed_holdouts() -> list[dict]:
        return [
            {
                "id": "h0", "geometry_match": True,
                "input_identity_match": True, "frequency_match": True,
                "hotspot_trace_identity_match": True,
                "temperature_grid_identity_match": True,
                "rom": {
                    "converged": True,
                    "final_period_grid_c": [[30.0, 31.0]],
                    "last_period_peak_c": 31.0, "safe": True,
                },
                "hotspot": {
                    "converged": True,
                    "final_period_grid_c": [[31.0, 33.0]],
                    "last_period_peak_c": 33.0, "safe": False,
                },
            },
            {
                "id": "h1", "geometry_match": True,
                "input_identity_match": True, "frequency_match": True,
                "hotspot_trace_identity_match": True,
                "temperature_grid_identity_match": True,
                "rom": {
                    "converged": False,
                    "final_period_grid_c": [[30.0, 30.0]],
                    "last_period_peak_c": 30.0, "safe": True,
                },
                "hotspot": {
                    "converged": True,
                    "final_period_grid_c": [[30.0, 30.0]],
                    "last_period_peak_c": 30.0, "safe": True,
                },
            },
        ]

    def test_holdout_gate_rejects_pss_peak_grid_or_safety_mismatch(self):
        # Break caught: accepting on average error alone can publish a ROM that
        # misses a peak, flips safety, or never reaches periodic steady state.
        result = validate_holdouts(self.failed_holdouts(), self.settings())

        self.assertFalse(result["accepted"])
        self.assertIn("periodic_steady_state", result["failure_reasons"])
        self.assertIn("grid_rmse", result["failure_reasons"])
        self.assertIn("peak_temperature_error", result["failure_reasons"])
        self.assertIn("safety_classification", result["failure_reasons"])

    def test_acceptance_artifact_is_published_only_after_every_gate_passes(self):
        # Break caught: leaving an acceptance marker after a failed validation
        # allows downstream optimization to consume an invalid package.
        identity = ROMContractTests.identity()
        passed = self.failed_holdouts()
        for holdout in passed:
            holdout["rom"] = {
                "converged": True,
                "final_period_grid_c": [[30.0, 30.5]],
                "last_period_peak_c": 30.5, "safe": True,
            }
            holdout["hotspot"] = {
                "converged": True,
                "final_period_grid_c": [[30.25, 30.75]],
                "last_period_peak_c": 30.75, "safe": True,
            }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            accepted = validate_holdouts(
                passed, self.settings(), output_dir=output, identity=identity
            )
            self.assertTrue(accepted["accepted"])
            self.assertTrue((output / "validation_report.json").is_file())
            self.assertTrue((output / "rom_acceptance.json").is_file())

            rejected = validate_holdouts(
                self.failed_holdouts(), self.settings(), output_dir=output,
                identity=identity,
            )
            self.assertFalse(rejected["accepted"])
            self.assertTrue((output / "validation_report.json").is_file())
            self.assertFalse((output / "rom_acceptance.json").exists())

            with self.assertRaisesRegex(ValueError, "canonical R1 metadata hash"):
                validate_holdouts(
                    passed, self.settings(), output_dir=output,
                    identity={"power_trace": "sha256:power"},
                )
            self.assertFalse((output / "rom_acceptance.json").exists())

    def _write_holdout_cases(self, root: Path, model: StateSpaceModel,
                             trace_names: tuple[str, ...]) -> tuple[dict, list[dict]]:
        config = {"frequency": {"f0_ghz": 2.0, "fmin_ghz": 1.0,
                                "ambient_c": 25.0, "tsafe_c": 100.0}}
        frequencies = (2.0, 1.6)

        def sha256(path: Path) -> str:
            return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

        cases = []
        for index in range(2):
            frequency_ghz = frequencies[index]
            expected = evaluate_layout_rom(
                model, self.design(), self.power_windows(), self.layout(),
                frequency_ghz, self.settings(), config,
            )
            case_dir = root / f"h{index}"
            case_dir.mkdir()
            layout_path = case_dir / "layout.json"
            power_path = case_dir / "power.json"
            trace_path = case_dir / "transient.ttrace"
            write_json(layout_path, self.layout())
            write_json(power_path, self.power_windows())
            rows = [expected["period_start_grid_c"], expected["final_period_grid_c"][-1]]
            trace_path.write_text(
                "\t".join(trace_names) + "\n"
                + "\n".join(
                    "\t".join(f"{value + 273.15:.15f}" for value in row)
                    for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            convergence = summarize_period_end_convergence(
                [[value + 273.15 for value in row] for row in rows], 1
            )
            cases.append({
                "id": f"h{index}",
                "point": {"id": f"h{index}", "tier": 0,
                          "x_mm": 0.0, "y_mm": 0.0},
                "frequency_ghz": frequency_ghz,
                "frequency_scale": frequency_ghz / 2.0,
                "window_count": 2,
                "artifacts": {
                    "layout": str(layout_path),
                    "power_windows": str(power_path),
                    "temperature_trace": str(trace_path),
                    "sha256": {
                        "layout": sha256(layout_path),
                        "power_windows": sha256(power_path),
                        "temperature_trace": sha256(trace_path),
                    },
                },
                "periodic_steady_state": {
                    "period_repeats": self.settings().pss_period_repeats,
                    "pss_tolerance_c": self.settings().pss_tolerance_c,
                    "full_grid_converged": True,
                    "grid_unit_names": list(trace_names),
                    "evidence": convergence,
                },
            })
        return config, cases

    def test_calibration_holdouts_compare_the_recorded_geometry_input_and_frequency(self):
        # Break caught: validating a prediction against a trace from another
        # placement, power artifact, or frequency can falsely accept the ROM.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, cases = self._write_holdout_cases(
                root, self.model(), ("cell0",)
            )

            report = validate_calibration_holdouts(
                self.model(), self.design(), cases, self.settings(), config,
                output_dir=root / "package", identity=ROMContractTests.identity(),
            )

            self.assertTrue(report["accepted"])
            self.assertTrue((root / "package" / "validation_report.json").is_file())
            self.assertTrue((root / "package" / "rom_acceptance.json").is_file())

            cases[0]["artifacts"]["sha256"]["temperature_trace"] = "sha256:changed"
            rejected = validate_calibration_holdouts(
                self.model(), self.design(), cases, self.settings(), config,
                output_dir=root / "tampered", identity=ROMContractTests.identity(),
            )
            self.assertFalse(rejected["accepted"])
            self.assertIn("hotspot_trace_identity", rejected["failure_reasons"])

    def test_calibration_rejects_reordered_hotspot_grid_before_rmse(self):
        # Break caught: equal-width arrays with a reordered HotSpot header must
        # not be compared positionally against differently ordered POD rows.
        model = replace(
            self.model(), temperature_basis=numpy.ones((2, 1)),
            grid_unit_names=("cell0", "cell1"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, cases = self._write_holdout_cases(
                root, model, ("cell1", "cell0")
            )

            report = validate_calibration_holdouts(
                model, self.design(), cases, self.settings(), config,
                output_dir=root / "package", identity=ROMContractTests.identity(),
            )

            self.assertFalse(report["accepted"])
            self.assertIn("temperature_grid_identity", report["failure_reasons"])
            self.assertIsNone(report["holdouts"][0]["grid_rmse_c"])


class ROMOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.modules = self.root / "modules.json"
        self.power_windows = self.root / "power_windows.json"
        self.config_path = self.root / "config.json"
        self.hotspot = self.root / "hotspot"
        self.canonical_r1 = self.root / "canonical-r1"
        self.accepted_package = self.root / "accepted"
        self.rejected_package = self.root / "rejected"
        self.output = self.root / "output"
        self.accepted_package.mkdir()
        self.rejected_package.mkdir()

        self.model_data = CalibrationDesignTests.model()
        self.model_data["ipc1"] = 2.0
        write_json(self.modules, self.model_data)
        self.canonical_r1.mkdir()
        write_json(self.canonical_r1 / "r1_metadata.json", {"source": "synthetic-r1"})
        self.hotspot.write_text("mock executable", encoding="utf-8")
        write_json(self.config_path, self.config())
        settings = parse_settings({})
        design = build_design(self.model_data, [1], settings)
        self.design = design
        write_json(self.rejected_package / "anchors.json", design)
        self.source_power_windows = ROMCalibrationCaseTests.raw_windows()
        self.source_power_windows["canonical_source_r1"] = str(self.canonical_r1)
        write_json(self.power_windows, self.source_power_windows)
        self._write_self_consistent_package(self.accepted_package, design)
        write_json(self.rejected_package / "rom_acceptance.json", {
            "accepted": False,
        })

    def _write_self_consistent_package(self, package: Path, design: dict) -> None:
        settings = parse_settings({})
        fixed_names = tuple(
            module["name"] for module in self.model_data["modules"]
            if module["kind"] != "l2"
        )
        rom = StateSpaceModel(
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.zeros((1, len(fixed_names))),
            b_l2_anchors={
                point["id"]: numpy.ones((1, 1))
                for point in design["training"]
            },
            module_names=fixed_names,
            grid_unit_names=("cell0",),
        )
        config = read_json(self.config_path)
        physical = config["physical"]
        identity = rom_input_identity(
            canonical_r1_metadata_hash="sha256:" + hashlib.sha256(
                (self.canonical_r1 / "r1_metadata.json").read_bytes()
            ).hexdigest(),
            power_trace=power_trace_identity(self.source_power_windows),
            modules_geometry_hash="sha256:" + hashlib.sha256(
                self.modules.read_bytes()
            ).hexdigest(),
            layout_geometry_hash="sha256:" + hashlib.sha256(json.dumps(
                design["base_layout"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            configuration_hash="sha256:" + hashlib.sha256(
                self.config_path.read_bytes()
            ).hexdigest(),
            hotspot_hash="sha256:" + hashlib.sha256(
                self.hotspot.read_bytes()
            ).hexdigest(),
            grid={"rows": physical["grid_size"], "columns": physical["grid_size"]},
            stack=physical["thermal_stack"],
            cooling={
                "ambient_c": config["frequency"]["ambient_c"],
                "r_convec_k_per_w": physical["r_convec_k_per_w"],
            },
            allowed_l2_tiers=config["layout_optimizer"]["allowed_l2_tiers"],
            calibration_design_hash=canonical_json_sha256(design),
        )
        write_synthetic_accepted_package(
            package, design, identity, settings, rom,
            self.modules, self.config_path, self.source_power_windows,
        )

    def _refresh_package_design(
        self, package: Path, design: dict, *, recompute_validation: bool = True,
    ) -> None:
        """Rehash every persisted design binding after a semantic mutation."""
        design_hash = canonical_json_sha256(design)
        write_json(package / "anchors.json", design)
        manifest_path = package / "calibration_manifest.json"
        manifest = read_json(manifest_path)
        manifest["calibration_design_hash"] = design_hash
        manifest["identity"]["calibration_design_hash"] = design_hash
        write_json(manifest_path, manifest)
        fit_path = package / "fit_report.json"
        fit_report = read_json(fit_path)
        fit_report["calibration_design_hash"] = design_hash
        write_json(fit_path, fit_report)
        model, _ = load_model(package / "pod_model.npz")
        save_model(package / "pod_model.npz", model, fit_report)
        if recompute_validation:
            cases = read_json(package / "calibration_cases.json")
            validation = validate_calibration_holdouts(
                model, design, cases["holdout_cases"], parse_settings({}),
                read_json(self.config_path), output_dir=package, identity=None,
                publish=False,
            )
            self.assertTrue(validation["accepted"])
            write_json(package / "validation_report.json", validation)
        acceptance_path = package / "rom_acceptance.json"
        acceptance = read_json(acceptance_path)
        acceptance["identity"]["calibration_design_hash"] = design_hash
        write_json(acceptance_path, acceptance)
        write_test_artifact_manifest(package)

    @staticmethod
    def config() -> dict:
        return {
            "frequency": {
                "ambient_c": 25.0,
                "f0_ghz": 2.0,
                "fmin_ghz": 1.0,
                "tsafe_c": 50.0,
            },
            "physical": {
                "utilization": 0.70,
                "grid_size": 1,
                "r_convec_k_per_w": 0.1,
                "thermal_stack": {"layers": ["silicon", "tim"]},
            },
            "layout_optimizer": {
                "lambda_wire": 0.25,
                "wire_objective": "continuous",
                "allowed_l2_tiers": [1],
            },
            "delay": {"wire_aggregation": "mean", "wire_rounding": "nearest"},
        }

    def test_frequency_grid_requires_fmin_and_f0_inside_bounds(self):
        # Break caught: a grid omitting a configured endpoint, or including an
        # out-of-range value, lets ROM and final-HotSpot searches differ.
        for values in ([1.2], [0.8, 2.0], [1.0, 2.2], [1.1, 1.9]):
            with self.assertRaisesRegex(ValueError, "frequency grid"):
                canonical_frequency_grid({
                    **self.config(),
                    "transient_rom": {"frequencies_ghz": values},
                })

    @staticmethod
    def _frequency_evidence(_model, _design, _powers, layout, _grid,
                            _settings, _config):
        l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
        frequency = 2.0 if l2["x_mm"] > 0.5 else 1.0
        temperature = 40.0 if frequency == 2.0 else 49.0
        evaluation = {
            "frequency_ghz": frequency,
            "converged": True,
            "last_period_peak_c": temperature,
            "safe": True,
        }
        return {
            "frequency_grid_ghz": [1.0, 2.0],
            "sustainable_frequency_ghz": frequency,
            "state": "thermal_headroom_within_grid",
            "evaluations": [evaluation],
            "grid_evaluations": [evaluation],
            "safe_unsafe_brackets": [],
        }

    @staticmethod
    def _search_must_not_run(*_args, **_kwargs):
        raise AssertionError("ROM search ran before the current identity gate")

    def test_optimizer_selects_higher_rom_bips_without_hotspot(self):
        # Break caught: proxy search, sparse grids, or hidden HotSpot calls can
        # select a layout without the accepted ROM evidence required by Task 7.
        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._frequency_evidence,
        ):
            report = optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertEqual(report["hotspot_calls_inside_optimizer"], 0)
        self.assertEqual(report["search"]["lattice_points_attempted"], 25 * 25)
        self.assertEqual(len(report["search"]["refinements"]), 5)
        self.assertTrue(report["search"]["rejections"])
        self.assertGreater(report["selected"]["f_sus_trans_rom_ghz"], 1.0)
        self.assertEqual(report["selected"]["bips1_trans_rom_pred"], 4.0)
        for candidate in report["search"]["legal_seeds"]:
            expected = (
                -2.0 * candidate["f_sus_trans_rom_ghz"]
                + 0.25 * 2.0 * candidate["wire_objective_cycles"]
            )
            self.assertAlmostEqual(candidate["score"], expected, places=12)
        proposed = self.output / "proposed_layout.json"
        self.assertTrue(proposed.is_file())
        self.assertEqual(read_json(proposed), report["selected_layout"])

    def _discrete_package(self, name, *, grid_steps=3, include_fixed=True):
        self.model_data["communication_profile"] = {
            "status": "available",
            "per_core": {
                "0": {"normalized_weight": 0.4},
                "1": {"normalized_weight": 0.3},
                "2": {"normalized_weight": 0.2},
                "3": {"normalized_weight": 0.1},
            },
        }
        write_json(self.modules, self.model_data)
        config = self.config()
        config["layout_optimizer"].update({
            "lambda_wire": 0.0020119160767721133,
            "wire_objective": "discrete-partition",
            "partition_grid_steps": grid_steps,
            "include_fixed_baseline": include_fixed,
        })
        config["delay"]["wire_aggregation"] = "traffic-weighted"
        write_json(self.config_path, config)
        package = self.root / name
        package.mkdir()
        design = build_design(self.model_data, [1], parse_settings(config))
        self._write_self_consistent_package(package, design)
        return package

    def test_optimizer_uses_nonzero_lambda_traffic_weighted_integer_partitions(self):
        # Break caught: accepting the config but retaining the old continuous
        # 25x25+refinement search would reproduce the rounding bug in ROM mode.
        package = self._discrete_package("discrete-package")
        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._frequency_evidence,
        ):
            report = optimize_transient_layout(
                self.modules, package, self.output, self.config_path,
                self.power_windows, hotspot=self.hotspot,
            )

        self.assertEqual(
            report["parameters"]["wire_objective"], "discrete-partition"
        )
        self.assertEqual(
            report["parameters"]["lambda_wire"], 0.0020119160767721133
        )
        self.assertEqual(
            report["parameters"]["wire_aggregation"], "traffic-weighted"
        )
        self.assertEqual(report["hotspot_calls_inside_optimizer"], 0)
        self.assertTrue(report["search"]["fixed_baseline_included"])
        self.assertTrue(report["search"]["shared_partition_engine"])
        self.assertEqual(
            report["selected"]["wire_objective_cycles"],
            report["selected"]["r2_wire_cycles"],
        )
        self.assertIsInstance(report["selected"]["r2_wire_cycles"], int)
        self.assertTrue((self.output / "partition_search.json").is_file())

    def test_discrete_optimizer_rejects_invalid_controls_before_rom_search(self):
        for index, (steps, include_fixed, message) in enumerate((
            (2, True, "odd integer"),
            (40, True, "odd integer"),
            (3, False, "fixed-bin baseline"),
        )):
            with self.subTest(steps=steps, include_fixed=include_fixed):
                package = self._discrete_package(
                    f"invalid-discrete-{index}", grid_steps=steps,
                    include_fixed=include_fixed,
                )
                output = self.root / f"invalid-output-{index}"
                with patch(
                    "workflow.transient.rom.optimize_layout."
                    "find_rom_sustainable_frequency",
                    side_effect=self._search_must_not_run,
                ), self.assertRaisesRegex(ValueError, message):
                    optimize_transient_layout(
                        self.modules, package, output, self.config_path,
                        self.power_windows, hotspot=self.hotspot,
                    )

    def test_optimizer_loads_saved_design_without_regenerating_it(self):
        # Break caught: rerunning current design-generation code can change the
        # anchor/simplex meaning of B_L2 columns in an already accepted model.
        import workflow.transient.rom.optimize_layout as optimizer_module

        with patch.object(
            optimizer_module, "build_design", create=True,
            side_effect=AssertionError("calibration design was regenerated"),
        ) as regenerate, patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._frequency_evidence,
        ):
            report = optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        regenerate.assert_not_called()
        self.assertEqual(
            report["calibration_design_hash"], canonical_json_sha256(self.design)
        )

    def test_optimizer_rejects_rehashed_shifted_fixed_module_before_search(self):
        # Break caught: self-consistent package hashes must not let calibration
        # geometry for a fixed non-L2 module differ from the current modules.
        design = build_design(self.model_data, [1], parse_settings({}))
        fixed = next(
            module for module in design["base_layout"]["modules"]
            if module["kind"] != "l2"
        )
        fixed["x_mm"] += 0.001
        package = self.root / "shifted-fixed-package"
        self._write_self_consistent_package(package, design)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "base layout fixed module geometry"):
            optimize_transient_layout(
                self.modules, package, self.output, self.config_path,
                self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_rehashed_bilinear_corner_before_search(self):
        # Break caught: a rehashed bilinear corner cannot silently change which
        # named anchor supplies an interpolation column.
        config = self.config()
        config["layout_optimizer"]["allowed_l2_tiers"] = [0, 1]
        write_json(self.config_path, config)
        design = build_design(self.model_data, [0, 1], parse_settings({}))
        package = self.root / "shifted-bilinear-package"
        self._write_self_consistent_package(package, design)
        design["domains"]["0"]["corners"][0][0] += 1e-6
        self._refresh_package_design(
            package, design, recompute_validation=False,
        )

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "domain anchor coordinate"):
            optimize_transient_layout(
                self.modules, package, self.output, self.config_path,
                self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_rehashed_delaunay_point_before_search(self):
        # Break caught: a rehashed Delaunay point cannot silently change which
        # named anchor supplies an interpolation column.
        design = build_design(self.model_data, [1], parse_settings({}))
        package = self.root / "shifted-delaunay-package"
        self._write_self_consistent_package(package, design)
        design["domains"]["1"]["points"][0][0] += 1e-6
        self._refresh_package_design(package, design)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "domain anchor coordinate"):
            optimize_transient_layout(
                self.modules, package, self.output, self.config_path,
                self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_reuses_a_relocated_self_contained_package(self):
        # Break caught: absolute case-artifact paths either read the old package
        # or fail after an accepted package is moved to a read-only reuse root.
        relocated = self.root / "relocated-package"
        self.accepted_package.rename(relocated)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._frequency_evidence,
        ):
            report = optimize_transient_layout(
                self.modules, relocated, self.output, self.config_path,
                self.power_windows, hotspot=self.hotspot,
            )

        self.assertEqual(report["package_dir"], str(relocated.resolve()))
        self.assertTrue((self.output / "proposed_layout.json").is_file())

    def test_optimizer_rejects_saved_design_that_differs_from_acceptance(self):
        # Break caught: loading anchors.json without checking its exact canonical
        # hash still permits a stale acceptance marker to authorize new geometry.
        changed = read_json(self.accepted_package / "anchors.json")
        changed["holdout"][0]["x_mm"] += 0.001
        write_json(self.accepted_package / "anchors.json", changed)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "calibration design hash identity"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_unaccepted_package(self):
        # Break caught: consuming a model archive without the holdout acceptance
        # marker bypasses the scientific gate on every optimized result.
        with self.assertRaisesRegex(ValueError, "ROM package is not accepted"):
            optimize_transient_layout(
                self.modules, self.rejected_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

    def test_optimizer_rejects_package_without_saved_holdout_validation(self):
        # Break caught: direct optimizer reuse must not trust rom_acceptance.json
        # while the saved case/error evidence it was derived from is absent.
        (self.accepted_package / "validation_report.json").unlink()

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "validation_report.json"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_training_case_that_differs_from_saved_design(self):
        # Break caught: checking only the eight ids lets a direct optimizer
        # consume case evidence produced for geometry other than its saved anchor.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        cases["training_cases"][0]["point"]["x_mm"] += 0.001
        write_json(cases_path, cases)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "training case.*evidence differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_training_layout_that_differs_from_case_point(self):
        # Break caught: self-consistent file hashes must not let the persisted
        # training layout disagree with the anchor geometry it claims to fit.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        case = cases["training_cases"][0]
        layout_path = self.accepted_package / case["artifacts"]["layout"]
        layout = read_json(layout_path)
        l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
        l2["x_mm"] += 0.001
        write_json(layout_path, layout)
        case["artifacts"]["sha256"]["layout"] = (
            "sha256:" + hashlib.sha256(layout_path.read_bytes()).hexdigest()
        )
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "training case.*layout evidence differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_manifest_with_changed_conversion_threshold(self):
        # Break caught: a package manifest that does not record the exact fitted
        # conversion gate cannot be reused under the current configuration.
        manifest_path = self.accepted_package / "calibration_manifest.json"
        manifest = read_json(manifest_path)
        manifest["settings"]["max_logm_error_estimate"] = 1e-4
        write_json(manifest_path, manifest)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "resolved settings"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_manifest_with_changed_source_identity(self):
        # Break caught: a rebuilt inventory must not let calibration evidence
        # claim a different config or HotSpot than the accepted package identity.
        manifest_path = self.accepted_package / "calibration_manifest.json"
        manifest = read_json(manifest_path)
        manifest["sources"]["config_sha256"] = "sha256:different-config"
        write_json(manifest_path, manifest)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "source identities"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_holdout_power_from_another_source_trace(self):
        # Break caught: rehashing a holdout artifact cannot substitute power
        # windows whose scientific identity differs from the accepted workload.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        case = cases["holdout_cases"][0]
        power_path = self.accepted_package / case["artifacts"]["power_windows"]
        power = read_json(power_path)
        module = power["windows"][0]["modules"][0]
        module["dynamic_power_w"] += 0.125
        module["total_power_w"] += 0.125
        power["windows"][0]["totals"]["dynamic_power_w"] += 0.125
        power["windows"][0]["totals"]["total_power_w"] += 0.125
        write_json(power_path, power)
        case["artifacts"]["sha256"]["power_windows"] = (
            "sha256:" + hashlib.sha256(power_path.read_bytes()).hexdigest()
        )
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "source power identity differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_training_power_not_derived_from_bound_source(self):
        # Break caught: truthful PRBS metadata and self-consistent hashes cannot
        # substitute excitation values that were not derived from source power.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        case = cases["training_cases"][0]
        power_path = self.accepted_package / case["artifacts"]["power_windows"]
        power = read_json(power_path)
        module = power["windows"][0]["modules"][0]
        module["dynamic_power_w"] += 0.125
        module["total_power_w"] += 0.125
        power["windows"][0]["totals"]["dynamic_power_w"] += 0.125
        power["windows"][0]["totals"]["total_power_w"] += 0.125
        power["timeline_audit"] = validate_power_windows(power)
        power["power_trace_identity"] = power_trace_identity(power)
        write_json(power_path, power)
        power_hash = "sha256:" + hashlib.sha256(power_path.read_bytes()).hexdigest()
        case["artifacts"]["sha256"]["power_windows"] = power_hash
        write_json(cases_path, cases)

        fit_path = self.accepted_package / "fit_report.json"
        fit_report = read_json(fit_path)
        fit_report["power_windows_sha256"][case["id"]] = power_hash
        write_json(fit_path, fit_report)
        model, _ = load_model(self.accepted_package / "pod_model.npz")
        save_model(self.accepted_package / "pod_model.npz", model, fit_report)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "PRBS power evidence differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_holdout_without_recorded_frequency(self):
        # Break caught: an accepted marker cannot stand in for the real
        # frequency at which a reusable holdout trace was produced.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        cases["holdout_cases"][0].pop("frequency_ghz")
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "invalid frequency_ghz"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_one_row_holdout_trace_after_rehash(self):
        # Break caught: a one-row trace cannot prove repeated-period full-grid
        # PSS even when every package and case hash has been regenerated.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        case = cases["holdout_cases"][0]
        trace_path = self.accepted_package / case["artifacts"]["temperature_trace"]
        trace_path.write_text("cell0\n300.0\n", encoding="utf-8")
        case["artifacts"]["sha256"]["temperature_trace"] = (
            "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest()
        )
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "temperature trace period structure"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_recorded_holdout_pss_metrics_that_differ(self):
        # Break caught: declarative full_grid_converged cannot override the
        # convergence metrics recomputed from the immutable temperature trace.
        cases_path = self.accepted_package / "calibration_cases.json"
        cases = read_json(cases_path)
        cases["holdout_cases"][0]["periodic_steady_state"]["evidence"][
            "period_count"
        ] = 999
        write_json(cases_path, cases)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "recomputed holdout validation differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_holdout_peak_not_derived_from_trace(self):
        # Break caught: an accepted validation report cannot assert a HotSpot
        # peak that differs from the inclusive final-period trace peak.
        validation_path = self.accepted_package / "validation_report.json"
        validation = read_json(validation_path)
        validation["holdouts"][0]["hotspot_peak_c"] += 1.0
        write_json(validation_path, validation)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "recomputed holdout validation differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_holdout_safety_not_derived_from_trace(self):
        # Break caught: the saved safety class must be recomputed from the real
        # trace peak and the bound configuration's thermal safety threshold.
        validation_path = self.accepted_package / "validation_report.json"
        validation = read_json(validation_path)
        validation["holdouts"][0]["hotspot_safe"] = False
        write_json(validation_path, validation)
        write_test_artifact_manifest(self.accepted_package)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "recomputed holdout validation differs"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def assert_persisted_holdout_forgery_rejected(self, mutate) -> None:
        validation_path = self.accepted_package / "validation_report.json"
        validation = read_json(validation_path)
        mutate(validation["holdouts"][0])
        write_json(validation_path, validation)
        write_test_artifact_manifest(self.accepted_package)
        forged_bytes = validation_path.read_bytes()

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(
            ValueError, "recomputed holdout validation differs"
        ):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertEqual(validation_path.read_bytes(), forged_bytes)
        self.assertFalse(self.output.exists())

    def test_optimizer_recomputes_and_rejects_forged_rom_peak(self):
        self.assert_persisted_holdout_forgery_rejected(
            lambda holdout: holdout.__setitem__("rom_peak_c", 49.5)
        )

    def test_optimizer_recomputes_and_rejects_forged_rom_safety(self):
        self.assert_persisted_holdout_forgery_rejected(
            lambda holdout: holdout.__setitem__("rom_safe", False)
        )

    def test_optimizer_recomputes_and_rejects_forged_peak_error(self):
        self.assert_persisted_holdout_forgery_rejected(
            lambda holdout: holdout.__setitem__(
                "peak_temperature_error_c", 0.0
            )
        )

    def test_optimizer_recomputes_gates_and_accepted_flag(self):
        def forge(holdout: dict) -> None:
            holdout["rom_safe"] = False
            holdout["gates"]["safety_classification"] = True
            holdout["accepted"] = True

        self.assert_persisted_holdout_forgery_rejected(forge)

    def test_optimizer_rejects_changed_current_power_trace_before_search(self):
        # Break caught: copying power identity out of the acceptance marker lets
        # a different current workload drive a ROM accepted for another trace.
        power = read_json(self.power_windows)
        module = power["windows"][0]["modules"][0]
        module["dynamic_power_w"] += 0.125
        module["total_power_w"] += 0.125
        power["windows"][0]["totals"]["dynamic_power_w"] += 0.125
        power["windows"][0]["totals"]["total_power_w"] += 0.125
        write_json(self.power_windows, power)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "power trace identity"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())

    def test_optimizer_rejects_changed_current_config_before_search(self):
        # Break caught: copying configuration_hash out of acceptance makes a
        # changed thermal environment appear compatible with the accepted ROM.
        config = self.config()
        config["frequency"]["ambient_c"] = 26.0
        write_json(self.config_path, config)

        with patch(
            "workflow.transient.rom.optimize_layout.find_rom_sustainable_frequency",
            side_effect=self._search_must_not_run,
        ), self.assertRaisesRegex(ValueError, "configuration hash identity"):
            optimize_transient_layout(
                self.modules, self.accepted_package, self.output,
                self.config_path, self.power_windows, hotspot=self.hotspot,
            )

        self.assertFalse(self.output.exists())


class ROMPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_r1 = self.root / "source_r1"
        self.steady = self.root / "steady_preflight"
        self.output = self.root / "transient_rom"
        self.transient_r1 = self.root / "periodic_r1"
        self.config_path = self.root / "config.json"
        self.hotspot = self.root / "hotspot"
        self.modules = self.steady / "modules.json"
        self.cacti = self.steady / "cacti/cacti_characterization.json"
        self.power_windows = self.output / "windows/mcpat/power_windows.json"
        self.proposed_layout = self.output / "optimization/proposed_layout.json"
        for directory in (self.source_r1, self.steady, self.transient_r1):
            directory.mkdir(parents=True)
        self.cacti.parent.mkdir(parents=True)
        self.hotspot.write_text("mock hotspot", encoding="utf-8")
        module_model = CalibrationDesignTests.model()
        module_model["ipc1"] = 2.0
        write_json(self.modules, module_model)
        write_json(self.cacti, {"frequency_ghz": 2.0, "records": []})
        write_json(self.config_path, self.config())

    @staticmethod
    def config() -> dict:
        return {
            "schema_version": 1,
            "name": "transient-rom-test",
            "frequency": {
                "ambient_c": 25.0,
                "f0_ghz": 2.0,
                "fmin_ghz": 1.0,
                "tsafe_c": 50.0,
            },
            "physical": {
                "grid_size": 1,
                "utilization": 0.7,
                "r_convec_k_per_w": 0.1,
                "thermal_stack": {"layers": ["silicon", "tim"]},
            },
            "layout_optimizer": {
                "allowed_l2_tiers": [1],
                "wire_objective": "continuous",
                "lambda_wire": 0.0,
            },
            "delay": {
                "wire_rounding": "nearest",
                "wire_aggregation": "mean",
                "cycles_per_tsv": 2,
                "l1_pipeline_cycles": 1,
            },
            "transient_rom": {
                "enabled": True,
                "calibration_runs": 8,
                "validation_runs": 2,
                "frequencies_ghz": [1.0, 1.8, 2.0],
            },
        }

    def prepared(self) -> dict:
        self.power_windows.parent.mkdir(parents=True, exist_ok=True)
        if not self.power_windows.is_file():
            write_json(self.power_windows, {"mock": "power windows"})
        return {
            "power_windows": str(self.power_windows.resolve()),
            "transient_r1": str(self.transient_r1.resolve()),
            "power_trace_identity": "sha256:power",
            "stage_seconds": {"windowed_mcpat": 1.0},
        }

    def optimization(self) -> dict:
        design = build_design(
            CalibrationDesignTests.model(), [1], parse_settings({})
        )
        self.proposed_layout.parent.mkdir(parents=True, exist_ok=True)
        if not self.proposed_layout.is_file():
            write_json(self.proposed_layout, {"mock": "proposed layout"})
        package_acceptance = {
            "schema_version": 1,
            "thermal_mode": "transient-rom",
            "non_formal": True,
            "paper_equivalent": False,
            "accepted": True,
            "identity": {
                **ROMContractTests.identity(),
                "calibration_design_hash": canonical_json_sha256(design),
            },
            "validation_report": "validation_report.json",
        }
        acceptance_path = self.output / "rom_package/rom_acceptance.json"
        if acceptance_path.is_file():
            candidate = read_json(acceptance_path)
            candidate_identity = (
                candidate.get("identity") if isinstance(candidate, dict) else None
            )
            if isinstance(candidate_identity, dict) and all(
                field in candidate_identity for field in ROMContractTests.identity()
            ):
                package_acceptance = candidate
        return {
            "hotspot_calls_inside_optimizer": 0,
            "proposed_layout": str(self.proposed_layout.resolve()),
            "package_acceptance": package_acceptance,
            "parameters": {"frequency_grid_ghz": [1.0, 1.8, 2.0]},
            "selected": {
                "f_sus_trans_rom_ghz": 1.9,
                "bips1_trans_rom_pred": 3.8,
            },
        }

    @staticmethod
    def final_validation_trace_rows(
        frequency_ghz: float, profile: str,
    ) -> list[list[float]]:
        """Return independent two-window, two-cell temperature facts in C."""
        if profile == "baseline":
            last_period_initial_peak_c = (
                49.0 if frequency_ghz <= 1.8 else 51.0
            )
            period_end_peaks_c = [
                last_period_initial_peak_c + (18 - index) * 0.005
                for index in range(20)
            ]
        elif profile == "nonconverged":
            last_period_peak_c = 48.0 + 0.5 * (frequency_ghz - 1.0)
            period_end_peaks_c = [
                last_period_peak_c - (19 - index)
                for index in range(20)
            ]
        elif profile == "all_unsafe":
            last_period_initial_peak_c = 51.0 + 0.5 * (frequency_ghz - 1.0)
            period_end_peaks_c = [
                last_period_initial_peak_c + (18 - index) * 0.005
                for index in range(20)
            ]
        else:
            raise ValueError(f"unknown final validation trace profile: {profile}")

        rows_c = []
        for period_end_peak_c in period_end_peaks_c:
            rows_c.extend([
                [period_end_peak_c - 1.5, period_end_peak_c - 0.75],
                [period_end_peak_c - 0.5, period_end_peak_c],
            ])
        return rows_c

    def write_final_validation_trace(
        self, output_dir: Path, frequency_ghz: float, profile: str,
        *, modules_path: Path | None = None, layout_path: Path | None = None,
        power_windows_path: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Write trace facts without consulting any mocked search evaluation."""
        rows_c = self.final_validation_trace_rows(frequency_ghz, profile)
        case_dir = output_dir / f"frequency_{frequency_ghz.hex()}_ghz"
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "window_count": len(rows_c),
            "windows_per_period": 2,
            "period_repeats": 20,
            "grid_cell_count": 2,
            "frequency_scaling": {
                "frequency_scale": frequency_ghz / 2.0,
            },
        }
        if all(path is not None for path in (
            modules_path, layout_path, power_windows_path, config_path,
        )):
            trace_config = case_dir / "trace_input_config.json"
            write_json(trace_config, read_json(config_path))
            manifest.update({
                "source_modules": str(Path(modules_path).resolve()),
                "source_layout": str(Path(layout_path).resolve()),
                "source_power_windows": str(Path(power_windows_path).resolve()),
                "trace_input_identity": trace_input_identity(
                    Path(modules_path), Path(layout_path),
                    Path(power_windows_path), trace_config,
                    frequency_ghz / 2.0,
                ),
            })
        write_json(case_dir / "transient_trace_manifest.json", manifest)
        (case_dir / "transient.ttrace").write_text(
            "cell0\tcell1\n"
            + "\n".join(
                "\t".join(f"{temperature_c + 273.15:.17g}" for temperature_c in row)
                for row in rows_c
            )
            + "\n",
            encoding="utf-8",
        )

    def final_validation(
        self, profile: str = "baseline", output_dir: Path | None = None,
        *, modules_path: Path | None = None, layout_path: Path | None = None,
        power_windows_path: Path | None = None,
        config_path: Path | None = None, hotspot_path: Path | None = None,
    ) -> dict:
        """Derive a canonical search solely from materialized trace facts."""
        grid = [1.0, 1.8, 2.0]
        artifact_root = (
            self.root / f"derived_final_validation_{profile}"
            if output_dir is None else Path(output_dir)
        )
        artifact_root.mkdir(parents=True, exist_ok=True)

        def evaluate(frequency_ghz: float) -> dict:
            self.write_final_validation_trace(
                artifact_root, frequency_ghz, profile,
                modules_path=modules_path, layout_path=layout_path,
                power_windows_path=power_windows_path,
                config_path=config_path,
            )
            case_dir = artifact_root / f"frequency_{frequency_ghz.hex()}_ghz"
            names, rows_k = parse_ttrace_grid(case_dir / "transient.ttrace")
            convergence = summarize_period_end_convergence(rows_k, 2)
            peak = last_period_peak(rows_k, 2)
            return {
                "frequency_ghz": frequency_ghz,
                "converged": (
                    convergence["period_count"] >= 2
                    and convergence["last_delta_max_c"] <= 0.01
                ),
                "last_period_peak_c": peak["tmax_c"],
                "last_period_peak_unit": names[peak["unit_index"]],
                "period_end_convergence": convergence,
                "trace_peak_c": max(
                    temperature for row in rows_k for temperature in row
                ) - 273.15,
            }

        search = find_sustainable_frequency(evaluate, grid, 50.0, 0.01)
        result = {
            "schema_version": 1,
            "state": search["state"],
            "f_sus_trans_ghz": search["sustainable_frequency_ghz"],
            "frequency": {"f0_ghz": 2.0, "tsafe_c": 50.0, "grid_ghz": grid},
            "periodic_steady_state": {
                "period_repeats": 20, "pss_tolerance_c": 0.01,
            },
            "search": search,
        }
        if all(path is not None for path in (
            modules_path, layout_path, power_windows_path, config_path,
            hotspot_path,
        )):
            result.update({
                "modules": str(Path(modules_path).resolve()),
                "layout": str(Path(layout_path).resolve()),
                "power_windows": str(Path(power_windows_path).resolve()),
                "config": str(Path(config_path).resolve()),
                "hotspot": str(Path(hotspot_path).resolve()),
            })
        write_json(
            artifact_root / "transient_sustainable_frequency.json", result,
        )
        return result

    def final_validation_side_effect(
        self, result: object = _USE_DERIVED_FINAL_VALIDATION, *,
        profile: str = "baseline", artifact_mutation=None,
    ):
        def run(*args, **_kwargs):
            output_dir = Path(args[3])
            derived = self.final_validation(
                profile, output_dir, modules_path=Path(args[0]),
                layout_path=Path(args[1]), power_windows_path=Path(args[2]),
                config_path=Path(args[4]), hotspot_path=Path(_kwargs["hotspot"]),
            )
            if artifact_mutation is not None:
                artifact_mutation(output_dir, derived)
            return (
                derived
                if result is _USE_DERIVED_FINAL_VALIDATION
                else result
            )

        return run

    def run_with_final_validation(
        self, final_validation: object = _USE_DERIVED_FINAL_VALIDATION,
        *, artifact_mutation=None,
    ):
        """Run the pipeline to the final validation boundary with tools mocked."""
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        self.bind_package(package)
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2",
            return_value={"ipc2": 1.5},
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(
                final_validation, artifact_mutation=artifact_mutation,
            ),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )
        return result, build_vector_mock, run_r2_mock

    def completed_rom_summary(self) -> dict:
        """Produce an audited completed-run summary with external work mocked."""
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        write_json(package / "rom_acceptance.json", {"accepted": True})
        self.bind_package(package)
        write_json(self.power_windows, {"mock": "power windows"})
        write_json(self.proposed_layout, {"mock": "proposed layout"})
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ), patch(
            "workflow.transient.rom.run_pipeline.run_r2",
            return_value={"ipc2": 1.5},
        ), patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(),
        ):
            return run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

    def steady_summary(self, *, cacti_path: Path | None = None,
                       cacti_sha256: str | None = None,
                       **overrides: object) -> dict:
        summary = {
            "layout_method": "fixed-bin",
            "ipc2": None,
            "bips2": None,
            "r2_source": None,
            "artifacts": {
                "cacti": str((cacti_path or self.cacti).resolve()),
            },
            "artifact_sha256": {
                "cacti": cacti_sha256 or hashlib.sha256(
                    self.cacti.read_bytes()
                ).hexdigest(),
            },
        }
        summary.update(overrides)
        return summary

    def bind_package(self, package: Path) -> None:
        source_model = CalibrationDesignTests.model()
        settings = parse_settings({})
        design = build_design(source_model, [1], settings)
        design_identity = canonical_json_sha256(design)
        identity = {
            **ROMContractTests.identity(),
            "modules_geometry_hash": "sha256:" + hashlib.sha256(
                self.modules.read_bytes()
            ).hexdigest(),
            "configuration_hash": "sha256:" + hashlib.sha256(
                self.config_path.read_bytes()
            ).hexdigest(),
            "power_trace": power_trace_identity(
                ROMCalibrationCaseTests.raw_windows()
            ),
            "calibration_design_hash": design_identity,
        }
        fixed_names = tuple(
            module["name"] for module in source_model["modules"]
            if module["kind"] != "l2"
        )
        model = StateSpaceModel(
            temperature_basis=numpy.ones((1, 1)),
            a_continuous=numpy.array([[-1.0]]),
            b_fixed=numpy.zeros((1, len(fixed_names))),
            b_l2_anchors={
                point["id"]: numpy.ones((1, 1))
                for point in design["training"]
            },
            module_names=fixed_names,
            grid_unit_names=("cell0",),
        )
        write_synthetic_accepted_package(
            package, design, identity, settings, model, self.modules,
            self.config_path,
            ROMCalibrationCaseTests.raw_windows(),
        )

    def test_pipeline_prepares_windows_once_reuses_package_and_validates_final_layout(self):
        # Break caught: recomputing power windows per layout, deriving R2 from
        # the fixed pilot, or reporting steady bips2 would invalidate comparisons.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        write_json(package / "rom_acceptance.json", {"accepted": True})
        self.bind_package(package)
        write_json(self.power_windows, {"mock": "power windows"})
        write_json(self.proposed_layout, {"mock": "proposed layout"})
        events = []

        def build_vector_side_effect(*args, **kwargs):
            events.append("r2-vector")
            return {"critical_l1d_to_l2_cycles": 7}

        def run_r2_side_effect(*args, **kwargs):
            events.append("r2")
            return {"ipc2": 1.5}

        def final_side_effect(*args, **kwargs):
            events.append("hotspot")
            return self.final_validation(
                output_dir=Path(args[3]), modules_path=Path(args[0]),
                layout_path=Path(args[1]), power_windows_path=Path(args[2]),
                config_path=Path(args[4]), hotspot_path=Path(kwargs["hotspot"]),
            )

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ) as prepare, patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ) as optimize, patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            side_effect=build_vector_side_effect,
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2",
            side_effect=run_r2_side_effect,
        ), patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=final_side_effect,
        ) as final_search:
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

        prepare.assert_called_once()
        optimize.assert_called_once()
        self.assertEqual(events, ["hotspot", "r2-vector", "r2"])
        self.assertEqual(
            build_vector_mock.call_args.args[5], self.proposed_layout.resolve()
        )
        self.assertEqual(
            final_search.call_args.args[1], self.proposed_layout.resolve()
        )
        self.assertEqual(result["f_sus_trans_rom_pred_ghz"], 1.9)
        self.assertEqual(result["f_sus_trans_hotspot_ghz"], 1.8)
        self.assertEqual(result["bips1_trans_rom_pred"], 3.8)
        self.assertEqual(result["bips2_trans"], 2.7)
        self.assertEqual(
            result["rom_acceptance"]["identity"]["power_trace"],
            power_trace_identity(ROMCalibrationCaseTests.raw_windows()),
        )
        keys = []

        def collect_keys(value):
            if isinstance(value, dict):
                keys.extend(value)
                for nested in value.values():
                    collect_keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_keys(nested)

        collect_keys(result)
        self.assertNotIn("bips2", keys)
        self.assertTrue(result["non_formal"])
        self.assertFalse(result["paper_equivalent"])
        self.assertEqual(result["thermal_mode"], "transient-rom")
        self.assertEqual(
            read_json(self.output / "transient_rom_summary.json"), result
        )
        artifact_manifest = read_json(self.output / "rom_artifact_manifest.json")
        self.assertTrue(artifact_manifest["non_formal"])
        self.assertFalse(artifact_manifest["paper_equivalent"])
        self.assertEqual(artifact_manifest["thermal_mode"], "transient-rom")
        records = {record["path"]: record for record in artifact_manifest["artifacts"]}
        for relative in (
            "windows/mcpat/power_windows.json",
            "optimization/proposed_layout.json",
            "transient_rom_summary.json",
        ):
            self.assertEqual(records[relative]["classification"], {
                "thermal_mode": "transient-rom",
                "non_formal": True,
                "paper_equivalent": False,
            })

    def test_summary_separates_predicted_and_validated_transient_results(self):
        summary = self.completed_rom_summary()
        self.assertTrue(summary["non_formal"])
        self.assertFalse(summary["paper_equivalent"])
        self.assertIn("bips1_trans_rom_pred", summary)
        self.assertIn("bips2_trans", summary)
        self.assertEqual(summary["bips1_trans_rom_pred"], 3.8)
        self.assertEqual(summary["bips2_trans"], 2.7)
        self.assertNotEqual(
            summary["bips1_trans_rom_pred"], summary["bips2_trans"]
        )
        self.assertNotIn("bips2", summary)

    def test_pipeline_rejects_reused_steady_preflight_with_r2_measurement(self):
        # Break caught: a reused pilot with R2 is not the fixed-bin/R2-disabled
        # preflight required by ROM mode and can leak the wrong-layout IPC2.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={
                "summary": self.steady_summary(ipc2=1.25, bips2=2.5),
            },
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows"
        ) as prepare:
            with self.assertRaisesRegex(ValueError, "R2-disabled"):
                run_transient_rom_pipeline(
                    self.source_r1, self.steady, self.output, self.config_path,
                    self.transient_r1, calibrate=False, execute_r2=False,
                )

        prepare.assert_not_called()

    def test_final_validation_fixture_derives_multicell_search_and_allows_r2(self):
        # Break caught: synthesizing trace artifacts from the mocked search
        # result makes the artifact-binding check circular instead of proving
        # that canonical multi-window, multi-cell trace evidence supports R2.
        result, _build_vector_mock, run_r2_mock = self.run_with_final_validation()

        validation = read_json(
            self.output
            / "final_hotspot_validation/transient_sustainable_frequency.json"
        )
        evaluations = validation["search"]["evaluations"]
        self.assertGreater(len(evaluations), 3)
        for evaluation in evaluations:
            frequency = evaluation["frequency_ghz"]
            case_dir = (
                self.output / "final_hotspot_validation"
                / f"frequency_{frequency.hex()}_ghz"
            )
            manifest = read_json(case_dir / "transient_trace_manifest.json")
            names, rows_k = parse_ttrace_grid(case_dir / "transient.ttrace")
            self.assertEqual(manifest["windows_per_period"], 2)
            self.assertEqual(manifest["grid_cell_count"], 2)
            self.assertEqual(names, ["cell0", "cell1"])
            self.assertEqual(len(rows_k), 40)
            peak = last_period_peak(rows_k, 2)
            self.assertEqual(peak["phase"], "period_initial_state")
            self.assertEqual(
                evaluation["last_period_peak_c"], peak["tmax_c"]
            )
            self.assertEqual(
                evaluation["period_end_convergence"],
                summarize_period_end_convergence(rows_k, 2),
            )

        run_r2_mock.assert_called_once()
        self.assertEqual(result["final_validation_classification"], "validated")
        self.assertTrue(result["r2_executed"])

    def test_pipeline_rejects_steady_preflight_cacti_path_before_power_or_r2(self):
        # Break caught: a steady summary pointing at a different CACTI result
        # must not silently authorize the expected-path file for R2 derivation.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        other_cacti = self.root / "other/cacti_characterization.json"
        other_cacti.parent.mkdir(parents=True)
        write_json(other_cacti, {"frequency_ghz": 9.0, "records": []})
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary(cacti_path=other_cacti)},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows"
        ) as prepare, patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock:
            with self.assertRaisesRegex(ValueError, "CACTI artifact path"):
                run_transient_rom_pipeline(
                    self.source_r1, self.steady, self.output, self.config_path,
                    self.transient_r1, calibrate=False, execute_r2=False,
                )

        prepare.assert_not_called()
        build_vector_mock.assert_not_called()

    def test_pipeline_rejects_replaced_steady_preflight_cacti_before_power_or_r2(self):
        # Break caught: replacing CACTI after the steady summary was written
        # must invalidate its recorded identity before any ROM/R2 work begins.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        steady_summary = self.steady_summary()
        write_json(self.cacti, {"frequency_ghz": 7.0, "records": []})
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": steady_summary},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows"
        ) as prepare, patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock:
            with self.assertRaisesRegex(ValueError, "CACTI artifact sha256"):
                run_transient_rom_pipeline(
                    self.source_r1, self.steady, self.output, self.config_path,
                    self.transient_r1, calibrate=False, execute_r2=False,
                )

        prepare.assert_not_called()
        build_vector_mock.assert_not_called()

    def test_final_pss_nonconvergence_is_audited_and_skips_r2(self):
        # Break caught: treating a nonconverged PSS point as thermal infeasibility
        # hides a validation failure and can still launch/report an R2 result.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        write_json(package / "rom_acceptance.json", {"accepted": True})
        self.bind_package(package)
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2"
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(
                profile="nonconverged"
            ),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "rom_final_validation_failed")
        self.assertEqual(
            result["final_validation_failure"]["category"], "pss_nonconvergence"
        )
        self.assertEqual(result["f_sus_trans_rom_pred_ghz"], 1.9)
        self.assertEqual(result["bips1_trans_rom_pred"], 3.8)
        self.assertIsNone(result["f_sus_trans_hotspot_ghz"])
        self.assertIn("bips2_trans", result)
        self.assertIsNone(result["bips2_trans"])
        self.assertFalse(result["r2_executed"])
        self.assertNotIn("bips2", result)
        self.assertEqual(
            read_json(self.output / "transient_rom_summary.json"), result
        )

    def test_final_hotspot_tool_error_writes_failure_summary_and_skips_r2(self):
        # Break caught: propagating HotSpot errors without a summary loses the ROM
        # prediction and leaves no auditable reason that bips2_trans is absent.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        self.bind_package(package)

        def fail_after_launch(*args, **kwargs):
            case = Path(args[3]) / "frequency_0x1.0000000000000p+0_ghz"
            case.mkdir(parents=True)
            (case / "hotspot_transient.log").write_text(
                "HotSpot failed\n", encoding="utf-8"
            )
            raise RuntimeError("transient HotSpot failed (rc=9)")

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2"
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=fail_after_launch,
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "rom_final_validation_failed")
        self.assertEqual(result["final_validation_failure"]["category"], "tool_error")
        self.assertEqual(
            result["final_validation_failure"]["error_type"], "RuntimeError"
        )
        self.assertIn("rc=9", result["final_validation_failure"]["message"])
        self.assertEqual(result["f_sus_trans_rom_pred_ghz"], 1.9)
        self.assertEqual(result["bips1_trans_rom_pred"], 3.8)
        self.assertEqual(result["final_validation_hotspot_calls"], 1)
        self.assertIsNone(result["bips2_trans"])
        self.assertTrue((self.output / "transient_rom_summary.json").is_file())

    def test_final_validation_contract_error_is_not_labeled_as_tool_failure(self):
        # Break caught: malformed trace/search evidence is a validation-contract
        # failure, not proof that the HotSpot executable itself failed.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        self.bind_package(package)
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2"
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=ValueError("final temperature grid order differs"),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "rom_final_validation_failed")
        self.assertEqual(
            result["final_validation_failure"]["category"],
            "validation_contract_error",
        )
        self.assertEqual(
            result["final_validation_failure"]["error_type"], "ValueError"
        )

    def assert_final_validation_contract_failure(self, returned: object) -> dict:
        result, build_vector_mock, run_r2_mock = self.run_with_final_validation(
            returned
        )
        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "rom_final_validation_failed")
        self.assertEqual(
            result["final_validation_failure"]["category"],
            "validation_contract_error",
        )
        failure_path = self.output / "final_hotspot_validation/final_validation_failure.json"
        self.assertTrue(failure_path.is_file())
        self.assertEqual(
            read_json(failure_path)["category"], "validation_contract_error"
        )
        return result

    def test_final_non_dictionary_result_is_validation_contract_failure(self):
        # Break caught: `.get()` on an unchecked final-search return escaped the
        # failure writer and left no ROM-preserving summary.
        self.assert_final_validation_contract_failure(None)

    def test_final_missing_evaluations_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["search"].pop("evaluations")

        self.assert_final_validation_contract_failure(returned)

    def test_final_invalid_evaluations_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["search"]["evaluations"] = {"not": "a list"}

        self.assert_final_validation_contract_failure(returned)

    def test_final_missing_sustainable_frequency_is_contract_failure(self):
        returned = self.final_validation()
        returned["f_sus_trans_ghz"] = None
        returned["search"]["sustainable_frequency_ghz"] = None

        self.assert_final_validation_contract_failure(returned)

    def test_final_nonfinite_sustainable_frequency_is_contract_failure(self):
        returned = self.final_validation()
        returned["f_sus_trans_ghz"] = float("nan")
        returned["search"]["sustainable_frequency_ghz"] = float("nan")

        self.assert_final_validation_contract_failure(returned)

    def test_final_inconsistent_state_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["state"] = "thermally_infeasible"
        returned["search"]["state"] = "thermally_infeasible"

        self.assert_final_validation_contract_failure(returned)

    def test_final_inconsistent_safety_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["search"]["evaluations"][0]["safe"] = False
        returned["search"]["grid_evaluations"][0]["safe"] = False

        self.assert_final_validation_contract_failure(returned)

    def test_final_noncanonical_refined_bracket_is_contract_failure(self):
        returned = self.final_validation()
        returned["search"]["safe_unsafe_brackets"][0][
            "refined_safe_frequency_ghz"
        ] = 1.7

        self.assert_final_validation_contract_failure(returned)

    def test_final_pss_evidence_inconsistent_with_convergence_is_contract_failure(self):
        returned = self.final_validation()
        evidence = returned["search"]["evaluations"][0][
            "period_end_convergence"
        ]
        evidence["last_delta_max_c"] = 0.02
        evidence["period_end_deltas"][-1]["delta_max_c"] = 0.02

        self.assert_final_validation_contract_failure(returned)

    def test_final_trace_mutation_is_contract_failure_and_skips_r2(self):
        # Break caught: artifact validation must compare the independently
        # returned search with the trace bytes, not merely replay JSON fields.
        def mutate_trace(output_dir: Path, returned: dict) -> None:
            frequency = returned["search"]["evaluations"][0]["frequency_ghz"]
            trace = (
                output_dir / f"frequency_{frequency.hex()}_ghz/transient.ttrace"
            )
            lines = trace.read_text(encoding="utf-8").splitlines()
            fields = lines[-1].split()
            fields[-1] = f"{float(fields[-1]) + 1.0:.17g}"
            lines[-1] = "\t".join(fields)
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result, build_vector_mock, run_r2_mock = self.run_with_final_validation(
            artifact_mutation=mutate_trace,
        )

        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "rom_final_validation_failed")
        self.assertEqual(
            result["final_validation_failure"]["category"],
            "validation_contract_error",
        )

    def test_final_huge_json_number_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["search"]["evaluations"][0]["last_period_peak_c"] = 10**400

        self.assert_final_validation_contract_failure(returned)

    def test_final_enormous_grid_cell_count_is_validation_contract_failure(self):
        returned = self.final_validation()
        returned["search"]["evaluations"][0]["period_end_convergence"][
            "grid_cell_count"
        ] = 10**400

        self.assert_final_validation_contract_failure(returned)

    def test_all_converged_unsafe_grid_is_true_thermal_infeasible_not_tool_error(self):
        # Break caught: folding a valid unsafe result into tool/nonconvergence
        # failure makes physical infeasibility indistinguishable from bad evidence.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        self.bind_package(package)
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2"
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(
                profile="all_unsafe"
            ),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=True,
            )

        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        self.assertEqual(result["state"], "thermally_infeasible")
        self.assertEqual(
            result["final_validation_classification"], "true_thermal_infeasible"
        )
        self.assertIsNone(result["final_validation_failure"])
        self.assertIsNone(result["bips2_trans"])

    def test_calibration_builds_and_accepts_package_before_optimization(self):
        # Break caught: allowing optimization before the fixed 8+2 holdout gate
        # bypasses the accepted-package contract.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        model = object()
        cases = {
            "training_hotspot_calls": 8,
            "holdout_hotspot_calls": 2,
            "training_cases": [{"id": str(i)} for i in range(8)],
            "holdout_cases": [{"id": "h0"}, {"id": "h1"}],
        }
        events = []

        def validation_side_effect(*args, **kwargs):
            events.append("accepted")
            package = self.output / "rom_package"
            write_json(package / "rom_acceptance.json", {
                "accepted": True, "identity": {"power_trace": "sha256:power"},
            })
            return {"accepted": True, "failure_reasons": []}

        def optimize_side_effect(*args, **kwargs):
            events.append("optimize")
            return self.optimization()

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.build_design",
            return_value={"allowed_l2_tiers": [1], "base_layout": {}},
        ), patch(
            "workflow.transient.rom.run_pipeline.execute_calibration_cases",
            return_value=cases,
        ), patch(
            "workflow.transient.rom.run_pipeline.fit_state_space",
            return_value=(model, {"pod": {"rank": 2}}),
        ), patch(
            "workflow.transient.rom.run_pipeline.save_model"
        ), patch(
            "workflow.transient.rom.run_pipeline.package_identity",
            return_value={"power_trace": "sha256:power"},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_calibration_holdouts",
            side_effect=validation_side_effect,
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            side_effect=optimize_side_effect,
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ), patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=True, execute_r2=False,
            )

        self.assertEqual(events, ["accepted", "optimize"])
        self.assertEqual(result["calibration_hotspot_calls"], 10)
        self.assertEqual(result["rom_package_status"], "calibrated")
        self.assertEqual(
            result["rom_acceptance"]["identity"]["power_trace"],
            "sha256:power",
        )
        self.assertTrue(result["rom_holdout_validation"]["accepted"])
        package_manifest = read_json(
            self.output / "rom_package/rom_artifact_manifest.json"
        )
        self.assertEqual(package_manifest["thermal_mode"], "transient-rom")
        self.assertTrue(package_manifest["non_formal"])
        self.assertFalse(package_manifest["paper_equivalent"])

    def test_calibrated_package_can_be_reused_from_a_fresh_output_root(self):
        # Break caught: coupling the package to a populated optimization root
        # makes the advertised calibration/reuse workflow impossible on rerun.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        first_output = self.output
        second_output = self.root / "transient_rom_reuse"
        package = first_output / "rom_package"
        package.mkdir(parents=True)
        write_json(package / "rom_acceptance.json", {
            "accepted": True,
            "identity": {"power_trace": "sha256:power"},
        })
        write_json(package / "validation_report.json", {
            "accepted": True,
            "failure_reasons": [],
        })
        self.bind_package(package)
        persisted_validation = read_json(package / "validation_report.json")
        optimization = self.optimization()
        second_proposed = second_output / "optimization/proposed_layout.json"
        second_proposed.parent.mkdir(parents=True)
        write_json(second_proposed, {"mock": "proposed layout"})
        optimization["proposed_layout"] = str(second_proposed.resolve())
        prepared = self.prepared()
        second_power = second_output / "windows/mcpat/power_windows.json"
        second_power.parent.mkdir(parents=True)
        write_json(second_power, {"mock": "power windows"})
        prepared["power_windows"] = str(second_power.resolve())

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=prepared,
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=optimization,
        ) as optimize, patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ), patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, second_output, self.config_path,
                self.transient_r1, calibrate=False, execute_r2=False,
                rom_package_dir=package,
            )

        self.assertEqual(optimize.call_args.args[1], package.resolve())
        self.assertEqual(result["rom_package"], str(package.resolve()))
        self.assertEqual(result["rom_package_status"], "reused")
        self.assertEqual(result["training_hotspot_calls"], 8)
        self.assertEqual(result["holdout_hotspot_calls"], 2)
        self.assertEqual(result["calibration_hotspot_calls"], 10)
        self.assertEqual(result["training_hotspot_calls_this_invocation"], 0)
        self.assertEqual(result["holdout_hotspot_calls_this_invocation"], 0)
        self.assertEqual(result["calibration_hotspot_calls_this_invocation"], 0)
        self.assertEqual(result["rom_holdout_validation"], persisted_validation)

    def test_reuse_rejects_stale_or_incomplete_package_manifest_before_search(self):
        # Break caught: a valid-looking acceptance marker cannot bind a model
        # archive whose classified inventory or hashes have changed.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        acceptance = package / "rom_acceptance.json"
        write_json(acceptance, {
            "accepted": True,
            "identity": {"power_trace": "sha256:power"},
        })
        (package / "pod_model.npz").write_bytes(b"model-v1")
        self.bind_package(package)
        (package / "pod_model.npz").write_bytes(b"model-tampered")

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=self.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout"
        ) as optimize:
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                run_transient_rom_pipeline(
                    self.source_r1, self.steady, self.output, self.config_path,
                    self.transient_r1, calibrate=False, execute_r2=False,
                )

        optimize.assert_not_called()

    def test_rerun_r2_reuses_completed_windows_and_optimization(self):
        # Break caught: retrying a failed R2 must not regenerate periodic power
        # or collide with the already-populated deterministic optimizer output.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        package = self.output / "rom_package"
        package.mkdir(parents=True)
        write_json(package / "rom_acceptance.json", {
            "accepted": True,
            "identity": {"power_trace": "sha256:power"},
        })
        write_json(package / "validation_report.json", {
            "accepted": True,
            "failure_reasons": [],
        })
        self.bind_package(package)
        write_json(self.power_windows, {"mock": "cached power windows"})
        write_json(self.proposed_layout, {"mock": "proposed layout"})
        write_json(
            self.output / "optimization/optimization_report.json",
            self.optimization(),
        )

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": self.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline._reuse_prepared_power_windows",
            return_value=self.prepared(),
        ) as reuse_windows, patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows"
        ) as prepare, patch(
            "workflow.transient.rom.run_pipeline._reuse_optimization",
            return_value=self.optimization(),
        ) as reuse_optimization, patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout"
        ) as optimize, patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            return_value={"critical_l1d_to_l2_cycles": 7},
        ), patch(
            "workflow.transient.rom.run_pipeline.run_r2",
            return_value={"ipc2": 1.5},
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=self.final_validation_side_effect(),
        ):
            result = run_transient_rom_pipeline(
                self.source_r1, self.steady, self.output, self.config_path,
                self.transient_r1, calibrate=True, execute_r2=True,
                rerun_r2=True,
            )

        reuse_windows.assert_called_once()
        prepare.assert_not_called()
        reuse_optimization.assert_called_once()
        optimize.assert_not_called()
        self.assertTrue(run_r2_mock.call_args.kwargs["rerun"])
        self.assertEqual(result["bips2_trans"], 2.7)
        self.assertEqual(result["rom_package_status"], "reused-calibration")

    def test_rerun_r2_refuses_existing_final_hotspot_validation(self):
        # Break caught: retrying after final HotSpot has started would overwrite
        # or silently mix validation state whose completeness is unknown.
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        final_dir = self.output / "final_hotspot_validation"
        write_json(
            final_dir / "transient_sustainable_frequency.json",
            {"state": "success", "f_sus_trans_ghz": 1.8},
        )
        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", self.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1"
        ) as validate_r1, patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows"
        ) as prepare, patch(
            "workflow.transient.rom.run_pipeline.build_vector"
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2"
        ) as run_r2_mock, patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency"
        ) as final_search:
            with self.assertRaisesRegex(
                ValueError,
                "--rerun-r2 is only for retries that failed before final HotSpot",
            ):
                run_transient_rom_pipeline(
                    self.source_r1, self.steady, self.output, self.config_path,
                    self.transient_r1, calibrate=False, execute_r2=True,
                    rerun_r2=True,
                )

        validate_r1.assert_not_called()
        prepare.assert_not_called()
        build_vector_mock.assert_not_called()
        run_r2_mock.assert_not_called()
        final_search.assert_not_called()

    def test_exploratory_config_uses_exact_nonformal_rom_settings(self):
        # Break caught: a hidden threshold or formal label would make a run
        # incomparable to the reviewed Task 2 contract.
        config = read_json(
            Path(__file__).parents[1]
            / "configs/experiments/clip3d_transient_rom_exploratory.json"
        )
        settings = parse_settings(config)

        self.assertEqual(settings.sample_interval_ms, 2.0)
        self.assertEqual(settings.calibration_windows, 64)
        self.assertEqual(settings.prbs_seed, 20260807)
        self.assertEqual(settings.prbs_fraction, 0.20)
        self.assertEqual(settings.pod_energy_threshold, 0.999)
        self.assertEqual(settings.max_pod_rank, 16)
        self.assertEqual(settings.ridge, 1e-8)
        self.assertEqual(settings.max_condition_number, 1e10)
        self.assertEqual(settings.max_logm_condition_number, 1e12)
        self.assertEqual(settings.max_logm_error_estimate, 1e-8)
        self.assertEqual(settings.max_exp_log_reconstruction_error, 1e-8)
        self.assertEqual(settings.pss_period_repeats, 20)
        self.assertEqual(settings.pss_tolerance_c, 0.01)
        self.assertEqual(settings.frequency_tolerance_ghz, 0.01)
        self.assertEqual(settings.max_holdout_peak_error_c, 1.0)
        self.assertEqual(settings.max_holdout_grid_rmse_c, 0.75)
        self.assertEqual(settings.search_grid_points_per_axis, 25)
        self.assertEqual(settings.refinement_starts, 5)
        self.assertTrue(config["transient_rom"]["enabled"])
        self.assertTrue(config["transient_rom"]["non_formal"])
        self.assertFalse(config["transient_rom"]["paper_equivalent"])
        self.assertEqual(
            config["transient_rom"]["max_logm_condition_number"], 1e12
        )
        self.assertEqual(
            config["transient_rom"]["max_logm_error_estimate"], 1e-8
        )
        self.assertEqual(
            config["transient_rom"]["max_exp_log_reconstruction_error"], 1e-8
        )

if __name__ == "__main__":
    unittest.main()
