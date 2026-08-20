from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from workflow.cacti.characterize_cache import parse_cacti_output
from workflow.cache_contract import (
    build_cache_contract,
    mcpat_embedded_cache_record_identity,
    stable_identity,
)
from workflow.analysis.summarize_sweep import summarize
from workflow.analysis.prepare_raw_power_validation import prepare
from workflow.analysis.evaluate_operational_proxy import evaluate as evaluate_operational_proxy
from workflow.analysis.promote_validated_config import promote
from workflow.common import parse_gem5_stats, read_json, sha256_file, write_json
from workflow.floorplan.comparison_layouts import generate as generate_comparison_layouts
from workflow.floorplan.build_module_model import (
    apply_physical_areas,
    build_model,
    extract_communication_profile,
)
from workflow.floorplan.generate_hotspot_inputs import baseline_layout, grid_power, materialize
from workflow.floorplan.layout_metrics import (
    aggregate_wire_cycles,
    communication_weights_from_model,
    derive_layout_delays,
    select_rounded_wire_cycles,
)
from workflow.floorplan.discrete_partition import (
    search_discrete_partitions as shared_discrete_partition_search,
)
from workflow.floorplan.optimize_layout import (
    discrete_wire_score,
    optimize,
    proxy_temperature,
    proxy_temperature_components,
)
from workflow.mcpat.parse_mcpat import parse_mcpat_text, subtract
from workflow.r2.calibrate_lambda_wire import (
    calibrate as calibrate_lambda_wire,
    calibrate_series as calibrate_lambda_wire_series,
    main as calibrate_lambda_wire_main,
)
from workflow.r2.build_latency_vector import build_vector
from workflow.r2.run_wire_sensitivity import (
    build_sensitivity_vectors,
    run_series,
    summarize_workloads,
)
from workflow.run_lifting_pipeline import (
    evaluate_comparison_candidates, main as run_lifting_pipeline_main,
    optimize_clip3d_layout, run_pipeline, select_clip3d_candidate, validate_config,
)
from workflow.run_lifting_sweep import (
    completed as lifting_completed,
    required_artifacts as lifting_required_artifacts,
)
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot
from workflow.thermal.calibrate_proxy import (
    calibrate, candidate_layouts, parse_external_case, proxy_acceptance_checks,
    proxy_prediction, run_one, sample_split,
)
from workflow.thermal.diagnose_proxy_gradient import (
    ProxyVariant, exclusive_output_directory, parse_variant, prepare_output_directory,
    summarize_variant,
)
from workflow.thermal.summarize_proxy_gradient_series import summarize_reports
from workflow.thermal.sustainable_frequency import closed_form_frequency
from workflow.thermal.run_anchor_validation import run_manifest
from workflow.thermal.validate_frequency import (
    compose_separated_ptrace,
    read_ptrace,
    two_point_affine_frequency,
    validate_case,
)


def metric_lines(area, dynamic, sub, gate, indent="  "):
    return (f"{indent}Area = {area} mm^2\n{indent}Runtime Dynamic = {dynamic} W\n"
            f"{indent}Subthreshold Leakage = {sub} W\n{indent}Gate Leakage = {gate} W\n")


class WorkflowTests(unittest.TestCase):
    def test_proxy_gradient_series_uses_weighted_sign_counts_and_keeps_labels(self):
        def diagnostic(sign_comparable, sign_agreement, spearman, regret, active,
                       observability=None, hotspot_observability=None):
            evaluation = {
                "candidate_count": 9,
                "sign_comparable_count": sign_comparable,
                "sign_agreement_count": sign_agreement,
                "sign_agreement_rate": sign_agreement / sign_comparable,
                "spearman": spearman,
                "proxy_selected": {"selection_regret_c": regret},
                "thermal_frequency_term_active": active,
            }
            if hotspot_observability is not None:
                evaluation["hotspot_frequency_observability"] = hotspot_observability
            if observability:
                evaluation.update(observability)
            return {
                "method": "common-HotSpot-grid equation-(14) gradient diagnostic",
                "model": "fixture-model", "config": "fixture-config",
                "hotspot": "fixture-hotspot", "grid_points_per_axis": 3,
                "allowed_l2_tiers": [1],
                "variants": [{
                    "name": "fitted-area", "proxy_spatial_model": "area-quadrature",
                    "lc_die_side_ratio": 0.0586007,
                }],
                "evaluations": {"fitted-area": evaluation},
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.json", root / "second.json"
            write_json(first, diagnostic(2, 2, 1.0, 0.0, False))
            write_json(second, diagnostic(6, 3, 0.5, 0.4, True, {
                "raw_thermal_frequency_term_active": True,
                "hotspot_frequency_varies": True,
                "anchored_proxy_frequency_varies": False,
                "raw_proxy_frequency_varies": False,
            }, {
                "safe_temperature_c": 95.0,
                "hotspot_tmax_range_c": [90.0, 96.0],
                "hottest_headroom_to_safe_c": -1.0,
                "sampled_positions_cross_safe_threshold": True,
            }))
            result = summarize_reports([("l2-holdout", first), ("in-fit", second)])

        values = result["variants"]["fitted-area"]
        self.assertEqual(values["weighted_sign_comparable_count"], 8)
        self.assertEqual(values["weighted_sign_agreement_count"], 5)
        self.assertEqual(values["weighted_sign_agreement_rate"], 0.625)
        self.assertEqual(values["selection_regret_c"]["mean"], 0.2)
        self.assertEqual(values["thermal_frequency_term_active_report_count"], 1)
        observability = values["frequency_observability"]
        self.assertEqual(observability["raw_thermal_frequency_term_active"], {
            "observed_report_count": 1, "true_report_count": 1,
        })
        self.assertEqual(observability["hotspot_frequency_varies"], {
            "observed_report_count": 1, "true_report_count": 1,
        })
        self.assertEqual(observability["anchored_proxy_frequency_varies"], {
            "observed_report_count": 1, "true_report_count": 0,
        })
        self.assertEqual(values["hotspot_frequency_observability"], {
            "observed_report_count": 1,
            "sampled_positions_cross_safe_threshold_report_count": 1,
            "hottest_headroom_to_safe_c": {
                "count": 1, "mean": -1.0, "median": -1.0,
                "min": -1.0, "max": -1.0,
            },
        })
        self.assertIsNone(values["per_report"][0]["hotspot_frequency_observability"])
        self.assertEqual(
            values["per_report"][1]["hotspot_frequency_observability"]
            ["hottest_headroom_to_safe_c"], -1.0,
        )
        self.assertEqual(values["per_report"][0]["label"], "l2-holdout")
        self.assertEqual(result["common_contract"]["config"], "fixture-config")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, incompatible = root / "first.json", root / "incompatible.json"
            write_json(first, diagnostic(2, 2, 1.0, 0.0, False))
            wrong_contract = diagnostic(2, 2, 1.0, 0.0, False)
            wrong_contract["config"] = "other-config"
            write_json(incompatible, wrong_contract)
            with self.assertRaisesRegex(ValueError, "same physical/proxy contract"):
                summarize_reports([("first", first), ("second", incompatible)])

    def test_proxy_gradient_resume_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "gradient"
            prepare_output_directory(output_dir, resume=False)
            (output_dir / "partial-artifact").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--resume"):
                prepare_output_directory(output_dir, resume=False)
            prepare_output_directory(output_dir, resume=True)

    def test_proxy_gradient_exclusively_locks_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "gradient"
            with exclusive_output_directory(output_dir):
                # The lock file itself must not make a brand-new output look
                # like a resumable partial run.
                prepare_output_directory(output_dir, resume=False)
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    with exclusive_output_directory(output_dir):
                        pass
            with exclusive_output_directory(output_dir):
                prepare_output_directory(output_dir, resume=False)

    def test_proxy_gradient_variant_parser_and_summary_preserve_directional_metrics(self):
        variant = parse_variant("paper-area=area-quadrature,0.5")
        self.assertEqual(variant, ProxyVariant("paper-area", "area-quadrature", 0.5))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_variant("bad=center,0")
        anchor = {
            "tmax_c": 100.0,
            "variants": {
                "paper-area": {
                    "raw_proxy_tmax_c": 80.0,
                    "raw_frequency": {
                        "state": "thermal_headroom", "sustainable_frequency_ghz": 2.0,
                    },
                    "anchored_frequency": {
                        "state": "thermally_limited", "sustainable_frequency_ghz": 1.9,
                    },
                }
            },
        }
        records = [
            {"row": 0, "column": 0, "tmax_c": 101.0,
             "hotspot_frequency": {
                 "state": "thermally_limited", "sustainable_frequency_ghz": 1.9,
             },
             "variants": {"paper-area": {
                 "raw_proxy_tmax_c": 81.0,
                 "raw_frequency": {
                     "state": "thermal_headroom", "sustainable_frequency_ghz": 2.0,
                 },
                 "anchored_frequency": {
                     "state": "thermally_limited", "sustainable_frequency_ghz": 1.9,
                 },
             }}},
            {"row": 0, "column": 1, "tmax_c": 99.0,
             "hotspot_frequency": {
                 "state": "thermal_headroom", "sustainable_frequency_ghz": 2.0,
             },
             "variants": {"paper-area": {
                 "raw_proxy_tmax_c": 79.0,
                 "raw_frequency": {
                     "state": "thermal_headroom", "sustainable_frequency_ghz": 2.0,
                 },
                 "anchored_frequency": {
                     "state": "thermally_limited", "sustainable_frequency_ghz": 1.9,
                 },
             }}},
            {"row": 1, "column": 1, "tmax_c": 100.5,
             "hotspot_frequency": {
                 "state": "thermally_limited", "sustainable_frequency_ghz": 1.95,
             },
             "variants": {"paper-area": {
                 "raw_proxy_tmax_c": 80.5,
                 "raw_frequency": {
                     "state": "thermal_headroom", "sustainable_frequency_ghz": 2.0,
                 },
                 "anchored_frequency": {
                     "state": "thermally_limited", "sustainable_frequency_ghz": 1.9,
                 },
             }}},
        ]
        summary = summarize_variant("paper-area", records, anchor, 0.02, 95.0)
        self.assertEqual(summary["sign_agreement_rate"], 1.0)
        self.assertEqual(summary["spearman"], 1.0)
        self.assertEqual(summary["proxy_selected"]["selection_regret_c"], 0.0)
        self.assertTrue(summary["thermal_frequency_term_active"])
        self.assertEqual(summary["hotspot_frequency_states"], {
            "thermally_limited": 2, "thermal_headroom": 1,
        })
        self.assertEqual(summary["anchored_proxy_frequency_states"], {
            "thermally_limited": 3,
        })
        self.assertEqual(summary["raw_proxy_frequency_states"], {
            "thermal_headroom": 3,
        })
        self.assertEqual(summary["frequency_state_agreement_rate"], 2 / 3)
        self.assertEqual(summary["raw_frequency_state_agreement_rate"], 1 / 3)
        self.assertFalse(summary["raw_thermal_frequency_term_active"])
        self.assertEqual(summary["hotspot_sustainable_frequency_range_ghz"], [1.9, 2.0])
        self.assertFalse(summary["anchored_proxy_frequency_varies"])
        self.assertFalse(summary["raw_proxy_frequency_varies"])
        self.assertTrue(summary["hotspot_frequency_varies"])
        self.assertEqual(summary["hotspot_frequency_observability"], {
            "safe_temperature_c": 95.0,
            "hotspot_tmax_range_c": [99.0, 101.0],
            "hottest_headroom_to_safe_c": -6.0,
            "sampled_positions_cross_safe_threshold": False,
        })

    def test_pipeline_forwards_hotspot_materialization_contract(self):
        """Fixed-bin and paper-single runs must preserve identification inputs."""
        # Break caught: omitting or changing a physical materialization control
        # in either pipeline branch silently reverts its HotSpot inputs to the
        # historical grid-cell/default-trace contract.
        materialization_controls = {
            "input_granularity": "module",
            "compact_trace": True,
            "ptrace_precision": 9,
        }
        config = {
            "schema_version": 1,
            "technology_nm": 45,
            "frequency": {
                "ambient_c": 25.0, "f0_ghz": 2.0, "fmin_ghz": 0.4,
                "tsafe_c": 95.0,
            },
            "physical": {
                "grid_size": 64, "tiers": 2, "utilization": 0.70,
                "r_convec_k_per_w": 5.0, **materialization_controls,
            },
            "layout_optimizer": {
                "alpha": 0.3, "beta": 0.0, "cross_tier_weight": 0.65,
                "lambda_wire": 0.01, "r_convec_k_per_w": 5.0,
                "validation_policy": "paper-single",
            },
            "delay": {},
            "mcpat": {},
        }
        model = {
            "modules": [{"name": "bottom", "tier": 0, "total_power_w": 1.0},
                        {"name": "top", "tier": 1, "total_power_w": 1.0}],
            "totals": {"total_power_w": 2.0}, "gamma": 0.2,
            "power_provenance": {}, "area_provenance": {},
            "cache_authority": "McPAT 1.3 embedded CACTI-P",
            "mcpat_provenance": {
                "schema_version": 1,
                "authority": "CLIP strict patched McPAT 1.3 runner",
                "hashes": {},
            },
            "power_distribution": {
                "movable_kinds": [], "movable_power_w": 0.0,
                "movable_power_fraction": 0.0,
            },
        }
        vector = {
            "wire_cycle_aggregation_for_r2": "mean",
            "critical_l1d_to_l2_cycles": 1,
            "layout_delays": {
                "wire_cycles_unrounded": 1.0,
                "maximum_wire_cycles_unrounded": 1.0,
                "wire_cycles": 1, "maximum_wire_cycles": 1,
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1_dir = root / "r1"
            r1_dir.mkdir()
            write_json(r1_dir / "r1_metadata.json", {
                "workload": "fixture", "l1d_size": "32kB", "l2_size": "512kB",
            })
            (r1_dir / "stats.txt").write_text("", encoding="utf-8")
            config_path = root / "config.json"
            write_json(config_path, config)

            def build_module_case(*args, **_kwargs):
                write_json(args[2], model)
                return model

            def run_mcpat_case(_r1, output_dir, _settings, executable):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "input.xml").write_text("<component/>", encoding="utf-8")
                (output_dir / "mcpat.out").write_text("native\n", encoding="utf-8")
                artifact = {
                    "provenance": {
                        "schema_version": 1,
                        "authority": "CLIP strict patched McPAT 1.3 runner",
                        "hashes": {
                            "output_sha256": sha256_file(output_dir / "mcpat.out"),
                            "binary_sha256": sha256_file(executable),
                        },
                    },
                }
                model["mcpat_provenance"] = artifact["provenance"]
                write_json(output_dir / "mcpat.json", artifact)
                return artifact

            def materialize_case(_modules, hotspot_dir, *_args, **_kwargs):
                hotspot_dir.mkdir(parents=True, exist_ok=True)
                write_json(hotspot_dir / "layout.json", {
                    "modules": model["modules"],
                })

            def optimize_case(_modules, _layout, report_path, _config):
                write_json(report_path, {
                    "selected": {"proxy_tmax_c": 80.0},
                    "candidates": [{"tier": 1, "collision_mm2": 0.0}],
                })

            with patch(
                "workflow.run_lifting_pipeline.materialize",
                side_effect=materialize_case,
            ) as materialize_mock, patch(
                "workflow.run_lifting_pipeline.run_mcpat",
                side_effect=run_mcpat_case,
            ), patch(
                "workflow.run_lifting_pipeline.build_model", side_effect=build_module_case,
            ), patch(
                "workflow.run_lifting_pipeline.optimize_clip3d_layout",
                side_effect=optimize_case,
            ), patch(
                "workflow.run_lifting_pipeline.run_hotspot", return_value={"tmax_c": 80.0},
            ), patch(
                "workflow.run_lifting_pipeline.evaluate",
                return_value={"sustainable_frequency_ghz": 1.0, "ipc1": 1.0,
                              "bips1_thermal": 1.0},
            ), patch(
                "workflow.run_lifting_pipeline.build_vector", return_value=vector,
            ):
                run_pipeline(r1_dir, root / "fixed", config_path, "fixed-bin")
                run_pipeline(r1_dir, root / "paper-single", config_path, "clip3d")

            self.assertEqual(materialize_mock.call_count, 2)
            for call in materialize_mock.call_args_list:
                self.assertEqual(call.kwargs, materialization_controls)

    def test_lifting_cli_accepts_discrete_partition_override(self):
        process = subprocess.run(
            [
                sys.executable, "-m", "workflow.run_lifting_pipeline",
                "--r1-dir", "/definitely/missing/r1",
                "--output-dir", "/definitely/missing/output",
                "--wire-objective", "discrete-partition",
            ],
            cwd=Path(__file__).resolve().parents[1], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn("invalid choice", process.stderr)
        self.assertIn("FileNotFoundError", process.stderr)

    def test_floorplanner_cli_accepts_discrete_partition_mode(self):
        process = subprocess.run(
            [
                sys.executable, "-m", "workflow.floorplan.optimize_layout",
                "--modules", "/definitely/missing/modules.json",
                "--output-layout", "/definitely/missing/layout.json",
                "--report", "/definitely/missing/report.json",
                "--wire-objective", "discrete-partition",
            ],
            cwd=Path(__file__).resolve().parents[1], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn("invalid choice", process.stderr)
        self.assertIn("FileNotFoundError", process.stderr)

    def test_temperature_text_and_csv_fields_use_six_decimals(self):
        from workflow.common import format_temperature_c, format_temperature_csv_row

        self.assertEqual(format_temperature_c(116.25), "116.250000")
        self.assertEqual(
            format_temperature_csv_row({"tmax_c": 116.25, "ipc1": 3.4}),
            {"tmax_c": "116.250000", "ipc1": 3.4},
        )

    def test_power_trace_round_trips_values_that_12g_breaks_total_invariant(self):
        """Raw total/dynamic/leakage files must retain their cell-level sum."""
        from workflow.floorplan.generate_hotspot_inputs import write_ptrace
        from workflow.thermal.validate_frequency import validate_total_trace

        dynamic, leakage = 0.00652379565084, 0.00145495463892
        # A fractional grid allocation can make these computed values larger
        # than twelve significant digits can preserve within 1e-9 W.
        dynamic = dynamic * 1e8 / 5.0
        leakage = leakage * 1e8 / 5.0
        tiers = [{"cells": [{
            "name": "cell",
            "dynamic_power_w": dynamic,
            "leakage_power_w": leakage,
            "total_power_w": dynamic + leakage,
        }]}]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dynamic_path = root / "dynamic.ptrace"
            leakage_path = root / "leakage.ptrace"
            total_path = root / "total.ptrace"
            write_ptrace(dynamic_path, tiers, "dynamic_power_w")
            write_ptrace(leakage_path, tiers, "leakage_power_w")
            write_ptrace(total_path, tiers, "total_power_w")

            names, dynamic_values = read_ptrace(dynamic_path)
            _, leakage_values = read_ptrace(leakage_path)
            validate_total_trace(names, dynamic_values, leakage_values, total_path)

    def test_hotspot_sources_are_not_tracked_by_this_repository(self):
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "tools/src/hotspot/hotspot.c"],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(completed.returncode, 0)

    def test_proxy_calibration_cli_prints_temperature_rmse_with_six_decimals(self):
        """A shortened RMSE presentation would lose calibrated temperature precision."""
        from workflow.thermal.calibrate_proxy import main as calibrate_proxy_main

        report = {
            "fit": {"parameters": {
                "alpha": 0.3, "beta": 0.0, "cross_tier_weight": 0.7,
            }},
            "evaluations": {"cross_validated_training_fit": {"validation": {
                "rmse_c": 12.3456789, "spatial_spearman": 0.8,
            }}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "modules.json"
            model.write_text("{}", encoding="utf-8")
            output = StringIO()
            argv = [
                "calibrate_proxy", "--model", f"fixture={model}",
                "--output-dir", str(root / "output"),
            ]
            with patch("sys.argv", argv), \
                 patch("workflow.thermal.calibrate_proxy.calibrate", return_value=report), \
                 redirect_stdout(output):
                calibrate_proxy_main()

        self.assertIn("validation RMSE=12.345679 C", output.getvalue())

    def test_rejected_alpha_lc_diagnostic_config_is_exact(self):
        """The diagnostic binds rejected alpha/Lc evidence to both layout paths."""
        root = Path(__file__).resolve().parents[1]
        manifest = read_json(root / "manifests/parameter_provenance/unscaled_alpha_lc_rejected_20260814.json")
        config = read_json(root / "configs/experiments/clip3d_unscaled_alpha_lc_rejected_diagnostic.json")

        self.assertEqual(manifest["fit_report"]["sha256"], "918f93b775d5131c98b5c8300dae468b8bc2a87e29f66cd109c53bcd25795c89")
        self.assertEqual(manifest["fit"]["parameters"], {
            "alpha": 3.348558894986864,
            "lc_die_side_ratio": 0.3525311644629249,
            "cross_tier_weight": 0.8378797681280803,
            "beta": 0.0,
        })
        self.assertFalse(manifest["accepted_for_formal_or_shared_use"])
        physical_identity = manifest["physical_identity"]
        self.assertEqual(physical_identity["technology_nm"], 45)
        self.assertEqual(physical_identity["tiers"], 2)
        self.assertEqual(physical_identity["grid_size"], 64)
        self.assertEqual(physical_identity["input_granularity"], "module")
        self.assertTrue(physical_identity["compact_trace"])
        self.assertEqual(physical_identity["ptrace_precision"], 9)
        self.assertEqual(physical_identity["utilization"], 0.7)
        self.assertEqual(physical_identity["ambient_c"], 25.0)
        self.assertEqual(physical_identity["r_convec_k_per_w"], 1.042)
        self.assertEqual(physical_identity["global_scaling"], "none")
        self.assertEqual(physical_identity["thermal_stack"], {
            "silicon_resistivity_mk_per_w": 0.01,
            "tim_resistivity_mk_per_w": 0.25,
            "interposer_thickness_m": 0.0001,
            "active_silicon_thickness_m": 0.00005,
            "tim_thickness_m": 0.00002,
            "local_resistance_scale": 1.0,
            "silicon_resistance_scale": 1.0,
            "tim_resistance_scale": 1.0,
        })
        self.assertEqual(config["experiment_classification"], {
            "mode": "rejected-alpha-lc-diagnostic", "non_formal": True,
            "paper_equivalent": False, "shared_parameter_accepted": False,
        })
        physical = config["physical"]
        self.assertEqual(physical["grid_size"], 64)
        self.assertEqual(physical["input_granularity"], "module")
        self.assertTrue(physical["compact_trace"])
        self.assertEqual(physical["ptrace_precision"], 9)
        self.assertEqual(physical["r_convec_k_per_w"], 1.042)
        self.assertEqual(physical["thermal_stack"]["local_resistance_scale"], 1.0)
        self.assertEqual(physical["thermal_stack"]["silicon_resistance_scale"], 1.0)
        self.assertEqual(physical["thermal_stack"]["tim_resistance_scale"], 1.0)
        optimizer = config["layout_optimizer"]
        self.assertEqual(optimizer["alpha"], 3.348558894986864)
        self.assertEqual(optimizer["lc_die_side_ratio"], 0.3525311644629249)
        self.assertEqual(optimizer["cross_tier_weight"], 0.8378797681280803)
        self.assertEqual(optimizer["beta"], 0.0)
        self.assertEqual(optimizer["allowed_l2_tiers"], [1])
        self.assertEqual(optimizer["validation_policy"], "paper-single")
        self.assertEqual(optimizer["proxy_spatial_model"], "area-quadrature")
        self.assertEqual(optimizer["proxy_quadrature_order"], 2)
        self.assertEqual(optimizer["wire_objective"], "discrete-partition")
        self.assertEqual(optimizer["lambda_wire"], 0.0020119160767721133)
        lambda_provenance = optimizer["parameter_provenance"]["lambda_wire"]
        self.assertEqual(lambda_provenance["source"], "manifests/parameter_provenance/lambda_wire_fft_rejected.json")
        self.assertEqual(lambda_provenance["field"], "lambda_wire")
        self.assertEqual(lambda_provenance["value"], optimizer["lambda_wire"])
        self.assertFalse(lambda_provenance["accepted_for_formal_or_shared_use"])
        self.assertFalse(config["formal_validation"]["accepted"])
        validate_config(config, "fixed-bin")
        validate_config(config, "clip3d")


class FrequencyTests(unittest.TestCase):
    def test_separated_frequency_trace_scales_only_dynamic_power(self):
        """Changing frequency must not scale per-cell leakage power."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dynamic.ptrace").write_text("a b\n8 4\n", encoding="utf-8")
            (root / "leakage.ptrace").write_text("a b\n2 6\n", encoding="utf-8")

            result = compose_separated_ptrace(
                root / "dynamic.ptrace", root / "leakage.ptrace", root / "one.ptrace",
                frequency_ghz=1.0, f0_ghz=2.0,
            )

            self.assertEqual(read_ptrace(root / "one.ptrace")[1], [6.0, 8.0])
            self.assertEqual(result["dynamic_scale"], 0.5)

    def test_separated_frequency_trace_rejects_misaligned_power_traces(self):
        """A reordered HotSpot power trace must not silently misplace power."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dynamic.ptrace").write_text("a b\n8 4\n", encoding="utf-8")
            (root / "leakage.ptrace").write_text("b a\n2 6\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "headers"):
                compose_separated_ptrace(
                    root / "dynamic.ptrace", root / "leakage.ptrace", root / "one.ptrace",
                    frequency_ghz=1.0, f0_ghz=2.0,
                )

    def test_two_point_affine_frequency_uses_cellwise_not_global_gamma_limit(self):
        """Equation (9) must be limited by the hottest affine cell, not a scalar average."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nominal = root / "nominal.grid.steady.txt"
            reference = root / "reference.grid.steady.txt"
            # At f0=2 GHz, cell 0 is 100 C and cell 1 is 90 C.  At 1 GHz
            # they are respectively 60 C and 80 C, so their affine maps are
            # 20 + 80*f/f0 and 70 + 20*f/f0 C.  Cell 0 limits 95 C at 1.875.
            nominal.write_text(
                "Layer 1:\n0\t373.15\n1\t363.15\n", encoding="utf-8",
            )
            reference.write_text(
                "Layer 1:\n0\t333.15\n1\t353.15\n", encoding="utf-8",
            )
            result = two_point_affine_frequency(
                {"grid_steady_file": str(nominal)},
                {"grid_steady_file": str(reference)},
                {"ambient_c": 25.0, "active_power_layers": [1]}, 1.0,
                {"f0_ghz": 2.0, "fmin_ghz": 0.4, "tsafe_c": 95.0},
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["equation"], 9)
            self.assertEqual(result["limiting_unit"], "layer_1_g0")
            self.assertEqual(result["frequency_state"], "thermally_limited")
            self.assertAlmostEqual(result["sustainable_frequency_ghz"], 1.875)
            self.assertAlmostEqual(
                result["predicted_tmax_at_sustainable_frequency_c"], 95.0,
            )

    def test_frequency_validation_runs_two_point_fallback_after_gamma_rejection(self):
        """A rejected scalar gamma must trigger, but not masquerade as, Eq.(9)."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            nominal = case / "grid.steady.txt"
            reference = case / "one.grid.steady.txt"
            nominal.write_text(
                "Layer 1:\n0\t373.15\n1\t363.15\n", encoding="utf-8",
            )
            reference.write_text(
                "Layer 1:\n0\t333.15\n1\t353.15\n", encoding="utf-8",
            )
            write_json(root / "modules.json", {"gamma": 0.2})
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
                "active_power_layers": [1],
            })
            write_json(case / "thermal_result.json", {
                "tmax_c": 100.0, "grid_steady_file": str(nominal),
            })
            (case / "power_dynamic.ptrace").write_text("a\n8\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a\n2\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a\n10\n", encoding="utf-8")

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       side_effect=[
                           {"tmax_c": 80.0, "grid_steady_file": str(reference)},
                           # Even an exact scalar f_sus safety point cannot
                           # repair the already-observed nonuniform power
                           # field at the reference frequency.
                           {"tmax_c": 95.0},
                           {"tmax_c": 95.005},
                       ]) as runner:
                result = validate_case(
                    case, root / "modules.json", root / "validation.json", [1.0],
                )

            self.assertFalse(result["recommendation"]["accepted"])
            self.assertFalse(result["uniform_gamma_acceptable"])
            self.assertTrue(result["solution_validation"]["accepted"])
            fallback = result["two_point_affine_frequency"]
            self.assertTrue(fallback["available"])
            self.assertAlmostEqual(fallback["sustainable_frequency_ghz"], 1.875)
            self.assertTrue(fallback["recommendation"]["accepted"])
            self.assertAlmostEqual(
                fallback["solution_validation"]["safe_error_c"], 0.005,
            )
            self.assertEqual(runner.call_count, 3)

    def test_frequency_validation_defaults_to_separated_hotspot_trace(self):
        """Formal frequency validation writes and reports the per-cell raw-power trace."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            write_json(root / "modules.json", {
                "gamma": 0.2,
                "modules": [
                    {"dynamic_power_w": 1.0, "leakage_power_w": 3.0},
                    {"dynamic_power_w": 4.0, "leakage_power_w": 0.0},
                ],
            })
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
            })
            write_json(case / "thermal_result.json", {"tmax_c": 100.0})
            (case / "power_dynamic.ptrace").write_text("a b\n8 4\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a b\n2 6\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a b\n10 10\n", encoding="utf-8")

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       return_value={"tmax_c": 80.0}):
                result = validate_case(
                    case, root / "modules.json", root / "validation.json", [1.0],
                    validate_solution=False,
                )

            run = result["frequencies"][0]
            self.assertEqual(result["scaling_mode"], "separated-dynamic-leakage")
            self.assertEqual(run["hotspot_tmax_c"], 80.0)
            self.assertEqual(read_ptrace(Path(run["power_trace"]))[1], [6.0, 8.0])
            self.assertEqual(run["trace_sums_w"]["dynamic"], 12.0)
            self.assertEqual(run["trace_sums_w"]["leakage"], 8.0)
            self.assertEqual(run["trace_sums_w"]["composed"], 14.0)
            self.assertEqual(run["trace_sums_w"]["total_at_f0"], 20.0)
            self.assertEqual(
                run["uniform_gamma_comparison"]["scaling_mode"], "paper-uniform-gamma"
            )
            self.assertIn("max_abs_uniform_gamma_comparison_error_c", result)
            self.assertNotIn("max_abs_linear_error_c", result)
            self.assertFalse(result["recommendation"]["accepted"])
            self.assertEqual(result["frequency_settings"]["max_safe_error_c"], 0.02)
            self.assertEqual(
                result["module_gamma_observability"]["module_gamma_range"],
                [0.0, 0.75],
            )
            self.assertEqual(
                result["module_gamma_observability"]["power_weighted_gamma"],
                0.375,
            )

    def test_frequency_validation_requires_paper_scale_safety_error(self):
        """A degree-scale DTM miss must not endorse the uniform-gamma shortcut."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            write_json(root / "modules.json", {"gamma": 0.2})
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
            })
            write_json(case / "thermal_result.json", {"tmax_c": 100.0})
            (case / "power_dynamic.ptrace").write_text("a\n8\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a\n2\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a\n10\n", encoding="utf-8")

            # The 1 GHz reference must also satisfy the scalar spatial-power
            # assumption; this test isolates the separate f_sus safety gate.
            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       side_effect=[{"tmax_c": 70.0}, {"tmax_c": 95.03}]):
                strict = validate_case(
                    case, root / "modules.json", root / "strict.json", [1.0],
                )
            self.assertAlmostEqual(
                strict["solution_validation"]["safe_error_c"], 0.03,
            )
            self.assertEqual(
                strict["solution_validation"]["max_safe_error_c"], 0.02,
            )
            self.assertFalse(strict["recommendation"]["accepted"])

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       side_effect=[{"tmax_c": 70.0}, {"tmax_c": 95.03}]):
                relaxed = validate_case(
                    case, root / "modules.json", root / "relaxed.json", [1.0],
                    frequency_settings={"max_safe_error_c": 0.05},
                )
            self.assertTrue(relaxed["recommendation"]["accepted"])
            self.assertEqual(
                relaxed["solution_validation"]["max_safe_error_c"], 0.05,
            )

    def test_frequency_validation_forwards_explicit_hotspot_binary(self):
        """A worktree may validate a real case with a shared built HotSpot."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            write_json(root / "modules.json", {"gamma": 0.2})
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
            })
            write_json(case / "thermal_result.json", {"tmax_c": 80.0})
            (case / "power_dynamic.ptrace").write_text("a\n8\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a\n2\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a\n10\n", encoding="utf-8")
            hotspot = root / "shared-hotspot"

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       return_value={"tmax_c": 60.0}) as runner:
                validate_case(
                    case, root / "modules.json", root / "validation.json", [1.0],
                    validate_solution=False, hotspot=hotspot,
                )

            self.assertEqual(runner.call_args.kwargs["hotspot"], hotspot)

    def test_below_f0_hotspot_failure_writes_a_rejected_validation_result(self):
        """A failed mandatory safety solve must leave an auditable rejection on disk."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            output = root / "validation.json"
            write_json(root / "modules.json", {"gamma": 0.2})
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
            })
            write_json(case / "thermal_result.json", {"tmax_c": 100.0})
            (case / "power_dynamic.ptrace").write_text("a\n8\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a\n2\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a\n10\n", encoding="utf-8")

            def fail_only_for_fsus(_case_dir, ptrace_name, result_name):
                if "_fsus" in ptrace_name:
                    raise RuntimeError("injected f_sus HotSpot failure")
                return {"tmax_c": 80.0}

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       side_effect=fail_only_for_fsus):
                result = validate_case(
                    case, root / "modules.json", output, [1.0],
                    validate_solution=True,
                )

            recorded = read_json(output)
            self.assertEqual(recorded, result)
            self.assertFalse(result["recommendation"]["accepted"])
            self.assertFalse(result["solution_validation"]["accepted"])
            self.assertIn("injected f_sus HotSpot failure",
                          result["solution_validation"]["error"])

    def test_anchor_summary_reports_actual_hotspot_run_counts_and_acceptance(self):
        """Anchor summaries must distinguish requested runs from mandatory safety solves."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "frequencies_ghz": [0.5, 1.0],
                "cases": [
                    {"label": "first", "case_dir": str(root / "first")},
                    {"label": "second", "case_dir": str(root / "second")},
                ],
            }
            write_json(root / "anchors.json", manifest)
            results = [
                {
                    "frequencies": [{}, {}],
                    "max_abs_uniform_gamma_comparison_error_c": 0.25,
                    "frequency_settings": {"max_safe_error_c": 0.02},
                    "solution_validation": {"accepted": True, "safe_error_c": 0.5},
                    "recommendation": {"accepted": True},
                },
                {
                    "frequencies": [{}],
                    "max_abs_uniform_gamma_comparison_error_c": 0.75,
                    "frequency_settings": {"max_safe_error_c": 0.02},
                    "solution_validation": None,
                    "recommendation": {"accepted": False},
                },
            ]

            with patch("workflow.thermal.run_anchor_validation.validate_case",
                       side_effect=results):
                summary = run_manifest(root / "anchors.json", root / "summary.json")

            self.assertEqual(read_json(root / "summary.json"), summary)
            self.assertEqual(summary["requested_frequency_hotspot_run_count"], 3)
            self.assertEqual(summary["fsus_safety_solve_count"], 1)
            self.assertEqual(summary["two_point_safety_solve_count"], 0)
            self.assertEqual(summary["hotspot_run_count"], 4)
            self.assertEqual(summary["max_abs_uniform_gamma_comparison_error_c"], 0.75)
            self.assertEqual(summary["safe_error_limit_c"], 0.02)
            self.assertFalse(summary["recommendation"]["accepted"])
            self.assertFalse(summary["two_point_fallback_recommendation"]["accepted"])

            mixed_limits = [dict(item) for item in results]
            mixed_limits[1] = dict(mixed_limits[1])
            mixed_limits[1]["frequency_settings"] = {"max_safe_error_c": 0.5}
            with patch("workflow.thermal.run_anchor_validation.validate_case",
                       side_effect=mixed_limits):
                with self.assertRaisesRegex(ValueError, "same max_safe_error_c"):
                    run_manifest(root / "anchors.json", root / "mixed.json")

    def test_anchor_summary_reports_validated_two_point_fallback_separately(self):
        """A rejected scalar gamma may be replaced only by an audited Eq.(9) result."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "anchors.json", {
                "cases": [{"label": "fft", "case_dir": str(root / "fft")}],
            })
            scalar_rejected_two_point_accepted = {
                "frequencies": [{}],
                "max_abs_uniform_gamma_comparison_error_c": 0.32,
                "frequency_settings": {"max_safe_error_c": 0.02},
                "solution_validation": {"accepted": True, "safe_error_c": 0.0},
                "recommendation": {"accepted": False},
                "two_point_affine_frequency": {
                    "available": True,
                    "solution_validation": {"accepted": True, "safe_error_c": 0.0},
                    "recommendation": {"accepted": True},
                },
            }
            with patch("workflow.thermal.run_anchor_validation.validate_case",
                       return_value=scalar_rejected_two_point_accepted):
                summary = run_manifest(root / "anchors.json", root / "summary.json")

            self.assertFalse(summary["recommendation"]["accepted"])
            self.assertEqual(summary["two_point_safety_solve_count"], 1)
            self.assertEqual(summary["hotspot_run_count"], 3)
            self.assertEqual(
                summary["two_point_fallback_recommendation"],
                {"available_case_count": 1, "accepted_case_count": 1, "accepted": True},
            )

    def test_anchor_manifest_forwards_frequency_settings(self):
        """Manifest f0 must control the dynamic scale used in every anchor case."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            write_json(root / "modules.json", {"gamma": 0.2})
            write_json(case / "hotspot_manifest.json", {
                "ambient_c": 25.0, "r_convec_k_per_w": 5.0,
            })
            write_json(case / "thermal_result.json", {"tmax_c": 80.0})
            (case / "power_dynamic.ptrace").write_text("a\n8\n", encoding="utf-8")
            (case / "power_leakage.ptrace").write_text("a\n2\n", encoding="utf-8")
            (case / "power.ptrace").write_text("a\n10\n", encoding="utf-8")
            manifest = {
                "frequency_settings": {
                    "f0_ghz": 1.0, "fmin_ghz": 0.4, "tsafe_c": 95.0,
                    "scaling_mode": "separated-dynamic-leakage",
                },
                "frequencies_ghz": [0.5], "validate_solution": False,
                "cases": [{"label": "one", "case_dir": str(case),
                           "modules": str(root / "modules.json")}],
            }
            write_json(root / "anchors.json", manifest)

            with patch("workflow.thermal.validate_frequency.run_hotspot",
                       return_value={"tmax_c": 60.0}):
                result = run_manifest(root / "anchors.json", root / "summary.json")

            run = result["cases"][0]["result"]["frequencies"][0]
            self.assertEqual(run["dynamic_scale"], 0.5)
            self.assertEqual(read_ptrace(Path(run["power_trace"]))[1], [6.0])

    def test_paper_anchor(self):
        frequency, state, raw = closed_form_frequency(100.0, 0.446)
        self.assertEqual(state, "thermally_limited")
        self.assertAlmostEqual(frequency, 1.759326, places=5)
        self.assertAlmostEqual(raw, frequency)

    def test_headroom(self):
        self.assertEqual(closed_form_frequency(90.0, 0.5)[0], 2.0)

    def test_lambda_wire_matched_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "base_result.json", {"ipc2": 3.0})
            write_json(root / "candidate_result.json", {"ipc2": 3.1})
            write_json(root / "base_latency.json", {
                "layout_delays": {"wire_cycles": 2}
            })
            write_json(root / "candidate_latency.json", {
                "layout_delays": {"wire_cycles": 1}
            })
            result = calibrate_lambda_wire(
                root / "base_result.json", root / "candidate_result.json",
                root / "base_latency.json", root / "candidate_latency.json",
                ipc1=4.0, frequency_ghz=2.0,
            )
            self.assertAlmostEqual(result["lambda_wire"], 0.05)

    def test_lambda_wire_multilevel_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for cycles, ipc2 in ((0, 3.2), (1, 3.1), (2, 3.0), (3, 2.9)):
                result = root / f"result_{cycles}.json"
                latency = root / f"latency_{cycles}.json"
                write_json(result, {"ipc2": ipc2})
                write_json(latency, {
                    # The geometry field may intentionally differ during a
                    # synthetic latency sweep; calibration must use the value
                    # actually injected into gem5.
                    "layout_delays": {"wire_cycles": 9},
                    "components_cycles": {"layout_wire": cycles},
                    "gem5_overrides": {"xbar_forward_latency": 5 + cycles},
                })
                samples.append((str(cycles), result, latency))
            report = calibrate_lambda_wire_series(
                samples, ipc1=4.0, frequency_ghz=2.0,
            )
            self.assertAlmostEqual(report["lambda_wire"], 0.05)
            self.assertAlmostEqual(report["fit"]["r_squared"], 1.0)
            self.assertTrue(report["recommendation"]["accepted_for_this_workload"])

    def test_wire_sensitivity_vectors_change_only_injected_wire(self):
        """A matched series must not alter any non-wire R2 input."""
        base = {
            "schema_version": 1,
            "components_cycles": {
                "l1i_mcpat_cacti_p": 2, "l1d_mcpat_cacti_p": 3,
                "l2_mcpat_cacti_p": 4,
                "l2_arbitration": 3, "tsv": 2, "l1_pipeline": 1,
                "layout_wire": 9,
            },
            "critical_l1d_to_l2_cycles": 22,
            "gem5_overrides": {
                "l1d_tag_latency": 3, "l1d_data_latency": 3,
                "xbar_forward_latency": 14, "xbar_response_latency": 1,
            },
            "gem5_args": ["--l1d-tag-latency", "3"],
            "layout": "/fixture/layout.json",
        }

        vectors = build_sensitivity_vectors(base, [0, 1, 2, 3])
        self.assertEqual(base["components_cycles"]["layout_wire"], 9)
        for cycle, vector in vectors.items():
            normalized_base = deepcopy(base)
            normalized_vector = deepcopy(vector)
            for candidate in (normalized_base, normalized_vector):
                del candidate["components_cycles"]["layout_wire"]
                del candidate["critical_l1d_to_l2_cycles"]
                del candidate["gem5_overrides"]["xbar_forward_latency"]
                del candidate["gem5_args"]
            self.assertEqual(normalized_vector, normalized_base)
            self.assertEqual(vector["components_cycles"]["layout_wire"], cycle)
            self.assertEqual(vector["critical_l1d_to_l2_cycles"], 13 + cycle)
            self.assertEqual(vector["gem5_overrides"]["xbar_forward_latency"], 5 + cycle)
            self.assertEqual(
                vector["gem5_args"],
                ["--l1d-tag-latency", "3", "--l1d-data-latency", "3",
                 "--xbar-forward-latency", str(5 + cycle),
                 "--xbar-response-latency", "1"],
            )

    def test_wire_sensitivity_rejects_fewer_than_three_distinct_cycles(self):
        """R2 calibration requires three distinct injected wire levels."""
        for cycles in ([0], [0, 1]):
            with self.subTest(cycles=cycles), self.assertRaisesRegex(
                ValueError, "at least three distinct"
            ):
                build_sensitivity_vectors({}, cycles)

    def test_wire_sensitivity_series_rejects_short_cycles_before_preparation(self):
        """Invalid cycle input must not build a vector or invoke an R2 run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "workflow.r2.run_wire_sensitivity._base_vector",
                side_effect=AssertionError("base vector must not be prepared"),
            ):
                with self.assertRaisesRegex(ValueError, "at least three distinct"):
                    run_series(
                        root / "r1", root / "point", root / "output", [0, 1],
                        root / "candidate.json", execute=False,
                    )

    def test_wire_sensitivity_series_rejects_short_cycles_before_output_or_execution(self):
        """One or two levels must fail before any sensitivity-series side effect."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cycles in ([0], [0, 1]):
                with self.subTest(cycles=cycles):
                    output = root / f"not-created-{'-'.join(map(str, cycles))}"
                    with patch(
                        "workflow.r2.run_wire_sensitivity._base_vector",
                        side_effect=AssertionError("base vector must not be prepared"),
                    ) as base_vector, patch(
                        "workflow.r2.run_wire_sensitivity.run_r2",
                        side_effect=AssertionError("R2 must not run"),
                    ) as r2_run:
                        with self.assertRaisesRegex(ValueError, "at least three distinct"):
                            run_series(
                                root / "r1", root / "point", output, cycles,
                                root / "candidate.json", execute=True,
                            )
                    self.assertFalse(output.exists())
                    base_vector.assert_not_called()
                    r2_run.assert_not_called()

    def test_wire_sensitivity_global_acceptance_rule(self):
        """A formal lambda exists only for four accepted, mutually-close workloads."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def report(workload, value):
                path = root / f"{workload.lower()}.json"
                write_json(path, {
                    "workload": workload,
                    "calibration": {
                        "lambda_wire": value,
                        "recommendation": {"accepted_for_this_workload": True},
                    },
                })
                return path

            paths = [
                report("FFT", 0.010), report("MATMUL", 0.011),
                report("STENCIL", 0.009), report("STREAM", 0.010),
            ]
            summary = summarize_workloads(paths)
            self.assertTrue(summary["recommendation"]["accepted"])
            self.assertEqual(summary["selected_lambda_wire"], 0.01)

            paths[-1] = report("STREAM", 0.040)
            rejected = summarize_workloads(paths)
            self.assertFalse(rejected["recommendation"]["accepted"])
            self.assertIsNone(rejected["selected_lambda_wire"])

    def test_local_wire_calibration_cannot_write_a_formal_config(self):
        """A local workload fit must not become a formal lambda configuration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for cycle, ipc2 in ((0, 3.2), (1, 3.1), (2, 3.0)):
                result = root / f"result_{cycle}.json"
                latency = root / f"latency_{cycle}.json"
                write_json(result, {"ipc2": ipc2})
                write_json(latency, {"components_cycles": {"layout_wire": cycle}})
                samples.append(f"{cycle}={result},{latency}")
            config = root / "candidate.json"
            output = root / "report.json"
            forbidden = root / "formal.json"
            write_json(config, {"layout_optimizer": {"lambda_wire": 0.0}})
            argv = [
                "calibrate_lambda_wire", "--ipc1", "4", "--frequency-ghz", "2",
                "--output", str(output), "--input-config", str(config),
                "--output-config", str(forbidden),
            ] + [argument for sample in samples for argument in ("--sample", sample)]
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                calibrate_lambda_wire_main()
            self.assertFalse(forbidden.exists())


class ParserTests(unittest.TestCase):
    def test_stats_parser_can_preserve_nonfinite_values_for_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            stats_path = Path(temporary) / "stats.txt"
            stats_path.write_text(
                "finite.counter 12\nnonfinite.counter inf\n", encoding="utf-8"
            )
            self.assertEqual(parse_gem5_stats(stats_path), {"finite.counter": 12.0})
            self.assertEqual(
                parse_gem5_stats(stats_path, include_nonfinite=True),
                {"finite.counter": 12.0, "nonfinite.counter": math.inf},
            )

    def test_communication_profile_sums_data_and_instruction_counters(self):
        stats = {
            "system.l2.demandAccesses::cpu0.data": 30.0,
            "system.l2.demandAccesses::cpu0.inst": 10.0,
            "system.l2.demandAccesses::cpu1.data": 60.0,
        }
        profile = extract_communication_profile(
            stats, 2, Path("/tmp/r1/stats.txt"), "roi", required=True
        )
        self.assertEqual(profile["status"], "available")
        self.assertEqual(profile["total_demand_accesses"], 100.0)
        self.assertEqual(profile["per_core"]["0"]["raw_demand_accesses"], 40.0)
        self.assertEqual(profile["per_core"]["0"]["normalized_weight"], 0.4)
        self.assertEqual(profile["per_core"]["1"]["normalized_weight"], 0.6)
        self.assertEqual(profile["per_core"]["0"]["matched_counters"], [
            "system.l2.demandAccesses::cpu0.data",
            "system.l2.demandAccesses::cpu0.inst",
        ])
        self.assertEqual(profile["source_stats"], "/tmp/r1/stats.txt")
        self.assertEqual(profile["instruction_window_scope"], "roi")
        self.assertEqual(
            profile["counter_family"],
            "system.l2.demandAccesses per CPU requestor",
        )

    def test_optional_communication_profile_records_invalid_inputs(self):
        cases = {
            "missing core": (
                {"system.l2.demandAccesses::cpu0.data": 1.0},
                "missing shared-L2 demand counter for core 1",
            ),
            "negative": ({
                "system.l2.demandAccesses::cpu0.data": -1.0,
                "system.l2.demandAccesses::cpu1.data": 2.0,
            }, "negative"),
            "non-finite": ({
                "system.l2.demandAccesses::cpu0.data": math.inf,
                "system.l2.demandAccesses::cpu1.data": 2.0,
            }, "non-finite"),
            "all zero": ({
                "system.l2.demandAccesses::cpu0.data": 0.0,
                "system.l2.demandAccesses::cpu1.data": 0.0,
            }, "total demand accesses must be positive"),
        }
        for label, (stats, message) in cases.items():
            with self.subTest(label=label):
                profile = extract_communication_profile(
                    stats, 2, Path("/tmp/stats.txt"), "roi", required=False
                )
                self.assertEqual(profile["status"], "unavailable")
                self.assertTrue(
                    any(message in item for item in profile["diagnostics"]),
                    profile["diagnostics"],
                )

    def test_required_communication_profile_rejects_invalid_inputs(self):
        cases = ({
            "system.l2.demandAccesses::cpu0.data": 1.0,
        }, {
            "system.l2.demandAccesses::cpu0.data": -1.0,
            "system.l2.demandAccesses::cpu1.data": 2.0,
        }, {
            "system.l2.demandAccesses::cpu0.data": math.nan,
            "system.l2.demandAccesses::cpu1.data": 2.0,
        }, {
            "system.l2.demandAccesses::cpu0.data": 0.0,
            "system.l2.demandAccesses::cpu1.data": 0.0,
        })
        for stats in cases:
            with self.subTest(stats=stats):
                with self.assertRaisesRegex(
                        ValueError, "communication profile unavailable"):
                    extract_communication_profile(
                        stats, 2, Path("/tmp/stats.txt"), "roi", required=True
                    )

    def test_build_model_writes_available_communication_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1_dir = root / "r1"
            r1_dir.mkdir()
            metadata = {
                "num_cores": 4,
                "l1i_size": "32kB",
                "l1d_size": "32kB",
                "l2_size": "512kB",
                "l1_associativity": 2,
                "l2_associativity": 8,
                "cache_line_bytes": 64,
                "instruction_window_scope": "roi",
            }
            write_json(r1_dir / "r1_metadata.json", metadata)
            (r1_dir / "stats.txt").write_text(
                "system.cpu0.commitStats0.numInsts 100\n"
                "system.cpu1.commitStats0.numInsts 100\n"
                "system.cpu2.commitStats0.numInsts 100\n"
                "system.cpu3.commitStats0.numInsts 100\n"
                "system.cpu0.numCycles 100\n"
                "system.cpu1.numCycles 100\n"
                "system.cpu2.numCycles 100\n"
                "system.cpu3.numCycles 100\n"
                "system.l2.demandAccesses::cpu0.data 25\n"
                "system.l2.demandAccesses::cpu1.data 75\n"
                "system.l2.demandAccesses::cpu2.data 0\n"
                "system.l2.demandAccesses::cpu3.data 0\n",
                encoding="utf-8",
            )
            mcpat_path = root / "mcpat.json"
            contract = build_cache_contract(
                metadata, technology_nm=45, temperature_k=320,
                device_type=0, interconnect_projection_type=1,
            )
            modules = []
            for core in range(4):
                modules.extend((
                    {"name": f"core{core}_logic", "kind": "core_logic", "core": core,
                     "area_mm2": 1.0, "dynamic_power_w": 1.0,
                     "subthreshold_leakage_w": 0.08,
                     "gate_leakage_w": 0.02,
                     "leakage_power_w": 0.1, "total_power_w": 1.1},
                    {"name": f"core{core}_l1i", "kind": "l1i", "core": core,
                     "area_mm2": 0.2, "dynamic_power_w": 0.1,
                     "subthreshold_leakage_w": 0.015,
                     "gate_leakage_w": 0.005,
                     "leakage_power_w": 0.02, "total_power_w": 0.12},
                    {"name": f"core{core}_l1d", "kind": "l1d", "core": core,
                     "area_mm2": 0.3, "dynamic_power_w": 0.1,
                     "subthreshold_leakage_w": 0.015,
                     "gate_leakage_w": 0.005,
                     "leakage_power_w": 0.02, "total_power_w": 0.12},
                ))
            modules.append({
                "name": "shared_l2", "kind": "l2", "area_mm2": 1.0,
                "dynamic_power_w": 0.1,
                "subthreshold_leakage_w": 0.08,
                "gate_leakage_w": 0.02, "leakage_power_w": 0.1,
                "total_power_w": 0.2,
            })
            records = []
            for cache, core in (
                *(("l1i", core) for core in range(4)),
                *(("l1d", core) for core in range(4)),
                ("l2", None),
            ):
                record = {
                    "cache": cache, "core": core,
                    "access_time_s": 1e-9, "cycle_time_s": 2e-9,
                    "height_mm": 0.5, "width_mm": 1.0,
                    "mcpat_version": "1.3", "model": "embedded-cacti-p",
                }
                record["record_id"] = mcpat_embedded_cache_record_identity(record)
                records.append(record)
            module_totals = {
                field: sum(module[field] for module in modules)
                for field in (
                    "area_mm2", "dynamic_power_w", "subthreshold_leakage_w",
                    "gate_leakage_w", "leakage_power_w", "total_power_w",
                )
            }
            write_json(mcpat_path, {
                "modules": modules, "module_totals": module_totals,
                "checks": {
                    "core_count": 4,
                    "core_logic_granularity": "legacy aggregate core_logic fallback",
                },
                "power_provenance": {"postprocessing": "none"},
                "cache_contract": contract,
                "embedded_cacti_p": {
                    "schema_version": 1,
                    "authority": "McPAT 1.3 embedded CACTI-P",
                    "records": records,
                },
                "provenance": {
                    "schema_version": 1,
                    "authority": "CLIP strict patched McPAT 1.3 runner",
                    "hashes": {
                        "xml_sha256": "1" * 64,
                        "mapping_sha256": "2" * 64,
                        "output_sha256": "3" * 64,
                        "binary_sha256": "4" * 64,
                        "patch_sha256": "5" * 64,
                    },
                },
            })
            output = root / "modules.json"
            metadata.pop("instruction_window_scope")
            write_json(r1_dir / "r1_metadata.json", metadata)
            model = build_model(
                r1_dir, mcpat_path, output,
                require_communication_profile=True,
                require_granular_cores=False,
            )
            self.assertEqual(model, read_json(output))
            self.assertEqual(model["architecture"]["instruction_window_scope"], "cpu0")
            self.assertEqual(
                model["communication_profile"]["instruction_window_scope"], "cpu0"
            )
            self.assertEqual(model["communication_profile"]["status"], "available")
            self.assertEqual(
                model["communication_profile"]["per_core"]["1"]
                     ["normalized_weight"],
                0.75,
            )

    def test_subtraction_rebuilds_derived_power_after_rounding_clamp(self):
        """Primitive rounding residuals must not break derived power sums."""
        base = {
            "area_mm2": 1.0,
            "dynamic_power_w": 1.0,
            "subthreshold_leakage_w": 0.2,
            "gate_leakage_w": 0.1,
            "leakage_power_w": 0.3,
            "total_power_w": 1.3,
        }
        child = {
            "area_mm2": 0.5,
            "dynamic_power_w": 1.0000023,
            "subthreshold_leakage_w": 0.15,
            "gate_leakage_w": 0.04,
            "leakage_power_w": 0.19,
            "total_power_w": 1.1900023,
        }

        remainder = subtract(base, [child])

        self.assertEqual(remainder["dynamic_power_w"], 0.0)
        self.assertEqual(
            remainder["leakage_power_w"],
            remainder["subthreshold_leakage_w"] + remainder["gate_leakage_w"],
        )
        self.assertEqual(
            remainder["total_power_w"],
            remainder["dynamic_power_w"] + remainder["leakage_power_w"],
        )
        self.assertAlmostEqual(
            remainder["subtraction_diagnostics"]["raw_residuals"]["total_power_w"],
            0.1099977,
        )
        self.assertAlmostEqual(
            remainder["subtraction_diagnostics"]["clipped_negative_magnitudes"][
                "dynamic_power_w"
            ],
            0.0000023,
        )

    def test_cacti_parser(self):
        text = """Access time (ns): 1.5
Cycle time (ns): 2.0
Total dynamic read energy per access (nJ): 0.1
Total dynamic write energy per access (nJ): 0.2
Total leakage power of a bank (mW): 3.0
Cache height x width (mm): 2 x 4
"""
        result = parse_cacti_output(text)
        self.assertEqual(result["area_mm2"], 8.0)

    def test_mcpat_parser_and_cache_subtraction(self):
        sep = "*" * 40
        processor = metric_lines(12, 6, 1, 1)
        core = (metric_lines(8, 4, 0.8, 0.2) + "Instruction Cache:\n" +
                metric_lines(1, 0.5, 0.1, 0.1, "    ") + "Data Cache:\n" +
                metric_lines(2, 1, 0.2, 0.1, "    "))
        l2 = metric_lines(3, 1, 0.3, 0.2)
        text = ("Technology 45 nm\nCore clock Rate(MHz) 2000\nProcessor:\n" + processor +
                f"\n{sep}\nCore:\n" + core + f"\n{sep}\nL2\n" + l2)
        result = parse_mcpat_text(text)
        logic = result["modules"][0]
        self.assertAlmostEqual(logic["area_mm2"], 5.0)
        self.assertAlmostEqual(logic["dynamic_power_w"], 2.5)
        self.assertAlmostEqual(logic["leakage_power_w"], 0.5)
        self.assertAlmostEqual(logic["total_power_w"], 3.0)
        self.assertEqual(result["power_provenance"]["postprocessing"], "none")

    def test_detailed_mcpat_preserves_functional_core_blocks(self):
        sep = "*" * 40
        processor = metric_lines(20, 10, 2, 1)
        core = (
            metric_lines(10, 5, 1, 0.5)
            + "Instruction Fetch Unit:\n" + metric_lines(2, 1, 0.2, 0.1, "    ")
            + "Instruction Cache:\n" + metric_lines(1, 0.2, 0.1, 0.05, "      ")
            + "Renaming Unit:\n" + metric_lines(1, 0.5, 0.1, 0.05, "    ")
            + "Load Store Unit:\n" + metric_lines(3, 1.5, 0.3, 0.15, "    ")
            + "Data Cache:\n" + metric_lines(2, 0.4, 0.2, 0.1, "      ")
            + "Memory Management Unit:\n" + metric_lines(1, 0.5, 0.1, 0.05, "    ")
            + "Execution Unit:\n" + metric_lines(2, 1.0, 0.2, 0.1, "    ")
        )
        l2 = metric_lines(3, 1, 0.3, 0.2)
        text = (
            "Technology 45 nm\nCore clock Rate(MHz) 2000\nProcessor:\n" + processor
            + f"\n{sep}\nCore:\n" + core + f"\n{sep}\nL2\n" + l2
        )
        result = parse_mcpat_text(text)
        kinds = {module["kind"] for module in result["modules"]}
        self.assertIn("core_exec", kinds)
        self.assertIn("core_ifu", kinds)
        self.assertEqual(
            result["checks"]["core_logic_granularity"],
            "McPAT top-level functional blocks",
        )

    def test_strict_parser_rejects_aggregate_core_fallback(self):
        sep = "*" * 40
        processor = metric_lines(12, 6, 1, 1)
        core = (
            metric_lines(8, 4, 0.8, 0.2)
            + "Instruction Cache:\n" + metric_lines(1, 0.5, 0.1, 0.1, "    ")
            + "Data Cache:\n" + metric_lines(2, 1, 0.2, 0.1, "    ")
        )
        text = (
            "Technology 45 nm\nCore clock Rate(MHz) 2000\nProcessor:\n" + processor
            + f"\n{sep}\nCore:\n" + core + f"\n{sep}\nL2\n" + metric_lines(3, 1, 0.3, 0.2)
        )
        self.assertEqual(
            parse_mcpat_text(
                text, require_granular_cores=False, require_embedded_cacti=False,
            )["checks"]["core_logic_granularity"],
            "legacy aggregate core_logic fallback",
        )
        with self.assertRaises(ValueError):
            parse_mcpat_text(text, expected_core_count=4)
        with self.assertRaises(ValueError):
            parse_mcpat_text(text, expected_core_count=1, require_granular_cores=True)
        with self.assertRaises(ValueError):
            parse_mcpat_text(text, expected_core_count=1, require_embedded_cacti=True)

    def test_legacy_diagnostic_helper_consumes_cacti_cache_geometry(self):
        """Standalone CACTI geometry remains diagnostic, not a corrected artifact."""
        modules = [
            {"name": "core0_logic", "kind": "core_logic", "area_mm2": 10.0,
             "dynamic_power_w": 1.0, "leakage_power_w": 0.1, "total_power_w": 1.1},
            {"name": "core0_l1i", "kind": "l1i", "area_mm2": 20.0,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.1, "total_power_w": 0.2},
            {"name": "core0_l1d", "kind": "l1d", "area_mm2": 30.0,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.1, "total_power_w": 0.2},
            {"name": "shared_l2", "kind": "l2", "area_mm2": 40.0,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.1, "total_power_w": 0.2},
        ]
        metadata = {"l1i_size": "32kB", "l1d_size": "64kB", "l2_size": "512kB"}
        cacti = {"records": [
            {"level": "l1i", "size": "32kB", "size_bytes": 32 * 1024,
             "area_mm2": 0.20, "width_mm": 0.50, "height_mm": 0.40,
             "value_source": "local CACTI run"},
            {"level": "l1d", "size": "64kB", "size_bytes": 64 * 1024,
             "area_mm2": 0.30, "width_mm": 0.60, "height_mm": 0.50,
             "value_source": "local CACTI run"},
            {"level": "l2", "size": "512kB", "size_bytes": 512 * 1024,
             "area_mm2": 2.50, "width_mm": 2.50, "height_mm": 1.00,
             "value_source": "local CACTI run"},
        ]}
        physical = apply_physical_areas(modules, metadata, cacti)
        by_kind = {module["kind"]: module for module in physical}
        self.assertAlmostEqual(by_kind["core_logic"]["area_mm2"], 10.0)
        self.assertEqual(by_kind["core_logic"]["area_source"], "McPAT")
        self.assertAlmostEqual(by_kind["l1d"]["area_mm2"], 0.30)
        self.assertAlmostEqual(by_kind["l1d"]["mcpat_reported_area_mm2"], 30.0)
        self.assertEqual(by_kind["l1d"]["area_source"], "local CACTI run")
        self.assertAlmostEqual(by_kind["l2"]["preferred_width_mm"], 2.5)
        self.assertNotIn("area_before_global_scale_mm2", by_kind["l2"])

    def test_pipeline_rejects_removed_paper_table_ii_switch(self):
        config = {
            "schema_version": 1,
            "cacti": {"use_paper_table_ii": True},
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {"r_convec_k_per_w": 5.0},
            "mcpat": {},
            "delay": {},
        }
        with self.assertRaisesRegex(ValueError, "use_paper_table_ii has been removed"):
            validate_config(config, "fixed-bin")

    def test_pipeline_rejects_invalid_hotspot_materialization_controls(self):
        """Identification controls must not accept coercible configuration values."""
        # Break caught: accepting an invalid value can silently materialize
        # grid-cell/default traces instead of the requested identification case.
        base = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {"r_convec_k_per_w": 5.0},
            "mcpat": {},
            "delay": {},
        }
        invalid_cases = (
            ("input_granularity", "cells", "input_granularity"),
            ("compact_trace", 1, "compact_trace"),
            ("ptrace_precision", True, "ptrace_precision"),
            ("ptrace_precision", 0, "ptrace_precision"),
        )
        for key, value, label in invalid_cases:
            with self.subTest(key=key, value=value):
                config = deepcopy(base)
                config["physical"][key] = value
                with self.assertRaisesRegex(ValueError, label):
                    validate_config(config, "fixed-bin")

    def test_proxy_diagnostic_preserves_hotspot_materialization_contract(self):
        """A module-level proxy fit must not be replayed as grid-cell HotSpot."""
        config = {
            "physical": {
                "grid_size": 64,
                "utilization": 0.7,
                "r_convec_k_per_w": 1.042,
                "input_granularity": "module",
                "compact_trace": True,
                "ptrace_precision": 9,
                "thermal_stack": {"local_resistance_scale": 1.0},
            },
            "frequency": {"ambient_c": 25.0},
        }
        sample = {
            "model": "/tmp/modules.json", "model_label": "probe",
            "tier": 1, "row": 0, "column": 0, "fx": 0.0, "fy": 0.0,
            "layout": {"die_width_mm": 1.0, "modules": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            sample["case_dir"] = str(Path(temporary) / "case")
            with patch("workflow.thermal.calibrate_proxy.materialize") as materialize_mock, \
                    patch("workflow.thermal.calibrate_proxy.run_hotspot",
                          return_value={"tmax_c": 100.0, "peak_unit": "t0"}):
                result = run_one(sample, config, Path("/tmp/hotspot"), force=True)

        self.assertEqual(result["tmax_c"], 100.0)
        self.assertEqual(materialize_mock.call_args.kwargs, {
            "input_granularity": "module",
            "compact_trace": True,
            "ptrace_precision": 9,
        })


class GridTests(unittest.TestCase):
    def model(self):
        modules = []
        for core in range(4):
            modules.append({"name": f"core{core}_logic", "kind": "core_logic", "core": core,
                            "area_mm2": 1.0, "dynamic_power_w": 0.8,
                            "leakage_power_w": 0.2, "total_power_w": 1.0})
        modules.extend((
            {"name": "shared_l2", "kind": "l2", "area_mm2": 1.0,
             "dynamic_power_w": 0.4, "leakage_power_w": 0.1, "total_power_w": 0.5},
            {"name": "noc", "kind": "interconnect", "area_mm2": 0.1,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.01, "total_power_w": 0.11},
        ))
        return {"schema_version": 1, "ipc1": 4.0, "gamma": 0.21,
                "modules": modules, "totals": {"total_power_w": 4.61,
                "dynamic_power_w": 3.7, "leakage_power_w": 0.91}}

    def r2_model(self):
        model = self.model()
        metadata = {
            "num_cores": 4, "cpu_clock": "2GHz",
            "l1i_size": "32kB", "l1d_size": "32kB", "l2_size": "512kB",
            "l1_associativity": 2, "l2_associativity": 8,
            "cache_line_bytes": 64,
        }
        records = []
        for cache, core, access in (
            *(("l1i", core, 1.0e-9) for core in range(4)),
            *(("l1d", core, 1.0e-9) for core in range(4)),
            ("l2", None, 2.0e-9),
        ):
            record = {
                "cache": cache, "core": core,
                "access_time_s": access, "cycle_time_s": access * 2.0,
                "height_mm": 0.5, "width_mm": 1.0,
                "mcpat_version": "1.3", "model": "embedded-cacti-p",
            }
            record["record_id"] = mcpat_embedded_cache_record_identity(record)
            records.append(record)
        model.update({
            "schema_version": 3,
            "architecture": metadata,
            "cache_contract": build_cache_contract(
                metadata, technology_nm=45, temperature_k=320,
                device_type=0, interconnect_projection_type=1,
            ),
            "cache_authority": "McPAT 1.3 embedded CACTI-P",
            "embedded_cacti_p": {
                "schema_version": 1,
                "authority": "McPAT 1.3 embedded CACTI-P",
                "records": records,
            },
            "mcpat_provenance": {
                "schema_version": 1,
                "authority": "CLIP strict patched McPAT 1.3 runner",
                "hashes": {
                    "xml_sha256": "1" * 64,
                    "mapping_sha256": "2" * 64,
                    "output_sha256": "3" * 64,
                    "binary_sha256": "4" * 64,
                    "patch_sha256": "5" * 64,
                },
            },
        })
        return model

    def test_discrete_wire_score_blocks_harmful_rounding_boundary(self):
        ipc1 = 4.31314420772608
        lambda_wire = 0.0020119160767721133
        fixed_score, fixed_cycle = discrete_wire_score(
            ipc1, 0.7894564656933903, lambda_wire, 1.3580618602253112,
            "nearest",
        )
        proposed_score, proposed_cycle = discrete_wire_score(
            ipc1, 0.7906404937722684, lambda_wire, 1.5577477284674477,
            "nearest",
        )

        self.assertEqual((fixed_cycle, proposed_cycle), (1, 2))
        self.assertLess(fixed_score, proposed_score)

    def test_discrete_partition_rejects_invalid_grid_and_disabled_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            for steps in (2, 4):
                with self.subTest(steps=steps):
                    with self.assertRaisesRegex(ValueError, "odd integer"):
                        optimize(
                            root / "modules.json", root / "layout.json",
                            root / "report.json",
                            wire_objective="discrete-partition",
                            partition_grid_steps=steps,
                        )
            with self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
                optimize(
                    root / "modules.json", root / "layout.json",
                    root / "report.json",
                    wire_objective="discrete-partition",
                    partition_grid_steps=3, include_fixed_baseline=False,
                )

    def test_discrete_partition_is_deterministic_and_reports_integer_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            reports = []
            for run in (1, 2):
                reports.append(optimize(
                    root / "modules.json", root / f"layout-{run}.json",
                    root / f"report-{run}.json", allowed_l2_tiers=[1],
                    require_scipy=False, wire_objective="discrete-partition",
                    wire_aggregation="mean", partition_grid_steps=5,
                    include_fixed_baseline=True,
                ))

            report = reports[0]
            self.assertEqual(
                report["parameters"]["wire_objective"], "discrete-partition"
            )
            self.assertEqual(report["discrete_search"]["grid_steps"], 5)
            self.assertTrue(
                report["discrete_search"]["fixed_baseline_included"]
            )
            self.assertIn(
                report["selected"]["origin"], ("fixed-bin", "partition-grid")
            )
            self.assertIsInstance(report["selected"]["r2_wire_cycles"], int)
            self.assertEqual(
                report["selected"]["wire_objective_cycles"],
                report["selected"]["r2_wire_cycles"],
            )
            self.assertGreaterEqual(
                len(report["discrete_search"]["partitions"]), 1
            )
            self.assertEqual(reports[0]["selected"], reports[1]["selected"])
            self.assertEqual(
                reports[0]["discrete_search"]["partitions"],
                reports[1]["discrete_search"]["partitions"],
            )

    def test_steady_discrete_optimizer_delegates_to_shared_partition_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())

            with patch(
                "workflow.floorplan.optimize_layout.search_discrete_partitions",
                wraps=shared_discrete_partition_search,
            ) as shared_search:
                report = optimize(
                    root / "modules.json", root / "layout.json",
                    root / "report.json", allowed_l2_tiers=[1],
                    require_scipy=False, wire_objective="discrete-partition",
                    wire_aggregation="mean", partition_grid_steps=5,
                    include_fixed_baseline=True,
                )

            shared_search.assert_called_once()
            self.assertEqual(shared_search.call_args.kwargs["grid_steps"], 5)
            self.assertTrue(
                shared_search.call_args.kwargs["include_fixed_baseline"]
            )
            self.assertTrue(report["discrete_search"]["shared_partition_engine"])

    def test_pipeline_forwards_discrete_partition_options_to_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.granular_model())
            config = {
                "frequency": {
                    "ambient_c": 25.0, "f0_ghz": 2.0, "fmin_ghz": 0.4,
                    "tsafe_c": 95.0,
                },
                "physical": {"utilization": 0.70, "r_convec_k_per_w": 5.0},
                "layout_optimizer": {
                    "alpha": 0.3, "beta": 0.1, "cross_tier_weight": 0.65,
                    "lc_die_side_ratio": 0.3525311644629249,
                    "lambda_wire": 0.0020119160767721133,
                    "allowed_l2_tiers": [1], "require_scipy": False,
                    "wire_objective": "discrete-partition",
                    "partition_grid_steps": 3,
                    "include_fixed_baseline": True,
                },
                "delay": {
                    "wire_rounding": "nearest", "wire_aggregation": "mean",
                },
            }

            report = optimize_clip3d_layout(
                root / "modules.json", root / "layout.json",
                root / "report.json", config,
            )

            self.assertEqual(report["discrete_search"]["grid_steps"], 3)
            self.assertTrue(
                report["discrete_search"]["fixed_baseline_included"]
            )
            self.assertEqual(
                report["parameters"]["lc_die_side_ratio"],
                0.3525311644629249,
            )

    def test_optimizer_uses_explicit_lc_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            side = baseline_layout(model)["die_width_mm"]
            reports = [
                optimize(
                    root / "modules.json", root / f"layout-{ratio}.json",
                    root / f"report-{ratio}.json", allowed_l2_tiers=[1],
                    require_scipy=False, lc_die_side_ratio=ratio,
                )
                for ratio in (0.25, 0.75)
            ]

            for ratio, report in zip((0.25, 0.75), reports):
                self.assertEqual(report["parameters"]["lc_die_side_ratio"], ratio)
                self.assertEqual(report["parameters"]["lc_mm"], side * ratio)
            self.assertNotAlmostEqual(
                reports[0]["selected"]["proxy_tmax_c"],
                reports[1]["selected"]["proxy_tmax_c"],
                places=9,
            )

    def test_proxy_anchor_uses_hotspot_baseline_without_changing_gradient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            raw = optimize(
                root / "modules.json", root / "raw-layout.json",
                root / "raw-report.json", allowed_l2_tiers=[1],
                require_scipy=False, lc_die_side_ratio=0.25,
            )
            anchored = optimize(
                root / "modules.json", root / "anchored-layout.json",
                root / "anchored-report.json", allowed_l2_tiers=[1],
                require_scipy=False, lc_die_side_ratio=0.25,
                proxy_anchor_tmax_c=110.0,
            )

        self.assertAlmostEqual(anchored["baseline"]["proxy_tmax_c"], 110.0)
        self.assertAlmostEqual(
            anchored["parameters"]["proxy_anchor_offset_c"],
            110.0 - raw["baseline"]["proxy_tmax_c"],
        )
        for candidate in anchored["candidates"]:
            self.assertAlmostEqual(
                candidate["proxy_tmax_c"] - candidate["proxy_tmax_raw_c"],
                anchored["parameters"]["proxy_anchor_offset_c"],
            )
        self.assertTrue(anchored["thermal_proxy"]["anchor"]["enabled"])
        self.assertLess(anchored["baseline"]["proxy_frequency_ghz"], 2.0)
        observability = anchored["observability_diagnostics"]
        self.assertTrue(observability["sampled_thermal_frequency_term_active"])
        self.assertGreaterEqual(
            len(observability["sampled_l2_positions"]), 2
        )
        self.assertIn(
            "thermally_limited", observability["sampled_proxy_frequency_states"]
        )
        baseline_bottom_power = sum(
            module["total_power_w"]
            for module in baseline_layout(self.model())["modules"]
            if module["tier"] == 0
        )
        self.assertTrue(all(
            observation["bottom_power_w"] == baseline_bottom_power
            for observation in observability["sampled_l2_positions"]
        ))

    def test_optimizer_rejects_nonfinite_proxy_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            with self.assertRaisesRegex(ValueError, "anchor temperature"):
                optimize(
                    root / "modules.json", root / "layout.json",
                    root / "report.json", proxy_anchor_tmax_c=math.nan,
                )

    def test_proxy_calibration_respects_configured_lc_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = baseline_layout(self.model())
            case_dir = root / "case"
            case_dir.mkdir()
            write_json(case_dir / "layout.json", layout)
            sample = {"case_dir": str(case_dir)}
            base = {
                "physical": {"r_convec_k_per_w": 5.0},
                "frequency": {"ambient_c": 25.0},
                "layout_optimizer": {"proxy_spatial_model": "center"},
            }
            short = deepcopy(base)
            short["layout_optimizer"]["lc_die_side_ratio"] = 0.1
            long = deepcopy(base)
            long["layout_optimizer"]["lc_die_side_ratio"] = 0.9
            parameters = [0.3, 0.0, 0.9]

            short_prediction = proxy_prediction(sample, parameters, short)
            long_prediction = proxy_prediction(sample, parameters, long)

        self.assertNotAlmostEqual(short_prediction, long_prediction, places=9)

    def test_optimizer_rejects_nonpositive_or_nonfinite_lc_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            for ratio in (0.0, -0.25, math.nan, math.inf):
                with self.subTest(ratio=ratio):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        optimize(
                            root / "modules.json", root / "layout.json",
                            root / "report.json", lc_die_side_ratio=ratio,
                        )

    def test_exact_power_conservation(self):
        gridded = grid_power(baseline_layout(self.model()), 8)
        for tier in gridded["power_conservation"]:
            for field in ("dynamic_power_w", "leakage_power_w", "total_power_w"):
                self.assertLess(abs(tier[field]["residual"]), 1e-10)

    def test_materialize_records_selected_grid_and_granularity(self):
        """Manifest provenance must describe the actual HotSpot inputs."""
        # Break caught: keeping the historical 32x32 manifest text obscures
        # a parameter-identification run's grid resolution.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "modules.json"
            write_json(model_path, self.model())

            manifest = materialize(
                model_path, root / "case", grid_size=64,
                input_granularity="module",
            )

        self.assertEqual(manifest["paper_parameters"][0], "64x64 per tier")
        self.assertEqual(manifest["input_granularity"], "module")

    def test_fixed_bin_l2_is_lower_left(self):
        layout = baseline_layout(self.model())
        l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
        self.assertEqual(l2["tier"], 1)
        self.assertAlmostEqual(l2["x_mm"], 0.0)
        self.assertAlmostEqual(l2["y_mm"], 0.0)

    def test_fixed_bin_preserves_l2_preferred_aspect_ratio(self):
        model = self.model()
        l2 = next(module for module in model["modules"] if module["kind"] == "l2")
        l2["preferred_width_mm"] = 1.5
        l2["preferred_height_mm"] = 2.0 / 3.0
        placed = next(
            module for module in baseline_layout(model)["modules"]
            if module["kind"] == "l2"
        )
        self.assertAlmostEqual(placed["width_mm"], 1.5)
        self.assertAlmostEqual(placed["height_mm"], 2.0 / 3.0)

    def test_proxy_calibration_candidates_are_legal_and_split_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "modules.json"
            write_json(path, self.model())
            candidates = candidate_layouts(path, grid_points=3, utilization=0.70)
            self.assertGreater(len(candidates), 0)
            for candidate in candidates:
                layout = candidate["layout"]
                self.assertEqual(len(layout["modules"]), len(self.model()["modules"]))
            self.assertEqual(sample_split(0, 0, 0, 0, 3), "validation")
            self.assertEqual(sample_split(0, 1, 1, 0, 3), "validation")
            self.assertEqual(sample_split(0, 0, 1, 0, 3), "train")

    def test_proxy_calibration_can_restrict_samples_to_p1_top_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "modules.json"
            write_json(path, self.model())
            candidates = candidate_layouts(
                path, grid_points=3, utilization=0.70, allowed_l2_tiers=(1,)
            )
            self.assertTrue(candidates)
            self.assertEqual({item["tier"] for item in candidates}, {1})

    def test_strict_p1_acceptance_allows_fixed_unidentifiable_beta(self):
        checks = proxy_acceptance_checks(
            validation={"rmse_c": 0.4, "spatial_centered_rmse_c": 0.1,
                        "spatial_spearman": 0.9},
            baseline={"rmse_c": 0.5, "spatial_centered_rmse_c": 0.2},
            fitted_rank=2, selected_weight=0.7,
            beta_status="fixed_unidentifiable_under_p1",
        )
        self.assertTrue(checks["beta_policy_valid"])
        self.assertNotIn("beta_tier_effect_identifiable", checks)

    def test_strict_p1_rejects_non_three_point_grid_before_creating_cases(self):
        """A strict-P1 run must not begin HotSpot work with a non-3x3 grid."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            model["power_provenance"] = {
                "dynamic": "McPAT Runtime Dynamic",
                "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
                "postprocessing": "none",
            }
            model_path = root / "modules.json"
            config_path = root / "strict.json"
            output_dir = root / "calibration"
            write_json(model_path, model)
            write_json(config_path, {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
                "layout_optimizer": {"alpha": 0.3, "beta": 0.0,
                                     "cross_tier_weight": 0.65},
                "formal_validation": {"strict_p1": True},
            })
            with patch("workflow.thermal.calibrate_proxy.run_one",
                       side_effect=AssertionError("HotSpot work must not start")):
                with self.assertRaisesRegex(
                        ValueError, "strict P1 calibration requires grid_points == 3"):
                    calibrate(
                        [("fft", model_path)], config_path, output_dir,
                        grid_points=4, workers=1, allowed_l2_tiers=(1,),
                        fixed_beta=0.0, target_grid_size=32,
                    )
            self.assertFalse((output_dir / "cases").exists())

    def test_strict_p1_rejects_non_target_hotspot_grid_before_creating_cases(self):
        """A strict-P1 run must use the fixed 32-cell target validation grid."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            model["power_provenance"] = {
                "dynamic": "McPAT Runtime Dynamic",
                "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
                "postprocessing": "none",
            }
            model_path = root / "modules.json"
            config_path = root / "strict.json"
            output_dir = root / "calibration"
            write_json(model_path, model)
            write_json(config_path, {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
                "layout_optimizer": {"alpha": 0.3, "beta": 0.0,
                                     "cross_tier_weight": 0.65},
                "formal_validation": {"strict_p1": True},
            })
            with patch("workflow.thermal.calibrate_proxy.run_one",
                       side_effect=AssertionError("HotSpot work must not start")):
                with self.assertRaisesRegex(
                        ValueError, "strict P1 calibration requires target_grid_size == 32"):
                    calibrate(
                        [("fft", model_path)], config_path, output_dir,
                        grid_points=3, workers=1, allowed_l2_tiers=(1,),
                        fixed_beta=0.0, target_grid_size=16,
                    )
            self.assertFalse((output_dir / "cases").exists())

    def test_strict_p1_report_promotes_the_held_out_cross_tier_weight(self):
        """The report's promotable fit must retain its selected held-out weight."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            model["power_provenance"] = {
                "dynamic": "McPAT Runtime Dynamic",
                "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
                "postprocessing": "none",
            }
            model_path = root / "modules.json"
            config_path = root / "strict.json"
            output_dir = root / "calibration"
            write_json(model_path, model)
            write_json(config_path, {
                "frequency": {"ambient_c": 25.0},
                "physical": {"grid_size": 4, "utilization": 0.70,
                             "r_convec_k_per_w": 5.0},
                "layout_optimizer": {"alpha": 0.3, "beta": 0.0,
                                     "cross_tier_weight": 0.65},
                "formal_validation": {"strict_p1": True},
            })

            def completed_sample(sample, config, hotspot, force):
                case_dir = Path(sample["case_dir"])
                case_dir.mkdir(parents=True)
                write_json(case_dir / "layout.json", sample["layout"])
                return {
                    key: value for key, value in sample.items() if key != "layout"
                } | {"tmax_c": 60.0, "peak_unit": "core0_logic", "reused": False}

            def synthetic_fit(samples, config, starts=None, spatial_weight=0.0,
                              fixed_beta=None, fixed_cross_tier_weight=None):
                cross_weight = (0.91 if fixed_cross_tier_weight is None
                                else fixed_cross_tier_weight)
                return {
                    "parameters": {"alpha": 1.25, "beta": 0.0,
                                   "cross_tier_weight": cross_weight},
                    "rank": 2 if fixed_cross_tier_weight is None else 1,
                    "active_parameters": (
                        ["alpha", "cross_tier_weight"]
                        if fixed_cross_tier_weight is None else ["alpha"]
                    ),
                    "beta_status": "fixed_unidentifiable_under_p1",
                    "fixed_cross_tier_weight": fixed_cross_tier_weight,
                }

            def synthetic_cross_validation(*args, **kwargs):
                return {
                    "selected": {
                        "cross_tier_weight": 0.25,
                        "training_fit": synthetic_fit(
                            *args[:2], fixed_beta=kwargs["fixed_beta"],
                            fixed_cross_tier_weight=0.25,
                        ),
                    },
                    "candidates": [],
                }

            def synthetic_metrics(samples, parameters, config):
                selected = parameters[2] == 0.25
                return {
                    "rmse_c": 0.4 if selected else 0.5,
                    "spatial_centered_rmse_c": 0.1 if selected else 0.2,
                    "spatial_spearman": 0.9,
                }

            with patch("workflow.thermal.calibrate_proxy.run_one", completed_sample), \
                 patch("workflow.thermal.calibrate_proxy.fit", synthetic_fit), \
                 patch("workflow.thermal.calibrate_proxy.cross_validate_weight",
                       synthetic_cross_validation), \
                 patch("workflow.thermal.calibrate_proxy.metrics", synthetic_metrics), \
                 patch("workflow.thermal.calibrate_proxy.write_samples_csv"):
                report = calibrate(
                    [("fft", model_path)], config_path, output_dir,
                    grid_points=3, workers=1, allowed_l2_tiers=(1,),
                    fixed_beta=0.0, target_grid_size=32,
                )

            self.assertEqual(read_json(output_dir / "calibration_report.json"), report)
            self.assertEqual(
                report["fit"]["parameters"]["cross_tier_weight"],
                report["cross_validation"]["selected"]["cross_tier_weight"],
            )
            self.assertEqual(report["fit"]["parameters"]["alpha"], 1.25)
            self.assertEqual(report["fit"]["parameters"]["beta"], 0.0)
            self.assertTrue(
                report["recommendation"]["checks"]
                ["active_parameter_jacobian_full_rank"]
            )

    def test_external_proxy_cases_preserve_spatial_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            for name in ("layout.json", "thermal_result.json", "hotspot_manifest.json"):
                write_json(case / name, {})
            group, label, parsed = parse_external_case(f"fft:center={case}")
            self.assertEqual(group, "fft")
            self.assertEqual(label, "center")
            self.assertEqual(parsed, case.resolve())

    def test_layout_delays_come_from_final_tiers_and_coordinates(self):
        layout = baseline_layout(self.model())
        delays = derive_layout_delays(layout)
        self.assertEqual(delays["tsv_hops"], 1)
        self.assertGreaterEqual(delays["wire_cycles_unrounded"], 0.0)
        self.assertGreaterEqual(
            delays["maximum_wire_cycles_unrounded"], delays["wire_cycles_unrounded"]
        )
        same_tier = {**layout, "modules": [dict(module) for module in layout["modules"]]}
        next(module for module in same_tier["modules"] if module["kind"] == "l2")["tier"] = 0
        self.assertEqual(derive_layout_delays(same_tier)["tsv_hops"], 0)

    def test_equal_traffic_weights_exactly_reproduce_arithmetic_mean(self):
        per_core = [
            {"core": core, "delay_cycles": value}
            for core, value in enumerate((1.0, 2.0, 3.0, 4.0))
        ]
        self.assertEqual(
            aggregate_wire_cycles(
                per_core, "traffic-weighted",
                {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            ),
            2.5,
        )

    def test_dominant_core_shifts_traffic_weighted_delay(self):
        per_core = [
            {"core": core, "delay_cycles": value}
            for core, value in enumerate((1.0, 2.0, 3.0, 4.0))
        ]
        self.assertAlmostEqual(
            aggregate_wire_cycles(
                per_core, "traffic-weighted",
                {0: 0.7, 1: 0.1, 2: 0.1, 3: 0.1},
            ),
            1.6,
        )

    def test_traffic_weight_validation_rejects_malformed_mappings(self):
        per_core = [
            {"core": core, "delay_cycles": value}
            for core, value in enumerate((1.0, 2.0))
        ]
        cases = {
            "missing": ({0: 1.0}, "keys must exactly match"),
            "extra": ({0: 0.5, 1: 0.5, 2: 0.0}, "keys must exactly match"),
            "negative": ({0: 1.1, 1: -0.1}, "finite and non-negative"),
            "nonfinite": ({0: math.inf, 1: 0.0}, "finite and non-negative"),
            "not normalized": ({0: 0.4, 1: 0.4}, "sum to one"),
        }
        for label, (weights, message) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    aggregate_wire_cycles(per_core, "traffic-weighted", weights)

    def test_zero_traffic_weight_is_valid_when_mapping_is_normalized(self):
        per_core = [
            {"core": 0, "delay_cycles": 1.0},
            {"core": 1, "delay_cycles": 8.0},
        ]
        self.assertEqual(
            aggregate_wire_cycles(
                per_core, "traffic-weighted", {"0": 1.0, "1": 0.0}
            ),
            1.0,
        )

    def test_derive_layout_delays_records_weighted_contributions(self):
        layout = baseline_layout(self.model())
        weights = {0: 0.7, 1: 0.1, 2: 0.1, 3: 0.1}
        delays = derive_layout_delays(
            layout, communication_weights=weights
        )
        expected = sum(
            weights[item["core"]] * item["delay_cycles"]
            for item in delays["per_core"]
        )
        self.assertAlmostEqual(
            delays["traffic_weighted_wire_cycles_unrounded"], expected
        )
        self.assertEqual(
            delays["traffic_weighted_wire_cycles"],
            int(math.floor(expected + 0.5)),
        )
        for item in delays["per_core"]:
            self.assertEqual(item["communication_weight"], weights[item["core"]])
            self.assertAlmostEqual(
                item["weighted_delay_cycles_contribution"],
                weights[item["core"]] * item["delay_cycles"],
            )

    def test_selected_rounded_wire_cycle_uses_requested_aggregation(self):
        delays = {
            "wire_cycles": 2,
            "maximum_wire_cycles": 5,
            "traffic_weighted_wire_cycles": 3,
        }
        self.assertEqual(select_rounded_wire_cycles(delays, "mean"), 2)
        self.assertEqual(select_rounded_wire_cycles(delays, "maximum"), 5)
        self.assertEqual(
            select_rounded_wire_cycles(delays, "traffic-weighted"), 3
        )

    def test_communication_weights_are_read_only_from_available_profile(self):
        model = self.model()
        model["communication_profile"] = {
            "status": "available",
            "per_core": {
                "0": {"normalized_weight": 0.7},
                "1": {"normalized_weight": 0.1},
                "2": {"normalized_weight": 0.1},
                "3": {"normalized_weight": 0.1},
            },
        }
        self.assertEqual(
            communication_weights_from_model(model, required=True),
            {0: 0.7, 1: 0.1, 2: 0.1, 3: 0.1},
        )
        model["communication_profile"]["status"] = "unavailable"
        self.assertIsNone(communication_weights_from_model(model, required=False))
        with self.assertRaisesRegex(ValueError, "communication profile unavailable"):
            communication_weights_from_model(model, required=True)

    def test_optimizer_and_r2_use_the_same_traffic_weighted_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.r2_model()
            weights = {0: 0.7, 1: 0.1, 2: 0.1, 3: 0.1}
            model["communication_profile"] = {
                "status": "available",
                "source_stats": "/tmp/r1/stats.txt",
                "instruction_window_scope": "roi",
                "counter_family": "system.l2.demandAccesses per CPU requestor",
                "per_core": {
                    str(core): {
                        "matched_counters": [
                            f"system.l2.demandAccesses::cpu{core}.data"
                        ],
                        "raw_demand_accesses": weight * 100.0,
                        "normalized_weight": weight,
                    }
                    for core, weight in weights.items()
                },
                "total_demand_accesses": 100.0,
                "missing_cores": [],
                "diagnostics": [],
            }
            modules_path = root / "modules.json"
            write_json(modules_path, model)
            layout_path = root / "layout.json"
            report = optimize(
                modules_path, layout_path, root / "optimizer.json",
                allowed_l2_tiers=[1], require_scipy=False,
                wire_aggregation="traffic-weighted",
            )
            vector = build_vector(
                modules_path, root / "latency.json",
                layout_path=layout_path, wire_aggregation="traffic-weighted",
            )
            optimized_delays = report["observability_diagnostics"][
                "optimized_layout_delays"
            ]
            self.assertEqual(
                vector["components_cycles"]["layout_wire"],
                vector["layout_delays"]["traffic_weighted_wire_cycles"],
            )
            self.assertEqual(
                optimized_delays["traffic_weighted_wire_cycles"],
                vector["layout_delays"]["traffic_weighted_wire_cycles"],
            )
            self.assertAlmostEqual(
                report["selected"]["wire_objective_cycles"],
                report["selected"]["traffic_weighted_wire_cycles"],
            )
            self.assertEqual(
                vector["wire_cycle_aggregation_for_r2"], "traffic-weighted"
            )
            expected_xbar = (
                vector["components_cycles"]["l2_arbitration"]
                + vector["components_cycles"]["tsv"]
                + vector["components_cycles"]["layout_wire"]
            )
            self.assertEqual(
                vector["gem5_overrides"]["xbar_forward_latency"], expected_xbar
            )
            self.assertTrue(any(
                "one scalar shared-L2XBar latency" in assumption
                for assumption in vector["reproduction_assumptions"]
            ))

    def test_traffic_weighted_r2_rejects_missing_layout_and_manual_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.r2_model()
            model["communication_profile"] = {
                "status": "available",
                "per_core": {
                    str(core): {"normalized_weight": 0.25}
                    for core in range(4)
                },
            }
            modules_path = root / "modules.json"
            write_json(modules_path, model)
            with self.assertRaisesRegex(ValueError, "requires a final layout"):
                build_vector(
                    modules_path, root / "missing_layout.json",
                    wire_aggregation="traffic-weighted",
                )

            layout_path = root / "layout.json"
            write_json(layout_path, baseline_layout(model))
            with self.assertRaisesRegex(ValueError, "cannot override"):
                build_vector(
                    modules_path, root / "manual_override.json",
                    wire_cycles=99, layout_path=layout_path,
                    wire_aggregation="traffic-weighted",
                )

    def test_area_quadrature_proxy_is_finite_and_geometry_sensitive(self):
        layout = baseline_layout(self.model())
        center = proxy_temperature(
            layout["modules"], layout["die_width_mm"], 25.0, 5.0,
            0.3, 0.0, 0.9, "center",
        )
        area = proxy_temperature(
            layout["modules"], layout["die_width_mm"], 25.0, 5.0,
            0.3, 0.0, 0.9, "area-quadrature", 2,
        )
        self.assertTrue(math.isfinite(area))
        self.assertNotAlmostEqual(center, area, places=9)

    def test_optimizer_reports_physical_observability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            write_json(root / "modules.json", model)
            report = optimize(
                root / "modules.json", root / "layout.json", root / "report.json",
                allowed_l2_tiers=[1], require_scipy=False,
            )
            diagnostics = report["observability_diagnostics"]
            self.assertAlmostEqual(
                diagnostics["movable_l2_power_fraction"],
                0.5 / model["totals"]["total_power_w"],
            )
            self.assertIn("layout_delays", report["baseline"])
            self.assertIn("mean_wire_cycles_rounded", report["predicted_deltas"])
            self.assertIn("paper_mean_r2_cycle_changed", diagnostics)

    def test_observability_uses_the_candidate_tier_for_bottom_power(self):
        """A tier move must not reuse the fixed-bin bottom-power term."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model()
            for module in model["modules"]:
                if module["kind"] == "core_logic":
                    module["area_mm2"] = 0.1
                elif module["kind"] == "l2":
                    # Leave a legal bottom-tier corner without changing the
                    # power difference this test is about.
                    module["area_mm2"] = 0.0004
            write_json(root / "modules.json", model)
            report = optimize(
                root / "modules.json", root / "layout.json", root / "report.json",
                allowed_l2_tiers=[0, 1], require_scipy=False, beta=0.7,
                proxy_anchor_tmax_c=110.0,
            )

        observations = report["observability_diagnostics"]["sampled_l2_positions"]
        by_tier = {}
        for observation in observations:
            by_tier.setdefault(observation["tier"], []).append(
                observation["bottom_power_w"]
            )
        fixed_bottom_power = sum(
            module["total_power_w"]
            for module in baseline_layout(model)["modules"] if module["tier"] == 0
        )
        self.assertEqual(set(by_tier), {0, 1})
        self.assertTrue(all(value == fixed_bottom_power for value in by_tier[1]))
        self.assertTrue(all(
            value == fixed_bottom_power + 0.5 for value in by_tier[0]
        ))

    def granular_model(self):
        """Eight fixed blocks per core plus shared L2 and NoC (34 modules)."""
        blocks = (
            ("core_ifu", 0.15, 0.12, 0.03),
            ("core_rename", 0.15, 0.12, 0.03),
            ("core_lsu", 0.15, 0.12, 0.03),
            ("core_mmu", 0.10, 0.08, 0.02),
            ("core_exec", 0.20, 0.16, 0.04),
            ("l1i", 0.10, 0.08, 0.02),
            ("l1d", 0.10, 0.08, 0.02),
            ("core_other", 0.05, 0.04, 0.01),
        )
        modules = []
        for core in range(4):
            for kind, area, dynamic, leakage in blocks:
                modules.append({
                    "name": f"core{core}_{kind}",
                    "kind": kind,
                    "core": core,
                    "area_mm2": area,
                    "dynamic_power_w": dynamic,
                    "leakage_power_w": leakage,
                    "total_power_w": dynamic + leakage,
                })
        modules.extend((
            {"name": "shared_l2", "kind": "l2", "area_mm2": 1.0,
             "dynamic_power_w": 0.4, "leakage_power_w": 0.1,
             "total_power_w": 0.5},
            {"name": "noc", "kind": "interconnect", "area_mm2": 0.1,
             "dynamic_power_w": 0.1, "leakage_power_w": 0.01,
             "total_power_w": 0.11},
        ))
        return {"schema_version": 1, "ipc1": 4.0, "gamma": 0.21,
                "modules": modules, "totals": {"total_power_w": 4.61,
                "dynamic_power_w": 3.7, "leakage_power_w": 0.91}}

    def test_granular_proxy_uses_all_34_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.granular_model())
            report = optimize(
                root / "modules.json", root / "layout.json",
                root / "report.json", allowed_l2_tiers=[1],
                require_scipy=False,
            )
            layout = read_json(root / "layout.json")
        proxy = report["thermal_proxy"]
        self.assertEqual(proxy["module_count"], 34)
        self.assertEqual(proxy["fixed_module_count"], 33)
        self.assertEqual(proxy["movable_names"], ["shared_l2"])
        self.assertEqual(len(layout["modules"]), 34)

    def test_optimizer_changes_only_shared_l2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.granular_model())
            report = optimize(
                root / "modules.json", root / "layout.json",
                root / "report.json", allowed_l2_tiers=[1],
                require_scipy=False,
            )
            baseline = {
                module["name"]: module
                for module in baseline_layout(self.granular_model())["modules"]
            }
            selected = {
                module["name"]: module
                for module in read_json(root / "layout.json")["modules"]
            }
        physical = ("tier", "x_mm", "y_mm", "width_mm", "height_mm")
        for name, before in baseline.items():
            if name == "shared_l2":
                continue
            after = selected[name]
            for field in physical:
                self.assertEqual(before[field], after[field], (name, field))
            self.assertEqual(before["area_mm2"], after["area_mm2"])
            self.assertEqual(before["total_power_w"], after["total_power_w"])

    def test_proxy_report_records_fixed_and_movable_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.granular_model())
            report = optimize(
                root / "modules.json", root / "layout.json",
                root / "report.json", allowed_l2_tiers=[1],
                require_scipy=False,
            )
        proxy = report["thermal_proxy"]
        self.assertEqual(proxy["role"], "search-heuristic")
        self.assertEqual(len(proxy["module_names"]), 34)
        self.assertEqual(len(proxy["fixed_names"]), 33)
        self.assertIn("shared_l2", proxy["movable_names"])
        self.assertGreaterEqual(
            proxy["quadrature_sample_pair_count"],
            proxy["module_pair_count"],
        )
        self.assertIn("HotSpot", proxy["warning"])
        self.assertIsNotNone(proxy["spatial_response_range_w"])

    def test_proxy_components_match_scalar_and_count_modules(self):
        layout = baseline_layout(self.granular_model())
        modules = layout["modules"]
        components = proxy_temperature_components(
            modules, layout["die_width_mm"], 25.0, 5.0,
            0.3, 0.1, 0.9, "area-quadrature", 2,
        )
        scalar = proxy_temperature(
            modules, layout["die_width_mm"], 25.0, 5.0,
            0.3, 0.1, 0.9, "area-quadrature", 2,
        )
        self.assertEqual(components["module_count"], 34)
        self.assertEqual(len(components["module_names"]), 34)
        self.assertEqual(components["module_pair_count"], 34 ** 2)
        self.assertEqual(components["quadrature_sample_pair_count"], (34 * 4) ** 2)
        self.assertAlmostEqual(components["proxy_temperature_c"], scalar)

    def test_formal_proxy_rejects_aggregate_core_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            with self.assertRaisesRegex(ValueError, "core_logic"):
                optimize(
                    root / "modules.json", root / "layout.json",
                    root / "report.json", require_granular_cores=True,
                )

    def test_proxy_rejects_degenerate_legal_l2_response(self):
        model = self.granular_model()
        for module in model["modules"]:
            module["dynamic_power_w"] = 0.0
            module["leakage_power_w"] = 0.0
            module["total_power_w"] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", model)
            with self.assertRaisesRegex(ValueError, "degenerate"):
                optimize(
                    root / "modules.json", root / "layout.json",
                    root / "report.json", require_granular_cores=True,
                )

    def test_comparison_layouts_emit_three_recorded_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            report = generate_comparison_layouts(
                root / "modules.json", root / "search", "sa-lambda",
                candidate_grid=5, top_k=3, sa_iterations=20, sa_seed=7,
            )
            self.assertEqual(report["hotspot_candidate_count"], 3)
            self.assertEqual(len(report["search"]), 11)

    @unittest.skipUnless(DEFAULT_HOTSPOT.is_file(), "HotSpot executable unavailable")
    def test_comparison_candidates_run_three_hotspot_solves(self):
        config = {
            "frequency": {"ambient_c": 25.0, "f0_ghz": 2.0,
                          "fmin_ghz": 0.4, "tsafe_c": 95.0},
            "physical": {"grid_size": 4, "utilization": 0.70,
                         "r_convec_k_per_w": 5.0},
            "layout_optimizer": {"alpha": 0.3, "beta": 0.1,
                                 "cross_tier_weight": 0.65},
            "comparison_layouts": {"candidate_grid": 5, "top_k_hotspot": 3,
                                   "sa_iterations": 20, "sa_seed": 7,
                                   "sa_selection_lambda": 0.5},
            "delay": {"wire_rounding": "nearest"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            layout, thermal, selection = evaluate_comparison_candidates(
                root / "modules.json", root / "output", config, "sa-lambda"
            )
            self.assertTrue(layout.is_file())
            self.assertGreater(thermal["tmax_c"], 25.0)
            self.assertEqual(selection["hotspot_solves"], 3)

    @unittest.skipUnless(DEFAULT_HOTSPOT.is_file(), "HotSpot executable unavailable")
    def test_small_real_hotspot_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "modules.json", self.model())
            materialize(root / "modules.json", root / "case", grid_size=4)
            result = run_hotspot(root / "case")
            self.assertGreater(result["tmax_c"], 25.0)
            self.assertTrue(math.isfinite(result["tmax_c"]))

            validation = validate_case(
                root / "case", root / "modules.json", root / "validation.json",
                [0.5, 1.0, 2.0], validate_solution=False,
                scaling_mode="separated-dynamic-leakage",
            )
            self.assertEqual(validation["scaling_mode"], "separated-dynamic-leakage")
            self.assertTrue(math.isfinite(
                validation["max_abs_uniform_gamma_comparison_error_c"]
            ))
            self.assertLess(
                validation["frequencies"][0]["trace_sums_w"]["composed"],
                validation["frequencies"][0]["trace_sums_w"]["total_at_f0"],
            )

class OperationalProfileTests(unittest.TestCase):
    @staticmethod
    def measured_report(target_rank: float = 0.5) -> dict:
        return {
            "fit": {"parameters": {
                "alpha": 1.5643788695171585,
                "beta": 0.0,
                "cross_tier_weight": 0.995,
            }},
            "evaluations": {
                "cross_validated_training_fit": {"validation": {
                    "rmse_c": 0.8048080400435677,
                    "spatial_centered_rmse_c": 0.12900228266536495,
                    "spatial_spearman": 0.6571428571428573,
                }},
                "defaults": {"validation": {
                    "rmse_c": 13.455896637598286,
                    "spatial_centered_rmse_c": 0.38150141151565364,
                    "spatial_spearman": 0.8857142857142858,
                }},
            },
            "external_target_grid_validation": {
                "fitted": {
                    "rmse_c": 3.047894945341817,
                    "spatial_centered_rmse_c": 0.3118797983194971,
                    "spatial_spearman": target_rank,
                },
                "defaults": {
                    "rmse_c": 12.748167980320552,
                    "spatial_centered_rmse_c": 0.39245312881639577,
                    "spatial_spearman": 0.5,
                },
            },
            "strict_p1": {"beta_status": "fixed_unidentifiable_under_p1"},
            "leave_one_model_out": {
                "fft": {"held_out_metrics": {"spatial_spearman": 0.6428571428571429}},
                "matmul": {"held_out_metrics": {"spatial_spearman": 0.5}},
                "stencil": {"held_out_metrics": {"spatial_spearman": 0.28571428571428575}},
            },
        }

    @staticmethod
    def operational_config() -> Path:
        return Path(
            "configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json"
        )

    def test_operational_evaluator_accepts_measured_boundary_profile(self):
        """A rank exactly 0.5 remains usable but explicitly non-formal."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "proxy.json"
            output_path = root / "operational.json"
            report = self.measured_report()
            write_json(report_path, report)

            result = evaluate_operational_proxy(
                report_path, self.operational_config(), output_path
            )

            self.assertTrue(result["recommendation"]["accepted"])
            self.assertEqual(result["mode"], "operational")
            self.assertTrue(result["non_formal"])
            self.assertEqual(
                result["recommendation"]["action"],
                "operational use permitted; non-formal and not promotable",
            )
            self.assertEqual(result["source"]["path"], str(report_path.resolve()))
            self.assertRegex(result["source"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                result["diagnostics"]["leave_one_workload_out"]["stencil"],
                0.28571428571428575,
            )
            self.assertNotIn("leave_one_workload_out", result["checks"])
            self.assertEqual(read_json(report_path), report)
            self.assertEqual(read_json(output_path), result)

    def test_operational_evaluator_rejects_low_target_rank_or_parameter_mismatch(self):
        """A sub-threshold target rank rejects, and a changed config cannot be relabeled."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "proxy.json"
            output_path = root / "operational.json"
            write_json(report_path, self.measured_report(target_rank=0.499))

            result = evaluate_operational_proxy(
                report_path, self.operational_config(), output_path
            )
            self.assertFalse(result["recommendation"]["accepted"])
            self.assertFalse(result["checks"]["target_spatial_rank_at_least_0p5"])

            config = read_json(self.operational_config())
            config["layout_optimizer"]["alpha"] = 1.5643788695171585 + 1e-9
            mismatch_config = root / "mismatch.json"
            write_json(mismatch_config, config)
            with self.assertRaisesRegex(ValueError, "alpha.*does not match"):
                evaluate_operational_proxy(report_path, mismatch_config, output_path)

    def test_operational_evaluator_rejects_output_aliases_without_modifying_inputs(self):
        """The separate decision artifact cannot overwrite either immutable input."""
        for alias in ("proxy", "config"):
            with self.subTest(output=alias):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    report_path = root / "proxy.json"
                    config_path = root / "operational.json"
                    write_json(report_path, self.measured_report())
                    write_json(config_path, read_json(self.operational_config()))
                    output_path = report_path if alias == "proxy" else config_path
                    original = output_path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "output.*input"):
                        evaluate_operational_proxy(report_path, config_path, output_path)
                    self.assertEqual(output_path.read_bytes(), original)

    def test_operational_evaluator_rejects_self_consistent_noncanonical_parameters(self):
        """Matching changed values cannot redefine the approved operational profile."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.measured_report()
            report["fit"]["parameters"]["alpha"] = 1.0
            config = read_json(self.operational_config())
            config["layout_optimizer"]["alpha"] = 1.0
            report_path = root / "proxy.json"
            config_path = root / "operational.json"
            write_json(report_path, report)
            write_json(config_path, config)

            with self.assertRaisesRegex(ValueError, "alpha.*approved"):
                evaluate_operational_proxy(report_path, config_path, root / "result.json")

    def test_operational_evaluator_rejects_mutated_operational_policy(self):
        """The approved rank floors and diagnostic-only LOO status are immutable."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "proxy.json"
            write_json(report_path, self.measured_report())
            for field, value in (
                    ("minimum_validation_spatial_spearman", 0.6),
                    ("minimum_external_target_spatial_spearman", 0.6),
                    ("leave_one_workload_out", "release_gate")):
                with self.subTest(field=field):
                    config = read_json(self.operational_config())
                    config["operational_validation"][field] = value
                    config_path = root / f"{field}.json"
                    write_json(config_path, config)
                    with self.assertRaisesRegex(ValueError, "operational_validation.*approved"):
                        evaluate_operational_proxy(
                            report_path, config_path, root / f"{field}-result.json"
                        )

    def test_operational_config_cannot_be_formally_promoted(self):
        """Accepted-looking synthetic reports cannot turn an operational config formal."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = {
                "recommendation": {"accepted": True},
                "strict_p1": {"beta_status": "fixed_unidentifiable_under_p1"},
                "fit": {"parameters": {"alpha": 0.31, "cross_tier_weight": 0.94}},
            }
            wire = {"recommendation": {"accepted": True}, "selected_lambda_wire": 0.125}
            frequency = {"recommendation": {"accepted": True}}
            write_json(root / "proxy.json", proxy)
            write_json(root / "wire.json", wire)
            write_json(root / "frequency.json", frequency)

            with self.assertRaisesRegex(ValueError, "strict_p1.*exactly true"):
                promote(
                    root / "proxy.json", root / "wire.json", root / "frequency.json",
                    self.operational_config(), root / "formal.json",
                )

    def test_operational_launcher_separates_pilot_from_full_r1_requirements(self):
        """The launcher may lift R2, but never starts R1 or formal promotion."""
        script = Path("scripts/run_operational_raw_power_p1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('case "$MODE" in', script)
        self.assertIn("--run-r2", script)
        self.assertIn("all 100 R1 status files must be success", script)
        self.assertNotIn("run_r1_sweep.py", script)
        self.assertNotIn("promote_validated_config", script)

    def test_operational_launcher_uses_nonexistent_roots_and_named_lock(self):
        """New output roots and distinct operational tmux/flock sessions are required."""
        script = Path("scripts/run_operational_raw_power_p1.sh").read_text(
            encoding="utf-8"
        )
        documentation = Path("docs/formal_reproduction_zh.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('test ! -e "$ROOT"', script)
        self.assertIn('runs/operational_raw_power_p1/$MODE', script)
        self.assertIn('"$ROOT/fixed-bin"', script)
        self.assertIn('"$ROOT/clip3d"', script)
        self.assertIn("clip_operational_pilot", documentation)
        self.assertIn("/tmp/clip-operational-pilot.lock", documentation)
        self.assertIn("clip_operational_full", documentation)
        self.assertIn("/tmp/clip-operational-full.lock", documentation)


class FormalGuardTests(unittest.TestCase):
    @staticmethod
    def write_complete_r1_tree(root: Path, instruction_window_scope: str = "roi") -> None:
        for workload in ("fft", "matmul", "stencil", "stream"):
            point = root / workload / "l1d_32kB" / "l2_512kB"
            point.mkdir(parents=True)
            write_json(point / "r1_metadata.json", {
                "workload": workload,
                "l1d_size": "32kB",
                "l2_size": "512kB",
                "instruction_window_scope": instruction_window_scope,
            })
            (point / "stats.txt").write_text("sim_ticks 100\n", encoding="utf-8")
            write_json(point / "status.json", {"state": "success"})

    def test_raw_power_p1_candidate_has_only_top_tier_l2(self):
        config = read_json(Path(
            "configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json"
        ))
        self.assertEqual(config["layout_optimizer"]["allowed_l2_tiers"], [1])
        self.assertEqual(config["layout_optimizer"]["validation_policy"], "paper-single")
        self.assertEqual(config["layout_optimizer"]["beta"], 0.0)
        self.assertFalse(config["formal_validation"]["accepted"])
        validate_config(config, "clip3d")

    def test_nonformal_config_accepts_traffic_weighted_extension(self):
        config = read_json(Path(
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
        ))
        config["delay"]["wire_aggregation"] = "traffic-weighted"
        validate_config(config, "clip3d")

    def test_accepted_formal_config_rejects_traffic_weighted_extension_first(self):
        config = read_json(Path(
            "configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json"
        ))
        config["formal_validation"]["accepted"] = True
        config["delay"]["wire_aggregation"] = "traffic-weighted"
        with self.assertRaisesRegex(ValueError, "non-formal research extension"):
            validate_config(config, "clip3d")

    def test_default_mean_aggregation_remains_valid(self):
        config = read_json(Path(
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
        ))
        config["delay"].pop("wire_aggregation", None)
        validate_config(config, "clip3d")

    def test_clip3d_optimizer_rejects_conservative_maximum_mode(self):
        config = read_json(Path(
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
        ))
        config["delay"]["wire_aggregation"] = "maximum"
        with self.assertRaisesRegex(ValueError, "maximum.*R2 sensitivity"):
            validate_config(config, "clip3d")

    def test_comparison_layouts_reject_traffic_weighted_selection(self):
        config = read_json(Path(
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
        ))
        config["delay"]["wire_aggregation"] = "traffic-weighted"
        for method in ("cool3d-standard", "sa-lambda"):
            with self.subTest(method=method), self.assertRaisesRegex(
                    ValueError, "comparison layout methods"):
                validate_config(config, method)

    def test_traffic_weighted_exploratory_config_is_non_formal(self):
        config = read_json(Path(
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_"
            "lambda0020119_traffic_weighted_exploratory.json"
        ))
        self.assertEqual(
            config["delay"]["wire_aggregation"], "traffic-weighted"
        )
        self.assertTrue(config["experiment_classification"]["non_formal"])
        self.assertFalse(config["experiment_classification"]["paper_equivalent"])
        self.assertFalse(config["formal_validation"]["accepted"])
        self.assertGreater(config["layout_optimizer"]["lambda_wire"], 0.0)
        self.assertEqual(config["layout_optimizer"]["allowed_l2_tiers"], [1])
        validate_config(config, "clip3d")

    def test_promotion_rejects_failed_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "proxy.json", {"recommendation": {"accepted": False}})
            write_json(root / "wire.json", {"recommendation": {"accepted": True}})
            write_json(root / "frequency.json", {"recommendation": {"accepted": True}})
            with self.assertRaisesRegex(ValueError, "proxy"):
                promote(
                    root / "proxy.json", root / "wire.json", root / "frequency.json",
                    Path("configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json"),
                    root / "formal.json",
                )

    def test_prepare_validation_manifest_requires_the_four_named_r1_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "fft"):
                prepare(root, root / "input_manifest.json")

    def test_accepted_formal_config_rejects_manual_parameters_without_artifact_provenance(self):
        config = read_json(Path(
            "configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json"
        ))
        config["layout_optimizer"].update({
            "alpha": 0.31,
            "cross_tier_weight": 0.94,
            "lambda_wire": 0.125,
            "parameter_provenance": {
                "alpha": "manually chosen",
                "beta": "fixed_unidentifiable_under_p1",
                "cross_tier_weight": "manually chosen",
                "lambda_wire": "manually chosen",
            },
        })
        config["formal_validation"]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "accepted strict-P1.*artifacts"):
            validate_config(config, "clip3d")

    def test_promotion_emits_report_derived_formal_config_that_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = {
                "recommendation": {"accepted": True},
                "strict_p1": {"beta_status": "fixed_unidentifiable_under_p1"},
                "fit": {"parameters": {"alpha": 0.31, "cross_tier_weight": 0.94}},
            }
            wire = {
                "recommendation": {"accepted": True},
                "selected_lambda_wire": 0.125,
            }
            frequency = {
                "recommendation": {"accepted": True},
                "selected_frequency_ghz": 2.5,
            }
            write_json(root / "proxy.json", proxy)
            write_json(root / "wire.json", wire)
            write_json(root / "frequency.json", frequency)
            output = root / "formal.json"
            formal = promote(
                root / "proxy.json", root / "wire.json", root / "frequency.json",
                Path("configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json"),
                output,
            )

            optimizer = formal["layout_optimizer"]
            self.assertEqual(optimizer["alpha"], 0.31)
            self.assertEqual(optimizer["cross_tier_weight"], 0.94)
            self.assertEqual(optimizer["lambda_wire"], 0.125)
            self.assertEqual(optimizer["beta"], 0.0)
            self.assertEqual(optimizer["parameter_provenance"]["alpha"], {
                "artifact": "proxy_report",
                "field": "fit.parameters.alpha",
                "value": 0.31,
            })
            self.assertEqual(optimizer["parameter_provenance"]["cross_tier_weight"], {
                "artifact": "proxy_report",
                "field": "fit.parameters.cross_tier_weight",
                "value": 0.94,
            })
            self.assertEqual(optimizer["parameter_provenance"]["lambda_wire"], {
                "artifact": "wire_summary",
                "field": "selected_lambda_wire",
                "value": 0.125,
            })
            self.assertEqual(optimizer["parameter_provenance"]["beta"], {
                "source": "fixed_unidentifiable_under_p1",
                "value": 0.0,
            })
            for artifact in formal["formal_validation"]["artifacts"].values():
                self.assertTrue(Path(artifact["path"]).is_absolute())
                self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
            validate_config(read_json(output), "clip3d")

    def test_prepare_rejects_noncanonical_cache_sizes_before_point_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "input_manifest.json"
            self.write_complete_r1_tree(root)
            with self.assertRaisesRegex(ValueError, "l1d_size=32kB and l2_size=512kB"):
                prepare(root, output, l1d_size="64kB")
            self.assertFalse(output.exists())

    def test_prepare_rejects_missing_instruction_window_scope_without_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "input_manifest.json"
            self.write_complete_r1_tree(root, instruction_window_scope=" ")
            with self.assertRaisesRegex(ValueError, "instruction_window_scope"):
                prepare(root, output)
            self.assertFalse(output.exists())

    def test_clip3d_real_hotspot_guard_rejects_lower_thermal_bips(self):
        fixed = {"policy": "fixed-bin", "bips1_thermal": 3.45,
                 "wire_cycles": 1, "tmax_c": 124.39}
        proposed = {"policy": "optimized", "bips1_thermal": 3.43,
                    "wire_cycles": 1, "tmax_c": 124.54}
        selected, reason = select_clip3d_candidate([fixed, proposed])
        self.assertEqual(selected["policy"], "fixed-bin")
        self.assertIn("baseline guard", reason)

    def test_clip3d_real_hotspot_guard_uses_discrete_wire_tie_break(self):
        fixed = {"policy": "fixed-bin", "bips1_thermal": 3.45,
                 "wire_cycles": 2, "tmax_c": 120.0}
        proposed = {"policy": "optimized", "bips1_thermal": 3.45,
                    "wire_cycles": 1, "tmax_c": 121.0}
        selected, reason = select_clip3d_candidate([proposed, fixed])
        self.assertEqual(selected["policy"], "optimized")
        self.assertIn("wire latency", reason)

    def test_lifting_resume_rejects_stale_config(self):
        from tests.test_mcpat_native_cache import (
            write_native_physical_fixture,
            write_native_r1_fixture,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = {"physical": {"r_convec_k_per_w": 5.0}, "mcpat": {"temperature_k": 370}}
            new = {"physical": {"r_convec_k_per_w": 5.0}, "mcpat": {"temperature_k": 320}}
            write_json(root / "run_config.json", {"config": old})
            for artifact in lifting_required_artifacts:
                path = root / artifact
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
            binary = root / "mcpat/test-binary"
            binary.write_bytes(b"strict McPAT test binary\n")
            r1 = root / "source-r1"
            metadata = {
                "workload": "fft", "num_cores": 4, "cpu_clock": "2GHz",
                "l1i_size": "32kB", "l1d_size": "32kB",
                "l2_size": "512kB", "l1_associativity": 2,
                "l2_associativity": 8, "cache_line_bytes": 64,
            }
            write_native_r1_fixture(r1, metadata)
            mcpat, _model = write_native_physical_fixture(
                root, r1, binary, metadata,
            )
            mcpat_output = root / "mcpat/mcpat.out"
            write_json(root / "pipeline_summary.json", {
                "r1": str(r1.resolve()),
                "layout_method": "fixed-bin", "cooling": {"r_convec_k_per_w": 5.0},
                "ipc2": 1.0, "bips2": 1.0,
                "stage_seconds": {},
                "cache_authority": "McPAT 1.3 embedded CACTI-P",
                "mcpat_provenance": mcpat["provenance"],
                "artifacts": {
                    "mcpat_json": str((root / "mcpat/mcpat.json").resolve()),
                    "mcpat_output": str(mcpat_output.resolve()),
                    "mcpat_binary": str(binary.resolve()),
                },
                "artifact_sha256": {
                    "mcpat_json": sha256_file(root / "mcpat/mcpat.json"),
                    "mcpat_output": mcpat["provenance"]["hashes"]["output_sha256"],
                    "mcpat_binary": mcpat["provenance"]["hashes"]["binary_sha256"],
                },
            })
            self.assertFalse(lifting_completed(root, new, "fixed-bin", True))
            self.assertTrue(lifting_completed(root, old, "fixed-bin", True))

    def test_mismatched_optimizer_and_hotspot_cooling_is_rejected(self):
        config = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 3.5},
            "layout_optimizer": {"r_convec_k_per_w": 5.0},
        }
        with self.assertRaises(ValueError):
            validate_config(config, "clip3d")
        validate_config(config, "fixed-bin")

    def test_negative_clip3d_guard_tolerance_is_rejected(self):
        config = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {"r_convec_k_per_w": 5.0,
                                 "baseline_guard_bips_tolerance": -1.0},
        }
        with self.assertRaises(ValueError):
            validate_config(config, "clip3d")

    def test_invalid_paper_mode_controls_are_rejected(self):
        config = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {
                "r_convec_k_per_w": 5.0,
                "validation_policy": "not-a-policy",
            },
        }
        with self.assertRaises(ValueError):
            validate_config(config, "clip3d")
        config["layout_optimizer"]["validation_policy"] = "paper-single"
        config["layout_optimizer"]["allowed_l2_tiers"] = [2]
        with self.assertRaises(ValueError):
            validate_config(config, "clip3d")
        config["layout_optimizer"]["allowed_l2_tiers"] = [1]
        config["layout_optimizer"]["wire_objective"] = "invented"
        with self.assertRaises(ValueError):
            validate_config(config, "clip3d")

    def test_strict_p1_rejects_additional_hotspot_proxy_anchor(self):
        config = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {
                "r_convec_k_per_w": 5.0,
                "allowed_l2_tiers": [1],
                "validation_policy": "paper-single",
                "beta": 0.0,
                "thermal_anchor_policy": "fixed-bin-hotspot",
            },
            "formal_validation": {"strict_p1": True, "accepted": False},
        }
        with self.assertRaisesRegex(ValueError, "strict P1 forbids"):
            validate_config(config, "clip3d")

    def test_discrete_partition_config_requires_valid_grid_and_baseline(self):
        config = {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {
                "r_convec_k_per_w": 5.0,
                "wire_objective": "discrete-partition",
                "partition_grid_steps": 41,
                "include_fixed_baseline": True,
            },
        }
        validate_config(config, "clip3d")

        config["layout_optimizer"]["partition_grid_steps"] = 40
        with self.assertRaisesRegex(ValueError, "odd integer"):
            validate_config(config, "clip3d")
        config["layout_optimizer"]["partition_grid_steps"] = 41
        config["layout_optimizer"]["include_fixed_baseline"] = False
        with self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
            validate_config(config, "clip3d")

    def test_formal_summary_requires_real_r2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            point = {
                "workload": "matmul", "l1d_size": "32kB", "l2_size": "512kB",
                "layout_method": "fixed-bin", "cooling": {"r_convec_k_per_w": 3.5},
                "ipc1": 2.0, "tmax_c": 80.0, "sustainable_frequency_ghz": 2.0,
                "bips1_thermal": 4.0, "ipc2": None, "bips2": None,
                "r2_critical_path_cycles": 10, "total_pipeline_seconds": 1.0,
            }
            write_json(root / "point/pipeline_summary.json", point)
            with self.assertRaises(ValueError):
                summarize(root, root / "strict.csv", root / "strict.json")
            result = summarize(
                root, root / "proxy.csv", root / "proxy.json",
                allow_proxy=True, expected_points=1,
            )
            self.assertFalse(result["r2_complete"])


class ThermalModeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_json(self.root / "config.json", {
            "schema_version": 1,
            "physical": {"r_convec_k_per_w": 5.0},
            "layout_optimizer": {"r_convec_k_per_w": 5.0},
            "delay": {"wire_aggregation": "mean"},
        })

    def invoke_cli(self, *arguments: str) -> str:
        argv = [
            "run_lifting_pipeline",
            "--r1-dir", str(self.root / "r1"),
            "--output-dir", str(self.root / "output"),
            "--config", str(self.root / "config.json"),
            *arguments,
        ]
        stdout = StringIO()
        with patch("sys.argv", argv), redirect_stdout(stdout):
            run_lifting_pipeline_main()
        return stdout.getvalue()

    @staticmethod
    def steady_summary() -> dict:
        return {
            "layout_method": "fixed-bin",
            "tmax_c": 75.0,
            "sustainable_frequency_ghz": 2.0,
        }

    @contextmanager
    def patch_rom_pipeline(self, **patch_kwargs):
        """Supply the optional ROM import only for CLI dispatch tests."""
        module_name = "workflow.transient.rom.run_pipeline"
        import workflow.transient.rom as rom_package

        stub = ModuleType(module_name)
        stub.run_transient_rom_pipeline = lambda *_args, **_kwargs: None
        with patch.dict(sys.modules, {module_name: stub}), patch.object(
            rom_package, "run_pipeline", stub, create=True,
        ), patch.object(
            stub, "run_transient_rom_pipeline", **patch_kwargs,
        ) as pipeline:
            yield pipeline

    def test_steady_mode_calls_legacy_pipeline_without_rom_dispatch(self):
        # Break caught: making ROM the default would alter every historical
        # steady invocation and import its optional NumPy/SciPy dependency.
        module_name = "workflow.transient.rom.run_pipeline"
        previous = sys.modules.pop(module_name, None)

        def restore_module():
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

        self.addCleanup(restore_module)
        with patch(
            "workflow.run_lifting_pipeline.run_pipeline",
            return_value=self.steady_summary(),
        ) as legacy:
            self.invoke_cli("--thermal-mode", "steady")

        legacy.assert_called_once()
        self.assertNotIn(module_name, sys.modules)
        self.assertEqual(legacy.call_args.kwargs["layout_method"], "fixed-bin")
        self.assertFalse(legacy.call_args.kwargs["execute_r2"])

    def test_rom_mode_runs_fixed_preflight_then_rom_with_r2_forwarded(self):
        # Break caught: forwarding --run-r2 to the steady pilot either runs R2
        # twice or derives it from the wrong (fixed-bin) layout.
        rom_summary = {
            "f_sus_trans_rom_pred_ghz": 1.9,
            "f_sus_trans_hotspot_ghz": 1.8,
            "bips1_trans_rom_pred": 3.8,
            "bips2_trans": 2.7,
        }
        with patch(
            "workflow.run_lifting_pipeline.run_pipeline",
            return_value=self.steady_summary(),
        ) as steady, self.patch_rom_pipeline(
            return_value=rom_summary,
        ) as rom:
            self.invoke_cli(
                "--thermal-mode", "transient-rom", "--run-r2", "--rerun-r2"
            )

        self.assertEqual(steady.call_args.kwargs["layout_method"], "fixed-bin")
        self.assertFalse(steady.call_args.kwargs["execute_r2"])
        self.assertTrue(rom.call_args.kwargs["execute_r2"])
        self.assertTrue(rom.call_args.kwargs["rerun_r2"])
        self.assertEqual(
            rom.call_args.kwargs["steady_preflight_dir"],
            (self.root / "output/steady_preflight").resolve(),
        )
        self.assertEqual(
            rom.call_args.kwargs["output_dir"],
            (self.root / "output/transient_rom").resolve(),
        )

    def test_rom_mode_rejects_legacy_transient_flag(self):
        # Break caught: combining both meanings of transient thermal analysis
        # produces overlapping output trees and ambiguous summaries.
        with self.assertRaisesRegex(SystemExit, "cannot combine"):
            self.invoke_cli(
                "--thermal-mode", "transient-rom", "--transient", "true"
            )

    def test_rom_mode_rejects_invalid_discrete_controls_before_dispatch(self):
        # Break caught: reusing an existing preflight must not bypass the common
        # discrete-partition gate and start expensive ROM work with no baseline.
        config = read_json(self.root / "config.json")
        config["layout_optimizer"].update({
            "wire_objective": "discrete-partition",
            "partition_grid_steps": 41,
            "include_fixed_baseline": False,
        })
        write_json(self.root / "config.json", config)
        write_json(
            self.root / "output/steady_preflight/pipeline_summary.json", {}
        )

        with patch(
            "workflow.run_lifting_pipeline.run_pipeline"
        ) as steady, self.patch_rom_pipeline(
            return_value={"f_sus_trans_hotspot_ghz": None},
        ) as rom, self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
            self.invoke_cli("--thermal-mode", "transient-rom")

        steady.assert_not_called()
        rom.assert_not_called()

    def test_rom_mode_renders_paired_clip3d_validated_frequency(self):
        # Break caught: the discrete paired summary intentionally removes the
        # legacy flat frequency field; the CLI must read the CLIP branch.
        config = read_json(self.root / "config.json")
        config["layout_optimizer"].update({
            "wire_objective": "discrete-partition",
            "partition_grid_steps": 41,
            "include_fixed_baseline": True,
        })
        write_json(self.root / "config.json", config)
        write_json(
            self.root / "output/steady_preflight/pipeline_summary.json", {}
        )
        paired_summary = {
            "branches": {
                "clip3d": {
                    "validated_f_sus_trans_hotspot_ghz": 1.75,
                }
            }
        }

        with patch(
            "workflow.run_lifting_pipeline.run_pipeline"
        ) as steady, self.patch_rom_pipeline(
            return_value=paired_summary,
        ) as rom:
            stdout = self.invoke_cli("--thermal-mode", "transient-rom")

        steady.assert_not_called()
        rom.assert_called_once()
        self.assertIn("f_sus_hotspot=1.750000 GHz", stdout)

    def test_rom_mode_rejects_arbitrary_output_directory_option(self):
        # Break caught: allowing callers to relocate ROM outputs breaks the
        # fixed OUTPUT/transient_rom sibling identity required for auditing.
        with patch(
            "workflow.run_lifting_pipeline.run_pipeline"
        ) as steady, self.patch_rom_pipeline(
        ) as rom:
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                self.invoke_cli(
                    "--thermal-mode", "transient-rom",
                    "--transient-rom-dir", str(self.root / "elsewhere"),
                )

        self.assertEqual(error.exception.code, 2)
        self.assertIn("unrecognized arguments: --transient-rom-dir", stderr.getvalue())
        steady.assert_not_called()
        rom.assert_not_called()


if __name__ == "__main__":
    unittest.main()
