from __future__ import annotations

import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from workflow.cacti.characterize_cache import PAPER_TABLE_II, parse_cacti_output
from workflow.analysis.summarize_sweep import summarize
from workflow.analysis.prepare_raw_power_validation import prepare
from workflow.analysis.promote_validated_config import promote
from workflow.common import read_json, write_json
from workflow.floorplan.comparison_layouts import generate as generate_comparison_layouts
from workflow.floorplan.build_module_model import apply_physical_areas
from workflow.floorplan.generate_hotspot_inputs import baseline_layout, grid_power, materialize
from workflow.floorplan.layout_metrics import derive_layout_delays
from workflow.floorplan.optimize_layout import optimize, proxy_temperature
from workflow.mcpat.parse_mcpat import (
    apply_power_calibration,
    parse_mcpat_text,
    resolve_power_calibration,
)
from workflow.r2.calibrate_lambda_wire import (
    calibrate as calibrate_lambda_wire,
    calibrate_series as calibrate_lambda_wire_series,
    main as calibrate_lambda_wire_main,
)
from workflow.r2.run_wire_sensitivity import (
    build_sensitivity_vectors,
    summarize_workloads,
)
from workflow.run_lifting_pipeline import (
    evaluate_comparison_candidates, select_clip3d_candidate, validate_config,
)
from workflow.run_lifting_sweep import completed as lifting_completed
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot
from workflow.thermal.calibrate_proxy import (
    calibrate, candidate_layouts, parse_external_case, proxy_acceptance_checks,
    sample_split,
)
from workflow.thermal.sustainable_frequency import closed_form_frequency
from workflow.thermal.run_anchor_validation import run_manifest
from workflow.thermal.validate_frequency import (
    compose_separated_ptrace,
    read_ptrace,
    validate_case,
)


def metric_lines(area, dynamic, sub, gate, indent="  "):
    return (f"{indent}Area = {area} mm^2\n{indent}Runtime Dynamic = {dynamic} W\n"
            f"{indent}Subthreshold Leakage = {sub} W\n{indent}Gate Leakage = {gate} W\n")


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

    def test_frequency_validation_defaults_to_separated_hotspot_trace(self):
        """Formal frequency validation writes and reports the per-cell raw-power trace."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            write_json(root / "modules.json", {"gamma": 0.2})
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
            self.assertIn("max_abs_uniform_gamma_comparison_error_c", result)
            self.assertNotIn("max_abs_linear_error_c", result)
            self.assertEqual(run["trace_sums_w"]["total_at_f0"], 20.0)
            self.assertEqual(
                run["uniform_gamma_comparison"]["scaling_mode"], "paper-uniform-gamma"
            )
            self.assertFalse(result["recommendation"]["accepted"])

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
                    "solution_validation": {"accepted": True, "safe_error_c": 0.5},
                    "recommendation": {"accepted": True},
                },
                {
                    "frequencies": [{}],
                    "max_abs_uniform_gamma_comparison_error_c": 0.75,
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
            self.assertEqual(summary["hotspot_run_count"], 4)
            self.assertEqual(summary["max_abs_uniform_gamma_comparison_error_c"], 0.75)
            self.assertFalse(summary["recommendation"]["accepted"])

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
                "l1i_cacti": 2, "l1d_cacti": 3, "l2_cacti": 4,
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

        vectors = build_sensitivity_vectors(base, [0, 1, 2])
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
    def test_workload_power_calibration_is_explicitly_resolved(self):
        config = {"power_calibration": {
            "dynamic_scale": 1.1,
            "leakage_scale": 1.2,
            "by_workload": {
                "fft": {"dynamic_scale": 0.9, "leakage_scale": 1.05}
            },
        }}
        fft = resolve_power_calibration(config, "fft")
        matmul = resolve_power_calibration(config, "matmul")
        self.assertEqual(fft["dynamic_scale"], 0.9)
        self.assertTrue(fft["selection"]["used_workload_override"])
        self.assertEqual(matmul["dynamic_scale"], 1.1)
        self.assertFalse(matmul["selection"]["used_workload_override"])

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

        apply_power_calibration(result, dynamic_scale=2.0, leakage_scale=3.0,
                                provenance={"kind": "unit test"})
        self.assertAlmostEqual(result["modules"][0]["dynamic_power_w"], 5.0)
        self.assertAlmostEqual(result["modules"][0]["leakage_power_w"], 1.5)
        self.assertAlmostEqual(result["modules"][0]["raw_power"]["total_power_w"], 3.0)

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

    def test_paper_table_ii_anchor(self):
        self.assertEqual(PAPER_TABLE_II[("l1d", 64 * 1024)]["area_mm2"], 1.16)
        self.assertEqual(PAPER_TABLE_II[("l2", 1024 * 1024)]["access_time_ns"], 1.984)

    def test_module_model_consumes_cacti_cache_geometry(self):
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
            {"level": "l1d", "size": "32kB", "size_bytes": 32 * 1024,
             "area_mm2": 0.74, "width_mm": 1.0, "height_mm": 0.74,
             "value_source": "paper Table II"},
            {"level": "l1d", "size": "64kB", "size_bytes": 64 * 1024,
             "area_mm2": 1.16, "width_mm": 1.45, "height_mm": 0.8,
             "value_source": "paper Table II"},
            {"level": "l2", "size": "512kB", "size_bytes": 512 * 1024,
             "area_mm2": 10.01, "width_mm": 5.0, "height_mm": 2.002,
             "value_source": "paper Table II"},
        ]}
        physical = apply_physical_areas(modules, metadata, cacti, 2.0)
        by_kind = {module["kind"]: module for module in physical}
        self.assertAlmostEqual(by_kind["core_logic"]["area_mm2"], 20.0)
        self.assertEqual(by_kind["core_logic"]["area_source"], "McPAT")
        self.assertAlmostEqual(by_kind["l1d"]["area_mm2"], 2.32)
        self.assertAlmostEqual(by_kind["l1d"]["raw_area_mm2"], 30.0)
        self.assertEqual(by_kind["l1d"]["area_source"], "paper Table II")
        self.assertAlmostEqual(by_kind["l2"]["preferred_width_mm"], 5.0 * math.sqrt(2.0))


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

    def test_exact_power_conservation(self):
        gridded = grid_power(baseline_layout(self.model()), 8)
        for tier in gridded["power_conservation"]:
            for field in ("dynamic_power_w", "leakage_power_w", "total_power_w"):
                self.assertLess(abs(tier[field]["residual"]), 1e-10)

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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "pipeline_summary.json", {
                "layout_method": "fixed-bin", "cooling": {"r_convec_k_per_w": 5.0},
                "ipc2": 1.0, "bips2": 1.0,
            })
            old = {"physical": {"r_convec_k_per_w": 5.0}, "mcpat": {"temperature_k": 370}}
            new = {"physical": {"r_convec_k_per_w": 5.0}, "mcpat": {"temperature_k": 320}}
            write_json(root / "run_config.json", {"config": old})
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


if __name__ == "__main__":
    unittest.main()
