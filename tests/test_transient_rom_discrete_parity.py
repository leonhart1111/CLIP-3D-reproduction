from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import tests.test_transient_rom as transient_tests
from workflow.common import read_json, sha256_file, write_json
from workflow.floorplan.discrete_partition import search_discrete_partitions
from workflow.run_lifting_pipeline import validate_config
from workflow.transient.rom.calibration_design import build_design
from workflow.transient.rom.contracts import parse_settings
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

    def materialized_branch(self, root: Path, name: str,
                            ipc2: float, frequency: float) -> dict:
        branch_dir = root / name
        layout = branch_dir / "layout.json"
        hotspot = branch_dir / "transient_sustainable_frequency.json"
        latency = branch_dir / "r2_latency.json"
        r2_result = branch_dir / "gem5_r2/r2_result.json"
        write_json(layout, {"branch": name})
        write_json(hotspot, {"f_sus_trans_ghz": frequency})
        write_json(latency, {"branch": name, "layout_wire": 7})
        write_json(r2_result, {"ipc2": ipc2})
        branch = self.branch(name, ipc2, frequency)
        branch.update({
            "r2_executed": True,
            "failure": None,
            "validation_classification": "validated",
            "artifacts": {
                "layout": str(layout.resolve()),
                "layout_sha256": sha256_file(layout),
                "hotspot": str(hotspot.resolve()),
                "hotspot_sha256": sha256_file(hotspot),
                "r2_latency": str(latency.resolve()),
                "r2_latency_sha256": sha256_file(latency),
                "r2_result": str(r2_result.resolve()),
                "r2_result_sha256": sha256_file(r2_result),
            },
        })
        return branch

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

            fixed = self.materialized_branch(
                output, "fixed-bin", 1.4, 1.8
            )
            clip = self.materialized_branch(output, "clip3d", 1.5, 1.9)
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

    def test_paired_report_rejects_claimed_measurements_without_real_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixed = self.branch("fixed-bin", 1.4, 1.8)
            clip = self.branch("clip3d", 1.5, 1.9)
            fixed["r2_executed"] = False
            fixed["measured_bips2_trans"] = 999.0

            with self.assertRaisesRegex(
                ValueError, "measurement|artifact|R2"
            ):
                publish_paired_comparison(
                    fixed, clip, output, r2_requested=True
                )
            self.assertFalse((output / "paired_comparison.json").exists())
            self.assertFalse((output / "paired_comparison.csv").exists())

    def test_paired_report_recomputes_bips_and_checks_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            fixed = self.materialized_branch(
                output, "fixed-bin", 1.4, 1.8
            )
            clip = self.materialized_branch(output, "clip3d", 1.5, 1.9)
            fixed["measured_bips2_trans"] = 999.0
            with self.assertRaisesRegex(ValueError, "BIPS2_trans"):
                publish_paired_comparison(
                    fixed, clip, output, r2_requested=True
                )

            fixed = self.materialized_branch(
                output, "fixed-bin", 1.4, 1.8
            )
            r2_path = Path(fixed["artifacts"]["r2_result"])
            write_json(r2_path, {"ipc2": 9.0})
            with self.assertRaisesRegex(ValueError, "artifact identity"):
                publish_paired_comparison(
                    fixed, clip, output, r2_requested=True
                )
            self.assertFalse((output / "paired_comparison.json").exists())
            self.assertFalse((output / "paired_comparison.csv").exists())


class TransientROMPairedPipelineTests(unittest.TestCase):
    """Catch single-layout validation or partially paired measured evidence."""

    def setUp(self):
        self.fixture = transient_tests.ROMPipelineTests(
            "test_summary_separates_predicted_and_validated_transient_results"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.temporary.cleanup)
        config = self.fixture.config()
        config["layout_optimizer"].update({
            "lambda_wire": 0.0020119160767721133,
            "wire_objective": "discrete-partition",
            "partition_grid_steps": 41,
            "include_fixed_baseline": True,
        })
        config["delay"]["wire_aggregation"] = "traffic-weighted"
        write_json(self.fixture.config_path, config)

        design = build_design(
            transient_tests.CalibrationDesignTests.model(),
            [1],
            parse_settings(config),
        )
        self.fixed_layout = self.fixture.steady / "hotspot/layout.json"
        write_json(self.fixed_layout, design["base_layout"])
        proposed = deepcopy(design["base_layout"])
        proposed_l2 = next(
            module for module in proposed["modules"] if module["kind"] == "l2"
        )
        proposed_l2["x_mm"] = float(proposed_l2["x_mm"]) + 0.01
        write_json(self.fixture.proposed_layout, proposed)

        package = self.fixture.output / "rom_package"
        package.mkdir(parents=True)
        self.fixture.bind_package(package)
        self.optimization = self.fixture.optimization()
        fixed_l2 = next(
            module for module in design["base_layout"]["modules"]
            if module["kind"] == "l2"
        )
        self.optimization["parameters"].update({
            "wire_objective": "discrete-partition",
            "lambda_wire": 0.0020119160767721133,
            "wire_aggregation": "traffic-weighted",
            "wire_rounding": "nearest",
        })
        self.optimization["selected"].update({
            "tier": int(proposed_l2["tier"]),
            "x_mm": float(proposed_l2["x_mm"]),
            "y_mm": float(proposed_l2["y_mm"]),
            "r2_wire_cycles": 7,
        })
        self.optimization["search"] = {
            "fixed_baseline_included": True,
            "fixed_baseline": {
                "tier": int(fixed_l2["tier"]),
                "x_mm": float(fixed_l2["x_mm"]),
                "y_mm": float(fixed_l2["y_mm"]),
                "r2_wire_cycles": 8,
            },
        }

    @staticmethod
    def vector(cycle: int) -> dict:
        return {
            "critical_l1d_to_l2_cycles": 12 + cycle,
            "components_cycles": {"layout_wire": cycle},
            "wire_cycle_aggregation_for_r2": "traffic-weighted",
            "layout_delays": {"traffic_weighted_wire_cycles": cycle},
        }

    def run_pipeline(self, *, execute_r2: bool, fixed_cycle: int = 8,
                     clip_cycle: int = 7,
                     mutate_fixed_trace_identity: bool = False):
        from workflow.transient.rom.run_pipeline import run_transient_rom_pipeline

        fixture = self.fixture
        search_dirs = []

        def final_search(*args, **kwargs):
            output_dir = Path(args[3])
            search_dirs.append(output_dir)
            result = fixture.final_validation(
                output_dir=output_dir, modules_path=Path(args[0]),
                layout_path=Path(args[1]), power_windows_path=Path(args[2]),
                config_path=Path(args[4]), hotspot_path=Path(kwargs["hotspot"]),
            )
            if mutate_fixed_trace_identity and output_dir.name == "fixed_bin":
                for manifest_path in output_dir.glob(
                    "frequency_*_ghz/transient_trace_manifest.json"
                ):
                    manifest = read_json(manifest_path)
                    manifest["source_layout"] = str(
                        fixture.proposed_layout.resolve()
                    )
                    write_json(manifest_path, manifest)
            return result

        def build_vector(*args, **kwargs):
            layout = Path(args[5]).resolve()
            cycle = (
                fixed_cycle if layout == self.fixed_layout.resolve()
                else clip_cycle
            )
            vector = self.vector(cycle)
            write_json(Path(args[2]), vector)
            return vector

        def run_r2(*args, **kwargs):
            output_dir = Path(args[2])
            ipc2 = 1.4 if output_dir.parent.name == "fixed_bin" else 1.5
            result = {"ipc2": ipc2}
            write_json(output_dir / "r2_result.json", result)
            return result

        with patch(
            "workflow.transient.rom.run_pipeline.DEFAULT_HOTSPOT", fixture.hotspot
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_source_r1",
            return_value={"metadata": {"workload": "matmul"}},
        ), patch(
            "workflow.transient.rom.run_pipeline.validate_steady_output",
            return_value={"summary": fixture.steady_summary()},
        ), patch(
            "workflow.transient.rom.run_pipeline.prepare_power_windows",
            return_value=fixture.prepared(),
        ), patch(
            "workflow.transient.rom.run_pipeline.optimize_transient_layout",
            return_value=self.optimization,
        ), patch(
            "workflow.transient.rom.run_pipeline.search_layout_frequency",
            side_effect=final_search,
        ), patch(
            "workflow.transient.rom.run_pipeline.build_vector",
            side_effect=build_vector,
        ) as build_vector_mock, patch(
            "workflow.transient.rom.run_pipeline.run_r2",
            side_effect=run_r2,
        ) as run_r2_mock:
            summary = run_transient_rom_pipeline(
                fixture.source_r1, fixture.steady, fixture.output,
                fixture.config_path, fixture.transient_r1,
                calibrate=False, execute_r2=execute_r2,
            )
        return summary, search_dirs, build_vector_mock, run_r2_mock

    def test_discrete_mode_validates_both_layouts_without_publishing_unmeasured_pair(self):
        summary, search_dirs, build_vector_mock, run_r2_mock = self.run_pipeline(
            execute_r2=False
        )

        self.assertEqual(search_dirs, [
            self.fixture.output / "final_validation/fixed_bin",
            self.fixture.output / "final_validation/clip3d",
        ])
        self.assertEqual(build_vector_mock.call_count, 2)
        self.assertEqual(
            build_vector_mock.call_args_list[0].args[5], self.fixed_layout.resolve()
        )
        self.assertEqual(
            build_vector_mock.call_args_list[1].args[5],
            self.fixture.proposed_layout.resolve(),
        )
        run_r2_mock.assert_not_called()
        self.assertFalse(
            (self.fixture.output / "final_validation/paired_comparison.json").exists()
        )
        self.assertIsNone(
            summary["branches"]["fixed_bin"]["measured_bips2_trans"]
        )
        self.assertIsNone(
            summary["branches"]["clip3d"]["measured_bips2_trans"]
        )

    def test_discrete_mode_publishes_two_real_hotspot_and_r2_measurements(self):
        summary, _search_dirs, _build_vector, run_r2_mock = self.run_pipeline(
            execute_r2=True
        )

        self.assertEqual(run_r2_mock.call_count, 2)
        fixed = summary["branches"]["fixed_bin"]
        clip = summary["branches"]["clip3d"]
        self.assertEqual(fixed["measured_bips2_trans"], 1.4 * 1.8)
        self.assertEqual(clip["measured_bips2_trans"], 1.5 * 1.8)
        paired = read_json(
            self.fixture.output / "final_validation/paired_comparison.json"
        )
        self.assertEqual(
            paired["bips2_trans_improvement_percent"],
            ((1.5 * 1.8) - (1.4 * 1.8)) / (1.4 * 1.8) * 100.0,
        )
        self.assertNotIn("bips2", summary)

    def test_cycle_mismatch_fails_closed_before_either_r2(self):
        summary, _search_dirs, _build_vector, run_r2_mock = self.run_pipeline(
            execute_r2=True, clip_cycle=6
        )

        run_r2_mock.assert_not_called()
        self.assertEqual(
            summary["branches"]["clip3d"]["failure"]["category"],
            "integer_cycle_identity",
        )
        self.assertTrue(
            (self.fixture.output
             / "final_validation/clip3d/branch_summary.json").is_file()
        )
        self.assertFalse(
            (self.fixture.output / "final_validation/paired_comparison.json").exists()
        )

    def test_cross_layout_hotspot_trace_fails_before_either_r2(self):
        summary, _search_dirs, _build_vector, run_r2_mock = self.run_pipeline(
            execute_r2=True, mutate_fixed_trace_identity=True
        )

        run_r2_mock.assert_not_called()
        self.assertEqual(
            summary["branches"]["fixed_bin"]["failure"]["category"],
            "validation_contract_error",
        )
        self.assertFalse(
            (self.fixture.output / "final_validation/paired_comparison.json").exists()
        )

    def test_fixed_cycle_mismatch_fails_closed_before_either_r2(self):
        summary, _search_dirs, _build_vector, run_r2_mock = self.run_pipeline(
            execute_r2=True, fixed_cycle=9
        )

        run_r2_mock.assert_not_called()
        self.assertEqual(
            summary["branches"]["fixed_bin"]["failure"]["category"],
            "integer_cycle_identity",
        )
        self.assertFalse(
            (self.fixture.output / "final_validation/paired_comparison.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
