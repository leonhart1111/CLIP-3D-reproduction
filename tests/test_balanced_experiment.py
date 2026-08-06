from __future__ import annotations

import tempfile
import unittest
import sys
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from workflow.common import read_json, sha256_file, write_json
from workflow.r1_catalog import (
    build_catalogue,
    canonical_directories,
    snapshot_canonical_artifacts,
)


def _paired_process_smoke(key, status_root):
    """Top-level picklable worker used only for a no-tools process smoke."""
    from workflow.r2.run_paired_sweep import _pending_pair
    return _pending_pair(key, status_root)


class CanonicalFixtureTests(unittest.TestCase):
    def make_grid_fixture(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "paper"
        experiment = Path(temporary.name) / "r1_cache_sweep.json"
        grid = {
            "workloads": ["fft", "matmul"],
            "l1d_sizes": ["16kB", "32kB"],
            "l2_sizes": ["128kB"],
            "profiles": {
                "paper": {
                    "instruction_window_scope": "cpu0",
                    "measure_insts": 100,
                },
            },
        }
        write_json(experiment, grid)
        stats = (
            "---------- Begin Simulation Statistics ----------\n"
            "system.cpu0.commitStats0.numInsts 100\n"
            "system.cpu0.numCycles 50\n"
            "system.cpu1.commitStats0.numInsts 1\n"
            "system.cpu1.numCycles 50\n"
            "system.cpu2.commitStats0.numInsts 1\n"
            "system.cpu2.numCycles 50\n"
            "system.cpu3.commitStats0.numInsts 1\n"
            "system.cpu3.numCycles 50\n"
        )
        for workload in grid["workloads"]:
            for l1d_size in grid["l1d_sizes"]:
                point = root / workload / f"l1d_{l1d_size}" / "l2_128kB"
                point.mkdir(parents=True)
                write_json(point / "status.json", {"state": "success"})
                write_json(point / "r1_metadata.json", {
                    "workload": workload,
                    "l1d_size": l1d_size,
                    "l2_size": "128kB",
                    "instruction_window_scope": "cpu0",
                    "measure_insts_cpu0": 100,
                    "num_cores": 4,
                })
                (point / "stats.txt").write_text(stats, encoding="utf-8")

        extra = root / "fft/l1d_16kB/l2_128kB.corrupt_duplicate_fixture"
        extra.mkdir(parents=True)
        write_json(extra / "status.json", {"state": "success"})
        return root, experiment


class CanonicalCatalogueTests(CanonicalFixtureTests):
    def test_catalogue_returns_only_expected_paths_and_reports_extra_status(self):
        """A sibling status directory must not create a canonical architecture."""
        root, experiment = self.make_grid_fixture()

        catalogue = build_catalogue(root, experiment, profile="paper")

        self.assertEqual(catalogue["expected_count"], 4)
        self.assertEqual(catalogue["valid_count"], 4)
        self.assertEqual(len(catalogue["canonical_records"]), 4)
        self.assertEqual(len(catalogue["excluded_noncanonical"]), 1)
        self.assertIn("corrupt_duplicate_fixture",
                      catalogue["excluded_noncanonical"][0]["relative_directory"])
        self.assertEqual(canonical_directories(catalogue), [
            root / "fft/l1d_16kB/l2_128kB",
            root / "fft/l1d_32kB/l2_128kB",
            root / "matmul/l1d_16kB/l2_128kB",
            root / "matmul/l1d_32kB/l2_128kB",
        ])

    def test_catalogue_rejects_metadata_that_disagrees_with_canonical_path(self):
        """A wrong cache-size field must invalidate, rather than relabel, a point."""
        root, experiment = self.make_grid_fixture()
        metadata_path = root / "fft/l1d_16kB/l2_128kB/r1_metadata.json"
        metadata = read_json(metadata_path)
        metadata["l1d_size"] = "32kB"
        write_json(metadata_path, metadata)

        catalogue = build_catalogue(root, experiment, profile="paper")

        record = next(item for item in catalogue["canonical_records"]
                      if item["key"]["workload"] == "fft"
                      and item["key"]["l1d_size"] == "16kB")
        self.assertFalse(record["valid"])
        self.assertIn("metadata l1d_size", " ".join(record["errors"]))

    def test_catalogue_rejects_profile_or_counter_mismatch(self):
        """A claimed paper run needs its profile scope and all four live cores."""
        root, experiment = self.make_grid_fixture()
        point = root / "fft/l1d_16kB/l2_128kB"
        metadata = read_json(point / "r1_metadata.json")
        metadata["instruction_window_scope"] = "all-cores"
        write_json(point / "r1_metadata.json", metadata)
        stats = (point / "stats.txt").read_text(encoding="utf-8")
        (point / "stats.txt").write_text(
            stats.replace("system.cpu3.numCycles 50", "system.cpu3.numCycles 0"),
            encoding="utf-8",
        )

        catalogue = build_catalogue(root, experiment, profile="paper")

        record = catalogue["canonical_records"][0]
        self.assertFalse(record["valid"])
        self.assertIn("instruction_window_scope", " ".join(record["errors"]))
        self.assertIn("CPU3", " ".join(record["errors"]))

    def test_catalogue_rejects_nonfinite_core_counter(self):
        """A non-finite gem5 counter cannot qualify as a positive measurement."""
        root, experiment = self.make_grid_fixture()
        point = root / "fft/l1d_16kB/l2_128kB"
        stats = (point / "stats.txt").read_text(encoding="utf-8")
        (point / "stats.txt").write_text(
            stats.replace("system.cpu2.numCycles 50", "system.cpu2.numCycles nan"),
            encoding="utf-8",
        )

        catalogue = build_catalogue(root, experiment, profile="paper")

        record = catalogue["canonical_records"][0]
        self.assertFalse(record["valid"])
        self.assertIn("CPU2 cycles", " ".join(record["errors"]))

    def test_catalogue_rejects_null_status_json(self):
        """A JSON null cannot prove that a canonical R1 point succeeded."""
        root, experiment = self.make_grid_fixture()
        point = root / "fft/l1d_16kB/l2_128kB"
        write_json(point / "status.json", None)

        catalogue = build_catalogue(root, experiment, profile="paper")

        record = catalogue["canonical_records"][0]
        self.assertFalse(record["valid"])
        self.assertIn("status.json must contain an object", record["errors"])

    def test_catalogue_rejects_null_metadata_json(self):
        """A JSON null cannot supply a canonical point's architecture metadata."""
        root, experiment = self.make_grid_fixture()
        point = root / "fft/l1d_16kB/l2_128kB"
        write_json(point / "r1_metadata.json", None)

        catalogue = build_catalogue(root, experiment, profile="paper")

        record = catalogue["canonical_records"][0]
        self.assertFalse(record["valid"])
        self.assertIn("r1_metadata.json must contain an object", record["errors"])

    def test_snapshot_hashes_only_valid_canonical_artifacts(self):
        """Artifact provenance must omit invalid points and extra directories."""
        root, experiment = self.make_grid_fixture()
        metadata_path = root / "fft/l1d_16kB/l2_128kB/r1_metadata.json"
        metadata = read_json(metadata_path)
        metadata["workload"] = "wrong"
        write_json(metadata_path, metadata)

        snapshot = snapshot_canonical_artifacts(
            build_catalogue(root, experiment, profile="paper")
        )

        self.assertEqual(len(snapshot), 9)
        self.assertNotIn("fft/l1d_16kB/l2_128kB/stats.txt", snapshot)
        self.assertIn("matmul/l1d_32kB/l2_128kB/status.json", snapshot)
        for digest in snapshot.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class CanonicalAuditTests(CanonicalFixtureTests):
    def test_audit_counts_canonical_and_excluded_separately(self):
        """A stray status must be reported, not treated as a fifth R1 point."""
        from workflow.analysis.audit_r1 import audit

        root, experiment = self.make_grid_fixture()
        output = root.parent / "audit.json"

        result = audit(root, output, experiment, "paper", expected_points=4)

        self.assertEqual(result["canonical_status_count"], 4)
        self.assertEqual(result["valid_success_count"], 4)
        self.assertEqual(result["excluded_noncanonical_count"], 1)
        self.assertTrue(result["complete"])
        self.assertEqual(read_json(output), result)

    def test_audit_rejects_a_plan_with_too_few_jobs(self):
        """A one-job plan cannot certify a four-point canonical R1 grid."""
        from workflow.analysis.audit_r1 import audit

        root, experiment = self.make_grid_fixture()
        write_json(root / "planned_jobs.json", {"job_count": 1})

        result = audit(root, root.parent / "audit.json", experiment, "paper",
                       expected_points=4)

        self.assertEqual(result["planned_points"], 1)
        self.assertFalse(result["complete"])

    def test_refresh_writes_equal_hash_manifests_after_plan_only_regeneration(self):
        """Refreshing the plan must preserve all canonical measured artifacts."""
        from workflow.analysis.refresh_r1_plan import refresh

        root, experiment = self.make_grid_fixture()
        output = root.parent / "refresh"

        def write_plan(command, **_kwargs):
            self.assertNotIn("--execute", command)
            output_root = Path(command[command.index("--output-root") + 1])
            profile = command[command.index("--profile") + 1]
            write_json(output_root / profile / "planned_jobs.json",
                       {"job_count": 4, "jobs": []})
            return CompletedProcess(command, 0)

        with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                   side_effect=write_plan):
            result = refresh(root, experiment, "paper", output)

        before = read_json(output / "canonical_r1_before.sha256.json")
        after = read_json(output / "canonical_r1_after.sha256.json")
        self.assertEqual(before, after)
        self.assertTrue(result["canonical_artifacts_unchanged"])
        self.assertEqual(result["plan_count"], 4)
        self.assertNotIn("--execute", result["command"])
        output_root = Path(result["command"][result["command"].index("--output-root") + 1])
        self.assertEqual(output_root / result["profile"], root)

    def test_refresh_requires_root_to_be_the_selected_profile_directory(self):
        """A planner cannot refresh a root different from its profile output path."""
        from workflow.analysis.refresh_r1_plan import refresh

        root, experiment = self.make_grid_fixture()
        wrong_root = root.parent / "not_the_paper_profile"
        root.rename(wrong_root)

        with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                   return_value=CompletedProcess([], 0)):
            with self.assertRaisesRegex(ValueError,
                                        "root directory must match profile"):
                refresh(wrong_root, experiment, "paper", wrong_root.parent / "refresh")

    def test_refresh_aborts_if_any_canonical_hash_changes(self):
        """Changing a canonical stats file during refresh invalidates the result."""
        from workflow.analysis.refresh_r1_plan import refresh

        root, experiment = self.make_grid_fixture()
        output = root.parent / "refresh"

        def mutate_stats_and_return_success(command, **_kwargs):
            write_json(root / "planned_jobs.json", {"job_count": 4, "jobs": []})
            stats_path = root / "fft/l1d_16kB/l2_128kB/stats.txt"
            stats_path.write_text(
                stats_path.read_text(encoding="utf-8").replace(
                    "system.cpu0.numCycles 50", "system.cpu0.numCycles 0"
                ),
                encoding="utf-8",
            )
            return CompletedProcess(command, 0)

        with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                   side_effect=mutate_stats_and_return_success):
            with self.assertRaisesRegex(RuntimeError,
                                        "canonical R1 artifacts changed"):
                refresh(root, experiment, "paper", output)
        self.assertTrue((output / "canonical_r1_after.sha256.json").is_file())

    def test_refresh_reports_mutation_before_a_failed_planner(self):
        """A failed planner cannot mask a changed canonical R1 artifact."""
        from workflow.analysis.refresh_r1_plan import refresh

        root, experiment = self.make_grid_fixture()
        output = root.parent / "refresh"

        def mutate_stats_and_return_failure(command, **_kwargs):
            stats_path = root / "fft/l1d_16kB/l2_128kB/stats.txt"
            stats_path.write_text(
                stats_path.read_text(encoding="utf-8").replace(
                    "system.cpu0.numCycles 50", "system.cpu0.numCycles 0"
                ),
                encoding="utf-8",
            )
            return CompletedProcess(command, 9)

        with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                   side_effect=mutate_stats_and_return_failure):
            with self.assertRaisesRegex(RuntimeError,
                                        "canonical R1 artifacts changed"):
                refresh(root, experiment, "paper", output)

        before = read_json(output / "canonical_r1_before.sha256.json")
        after = read_json(output / "canonical_r1_after.sha256.json")
        self.assertNotEqual(before, after)

    def test_refresh_reports_failed_planner_after_writing_unchanged_hashes(self):
        """An unchanged failed planner still records its two immutable manifests."""
        from workflow.analysis.refresh_r1_plan import refresh

        root, experiment = self.make_grid_fixture()
        output = root.parent / "refresh"

        with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                   return_value=CompletedProcess([], 7)):
            with self.assertRaisesRegex(RuntimeError, "exit status 7"):
                refresh(root, experiment, "paper", output)

        before = read_json(output / "canonical_r1_before.sha256.json")
        after = read_json(output / "canonical_r1_after.sha256.json")
        self.assertEqual(before, after)


class CanonicalLiftingTests(CanonicalFixtureTests):
    classification = {
        "mode": "operational-exploratory-traffic-weighted",
        "non_formal": True,
        "paper_equivalent": False,
        "shared_parameter_accepted": False,
    }

    def _add_noncanonical_duplicate(self, root: Path) -> None:
        source = root / "fft/l1d_16kB/l2_128kB"
        duplicate = root / "fft/l1d_16kB/l2_128kB.corrupt_duplicate_fixture"
        write_json(duplicate / "r1_metadata.json",
                   read_json(source / "r1_metadata.json"))
        (duplicate / "stats.txt").write_text(
            (source / "stats.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    def _write_config(self, root: Path, classification: dict | None = None) -> Path:
        config = root / "lifting_config.json"
        payload = {
            "name": "fixture_layout_only",
            "physical": {"r_convec_k_per_w": 5.0},
        }
        if classification is not None:
            payload["experiment_classification"] = classification
        write_json(config, payload)
        return config

    def _write_completed_layout_output(self, output: Path, config: dict,
                                       layout_method: str = "fixed-bin") -> None:
        from workflow.run_lifting_sweep import required_artifacts

        for artifact in required_artifacts:
            write_json(output / artifact, {})
        write_json(output / "run_config.json", {"config": config})
        write_json(output / "pipeline_summary.json", {
            "layout_method": layout_method,
            "cooling": {"r_convec_k_per_w": 5.0},
            "ipc2": None,
            "bips2": None,
        })

    def _cli_argv(self, root: Path, experiment: Path, output: Path,
                  config: Path) -> list[str]:
        return [
            "run_lifting_sweep.py", "--r1-root", str(root),
            "--r1-experiment", str(experiment), "--r1-profile", "paper",
            "--output-root", str(output), "--config", str(config),
        ]

    def test_discover_uses_only_valid_canonical_catalogue_points(self):
        """A metadata-bearing sibling is excluded instead of becoming a fifth job."""
        from workflow.run_lifting_sweep import discover

        root, experiment = self.make_grid_fixture()
        self._add_noncanonical_duplicate(root)

        points, catalogue = discover(root, experiment, "paper", None, True)

        self.assertEqual(len(points), 4)
        self.assertEqual(len(catalogue["excluded_noncanonical"]), 1)
        self.assertNotIn("corrupt_duplicate", "\n".join(map(str, points)))

    def test_discover_filters_workloads_after_validating_full_catalogue(self):
        """Selecting FFT still validates the configured MATMUL points first."""
        from workflow.run_lifting_sweep import discover

        root, experiment = self.make_grid_fixture()

        points, _catalogue = discover(root, experiment, "paper", {"fft"}, True)

        self.assertEqual(len(points), 2)
        self.assertTrue(all("/fft/" in str(point) for point in points))

    def test_discover_rejects_missing_canonical_point_before_workload_filter(self):
        """A selected workload cannot hide a missing configured architecture."""
        from workflow.run_lifting_sweep import discover

        root, experiment = self.make_grid_fixture()
        missing = root / "matmul/l1d_32kB/l2_128kB"
        (missing / "stats.txt").unlink()

        with self.assertRaisesRegex(ValueError, "canonical R1 catalogue"):
            discover(root, experiment, "paper", {"fft"}, True)

    def test_cli_writes_canonical_layout_only_provenance_report(self):
        """The launcher records catalogue and config provenance without running tools."""
        from workflow.run_lifting_sweep import main

        root, experiment = self.make_grid_fixture()
        self._add_noncanonical_duplicate(root)
        output = root.parent / "lifting"
        config = self._write_config(root.parent, self.classification)

        def layout_only_job(job):
            r1, destination, *_rest = job
            return {
                "r1": str(r1), "output": str(destination), "state": "success",
                "summary": {"ipc2": None, "bips2": None},
            }

        with patch("workflow.run_lifting_sweep.one_job", side_effect=layout_only_job):
            with patch.object(sys, "argv", self._cli_argv(root, experiment, output, config)), redirect_stdout(StringIO()):
                main()

        report = read_json(output / "sweep_status.json")
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["config"], str(config.resolve()))
        self.assertEqual(report["config_sha256"], sha256_file(config))
        self.assertEqual(report["classification"], self.classification)
        self.assertEqual(report["canonical_expected"], 4)
        self.assertEqual(report["canonical_selected"], 4)
        self.assertEqual(len(report["excluded_noncanonical"]), 1)
        self.assertFalse(report["run_r2"])
        self.assertFalse(report["contains_r2"])
        self.assertEqual(report["selected_workload_count"], 2)
        self.assertEqual(report["executed"], 4)
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["failed"], 0)

    def test_completed_rejects_a_summary_missing_required_pipeline_artifacts(self):
        """A matching config alone cannot resume an incomplete lifting output."""
        from workflow.run_lifting_sweep import completed

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"physical": {"r_convec_k_per_w": 5.0}}
            write_json(output / "run_config.json", {"config": config})
            write_json(output / "pipeline_summary.json", {
                "layout_method": "fixed-bin",
                "cooling": {"r_convec_k_per_w": 5.0},
                "ipc2": None,
                "bips2": None,
            })

            self.assertFalse(completed(output, config, "fixed-bin", False))

    def test_cli_reports_then_exits_for_layout_only_r2_result(self):
        """A layout-only R2 result is persisted as a controlled sweep failure."""
        from workflow.run_lifting_sweep import main

        root, experiment = self.make_grid_fixture()
        output = root.parent / "lifting"
        config = self._write_config(root.parent, self.classification)

        def r2_job(job):
            r1, destination, *_rest = job
            return {
                "r1": str(r1), "output": str(destination), "state": "success",
                "summary": {"ipc2": 1.0, "bips2": 2.0},
            }

        stderr = StringIO()
        with patch("workflow.run_lifting_sweep.one_job", side_effect=r2_job):
            with patch.object(sys, "argv", self._cli_argv(root, experiment, output, config)):
                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(stderr.getvalue(), "")
        report = read_json(output / "sweep_status.json")
        self.assertEqual(report["schema_version"], 3)
        self.assertTrue(report["contains_r2"])
        self.assertEqual(report["validation_error"],
                         "layout-only lifting sweep contains R2 performance results")

    def test_cli_rejects_missing_or_malformed_classification_before_jobs(self):
        """Every sweep must have a complete non-formal source classification."""
        from workflow.run_lifting_sweep import main

        root, experiment = self.make_grid_fixture()
        malformed = {"mode": "operational-exploratory-traffic-weighted",
                     "non_formal": True}

        def layout_only_job(job):
            r1, destination, *_rest = job
            return {
                "r1": str(r1), "output": str(destination), "state": "success",
                "summary": {"ipc2": None, "bips2": None},
            }

        for index, classification in enumerate((None, malformed)):
            config = self._write_config(root.parent / str(index), classification)
            output = root.parent / f"lifting-{index}"
            with patch("workflow.run_lifting_sweep.one_job", side_effect=layout_only_job) as one_job:
                with patch.object(sys, "argv", self._cli_argv(root, experiment, output, config)):
                    with self.assertRaisesRegex(ValueError, "experiment_classification"):
                        main()
            one_job.assert_not_called()
            self.assertFalse((output / "sweep_status.json").exists())

    def test_completed_clip3d_requires_optimizer_and_layout_selection(self):
        """CLIP resume state is incomplete until both selection artifacts exist."""
        from workflow.run_lifting_sweep import completed

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"physical": {"r_convec_k_per_w": 5.0}}
            self._write_completed_layout_output(output, config, "clip3d")

            self.assertFalse(completed(output, config, "clip3d", False))
            write_json(output / "optimizer_report.json", {})
            self.assertFalse(completed(output, config, "clip3d", False))
            write_json(output / "layout_selection.json", {})
            self.assertTrue(completed(output, config, "clip3d", False))

    def test_cli_counts_valid_skipped_layout_only_outputs_without_r2(self):
        """All valid completed layout-only outputs are skipped and remain R2-free."""
        from workflow.run_lifting_sweep import main

        root, experiment = self.make_grid_fixture()
        output = root.parent / "lifting"
        config_path = self._write_config(root.parent, self.classification)
        config = read_json(config_path)
        for workload in ("fft", "matmul"):
            for l1d_size in ("16kB", "32kB"):
                self._write_completed_layout_output(
                    output / workload / f"l1d_{l1d_size}" / "l2_128kB", config
                )

        with patch("workflow.run_lifting_sweep.one_job") as one_job:
            with patch.object(sys, "argv", self._cli_argv(root, experiment, output, config_path)):
                with redirect_stdout(StringIO()):
                    main()

        one_job.assert_not_called()
        report = read_json(output / "sweep_status.json")
        self.assertEqual(report["executed"], 0)
        self.assertEqual(report["skipped_count"], 4)
        self.assertFalse(report["contains_r2"])


class BalancedSelectionTests(unittest.TestCase):
    """Exercise the predeclared sample against complete temporary layout roots."""

    expected_pairs = [
        ("16kB", "128kB"), ("16kB", "512kB"), ("16kB", "2048kB"),
        ("32kB", "256kB"), ("32kB", "1024kB"),
        ("64kB", "128kB"), ("64kB", "512kB"), ("64kB", "2048kB"),
        ("128kB", "256kB"), ("128kB", "1024kB"),
    ]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.grid_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/r1_cache_sweep.json"
        )
        self.grid = read_json(self.grid_path)
        self.config_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
            "traffic_weighted_exploratory.json"
        )
        self.config = read_json(Path(__file__).resolve().parents[1] / (
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
            "traffic_weighted_exploratory.json"
        ))

    @property
    def manifest_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )

    def _write_layout_roots(self, config_path: Path | None = None,
                            config: dict | None = None) -> tuple[Path, Path]:
        """Create real, complete 100-point roots with layout-only artifacts."""
        from workflow.r1_catalog import expected_keys
        from workflow.run_lifting_sweep import (
            CLIP3D_REQUIRED_ARTIFACTS,
            required_artifacts,
        )

        config_path = (config_path or self.config_path).resolve()
        config = config if config is not None else self.config
        fixed_root = self.root / "fixed"
        clip_root = self.root / "clip"
        for root, method in ((fixed_root, "fixed-bin"), (clip_root, "clip3d")):
            for key in expected_keys(self.grid):
                point = root / key.relative_path()
                for artifact in required_artifacts:
                    write_json(point / artifact, {})
                if method == "clip3d":
                    for artifact in CLIP3D_REQUIRED_ARTIFACTS:
                        write_json(point / artifact, {})
                write_json(point / "run_config.json", {
                    "schema_version": 1,
                    "source": str(config_path),
                    "layout_method": method,
                    "config": config,
                })
                write_json(point / "pipeline_summary.json", {
                    "layout_method": method,
                    "layout_mode": method,
                    "ipc2": None,
                    "bips2": None,
                    "communication_profile": {
                        "status": "available",
                        "per_core": {
                            str(core): {"normalized_weight": 0.25}
                            for core in range(4)
                        },
                    },
                })
        return fixed_root, clip_root

    def _selection(self):
        from workflow.experiments.balanced50 import load_selection, selection_keys

        manifest = load_selection(self.manifest_path)
        return manifest, selection_keys(manifest)

    def test_tracked_manifest_has_the_exact_balanced_order_and_coverage(self):
        """An outcome-dependent or reordered sample cannot silently replace Balanced-50."""
        from workflow.experiments.balanced50 import validate_selection

        manifest, keys = self._selection()
        result = validate_selection(manifest, self.grid)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(result["selected_count"], 50)
        self.assertEqual(result["workload_count"], 5)
        self.assertEqual(len(set(keys)), 50)
        for workload in self.grid["workloads"]:
            pairs = [(key.l1d_size, key.l2_size) for key in keys
                     if key.workload == workload]
            self.assertEqual(pairs, self.expected_pairs)
            self.assertEqual(len(set(pairs)), 10)
            self.assertEqual({pair[0] for pair in pairs}, set(self.grid["l1d_sizes"]))
            self.assertEqual({pair[1] for pair in pairs}, set(self.grid["l2_sizes"]))

    def test_selection_rejects_reordered_workload_blocks(self):
        """Per-workload pair checks alone cannot preserve workload-major manifest order."""
        from workflow.experiments.balanced50 import validate_selection

        manifest, _keys = self._selection()
        reordered = deepcopy(manifest)
        reordered["points"] = reordered["points"][10:20] + reordered["points"][0:10] + (
            reordered["points"][20:]
        )

        with self.assertRaisesRegex(ValueError, "entire predeclared order"):
            validate_selection(reordered, self.grid)

    def test_load_selection_rejects_absolute_and_parent_reference_paths(self):
        """Manifest references must remain pinned beneath the repository root."""
        from workflow.experiments.balanced50 import load_selection

        manifest, _keys = self._selection()
        for bad_path in (str(self.grid_path.resolve()), "../r1_cache_sweep.json"):
            with self.subTest(path=bad_path):
                malformed = deepcopy(manifest)
                malformed["canonical_grid_config"]["path"] = bad_path
                path = self.root / f"malformed-{len(bad_path)}.json"
                write_json(path, malformed)
                with self.assertRaisesRegex(ValueError, "project-relative"):
                    load_selection(path)

    def test_preflight_reports_two_complete_layout_only_roots(self):
        """The selected 50 may proceed only after both whole 100-point roots validate."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()

        report = validate_layout_roots(
            fixed_root, clip_root, keys, self.config_path, selection=manifest,
        )

        self.assertEqual(report["mode"], "layout-only")
        self.assertEqual(report["selected_count"], 50)
        self.assertEqual(report["fixed"]["canonical_count"], 100)
        self.assertEqual(report["clip3d"]["canonical_count"], 100)
        self.assertEqual(report["fixed"]["existing_r2_count"], 0)
        self.assertEqual(report["clip3d"]["existing_r2_count"], 0)
        self.assertEqual(report["config_sha256"], sha256_file(self.config_path))

    def test_preflight_rejects_a_missing_unselected_canonical_point(self):
        """Checking only the selected 50 would hide an incomplete layout generation."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        missing = fixed_root / "fft/l1d_16kB/l2_256kB/run_config.json"
        missing.unlink()

        with self.assertRaisesRegex(ValueError, "fixed-bin root.*100"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_mismatched_layout_method(self):
        """A fixed output must never be mistaken for the CLIP-3D branch."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = clip_root / keys[0].relative_path()
        summary = read_json(point / "pipeline_summary.json")
        summary["layout_method"] = "fixed-bin"
        write_json(point / "pipeline_summary.json", summary)

        with self.assertRaisesRegex(ValueError, "clip3d.*layout method"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_changed_embedded_run_config(self):
        """A point regenerated with different lifting parameters cannot join the root."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        run_config = read_json(point / "run_config.json")
        run_config["config"]["physical"]["r_convec_k_per_w"] = 4.0
        write_json(point / "run_config.json", run_config)

        with self.assertRaisesRegex(ValueError, "embedded config"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_config_not_pinned_by_selection(self):
        """Consistent roots cannot substitute a different classified experiment config."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        changed_config = deepcopy(self.config)
        changed_config["experiment_classification"]["non_formal"] = False
        changed_path = self.root / "changed_classified_config.json"
        write_json(changed_path, changed_config)
        fixed_root, clip_root = self._write_layout_roots(changed_path, changed_config)

        with self.assertRaisesRegex(ValueError, "does not match pinned"):
            validate_layout_roots(fixed_root, clip_root, keys, changed_path,
                                  selection=manifest)

    def test_preflight_rejects_missing_traffic_communication_profile(self):
        """Traffic-weighted layouts require recorded, usable per-core communication weights."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        summary = read_json(point / "pipeline_summary.json")
        summary.pop("communication_profile")
        write_json(point / "pipeline_summary.json", summary)

        with self.assertRaisesRegex(ValueError, "communication profile"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_invalid_traffic_weight_values(self):
        """Traffic weights must be finite, non-negative, and normalized to one."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        for label, value, all_weights in (
                ("nonfinite", float("nan"), False),
                ("negative", -0.1, False),
                ("unnormalized", 0.2, True)):
            with self.subTest(label=label):
                fixed_root, clip_root = self._write_layout_roots()
                point = fixed_root / keys[0].relative_path()
                summary = read_json(point / "pipeline_summary.json")
                weights = summary["communication_profile"]["per_core"]
                if all_weights:
                    for record in weights.values():
                        record["normalized_weight"] = value
                else:
                    weights["0"]["normalized_weight"] = value
                write_json(point / "pipeline_summary.json", summary)

                with self.assertRaisesRegex(ValueError, "communication profile"):
                    validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                          selection=manifest)

    def test_layout_only_preflight_rejects_existing_r2_results(self):
        """The 200-output layout checkpoint must contain no measured R2 performance."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        summary = read_json(point / "pipeline_summary.json")
        summary["ipc2"] = 1.25
        summary["bips2"] = 2.5
        write_json(point / "pipeline_summary.json", summary)

        with self.assertRaisesRegex(ValueError, "layout-only.*R2"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_resume_preflight_requires_and_uses_an_existing_r2_validator(self):
        """Resume mode delegates all existing R2 provenance decisions to its caller."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        summary = read_json(point / "pipeline_summary.json")
        summary["ipc2"] = 1.25
        summary["bips2"] = 2.5
        write_json(point / "pipeline_summary.json", summary)

        with self.assertRaisesRegex(ValueError, "existing_r2_validator"):
            validate_layout_roots(
                fixed_root, clip_root, keys, self.config_path,
                require_layout_only=False, selection=manifest,
            )

        calls = []

        def validator(method, key, output):
            calls.append((method, key, output))
            return {"accepted": True}

        report = validate_layout_roots(
            fixed_root, clip_root, keys, self.config_path,
            require_layout_only=False, selection=manifest,
            existing_r2_validator=validator,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "fixed-bin")
        self.assertEqual(calls[0][1], keys[0])
        self.assertEqual(report["fixed"]["existing_r2_count"], 1)
        self.assertEqual(report["clip3d"]["existing_r2_count"], 0)

    def test_resume_preflight_rejects_a_validator_that_does_not_accept(self):
        """Calling a provenance callback is insufficient unless it explicitly accepts R2."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        summary = read_json(point / "pipeline_summary.json")
        summary["ipc2"] = 1.25
        summary["bips2"] = 2.5
        write_json(point / "pipeline_summary.json", summary)

        with self.assertRaisesRegex(ValueError, "did not accept"):
            validate_layout_roots(
                fixed_root, clip_root, keys, self.config_path,
                require_layout_only=False, selection=manifest,
                existing_r2_validator=lambda *_args: {"accepted": False},
            )


class StrictR2ReuseTests(unittest.TestCase):
    """Reject stale or scientifically incompatible R2 cache attachments."""

    overrides = {
        "l1i_tag_latency": 2,
        "l1i_data_latency": 2,
        "l1i_response_latency": 1,
        "l1d_tag_latency": 3,
        "l1d_data_latency": 3,
        "l1d_response_latency": 1,
        "l2_tag_latency": 12,
        "l2_data_latency": 12,
        "l2_response_latency": 1,
        "xbar_frontend_latency": 1,
        "xbar_forward_latency": 8,
        "xbar_response_latency": 1,
        "xbar_snoop_response_latency": 1,
    }
    gem5_args = [
        "--l1i-tag-latency", "2",
        "--l1i-data-latency", "2",
        "--l1i-response-latency", "1",
        "--l1d-tag-latency", "3",
        "--l1d-data-latency", "3",
        "--l1d-response-latency", "1",
        "--l2-tag-latency", "12",
        "--l2-data-latency", "12",
        "--l2-response-latency", "1",
        "--xbar-frontend-latency", "1",
        "--xbar-forward-latency", "8",
        "--xbar-response-latency", "1",
        "--xbar-snoop-response-latency", "1",
    ]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.r1 = self.root / "canonical-r1/fft/l1d_32kB/l2_512kB"
        self.fixed = self.root / "fixed/fft/l1d_32kB/l2_512kB"
        self.clip = self.root / "clip/fft/l1d_32kB/l2_512kB"
        self.config_path = self.root / "selected_experiment.json"
        source_config = Path(__file__).resolve().parents[1] / (
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
            "traffic_weighted_exploratory.json"
        )
        self.config = read_json(source_config)
        write_json(self.config_path, self.config)
        self.metadata = {
            "schema_version": 2,
            "workload": "fft",
            "l1i_size": "32kB",
            "l1d_size": "32kB",
            "l2_size": "512kB",
            "num_cores": 4,
            "cpu_type": "X86O3CPU",
            "clock": "2GHz",
            "warmup_insts_cpu0": 100000000,
            "measure_insts_cpu0": 500000000,
            "instruction_window_scope": "cpu0",
            "command": ["fft", "--threads", "4"],
            "stdin": None,
        }
        write_json(self.r1 / "r1_metadata.json", self.metadata)
        (self.r1 / "stats.txt").write_text(
            "---------- Begin Simulation Statistics ----------\n"
            "system.cpu0.commitStats0.numInsts 500000000\n"
            "system.cpu0.numCycles 200000000\n"
            "system.cpu1.commitStats0.numInsts 490000000\n"
            "system.cpu1.numCycles 200000000\n"
            "system.cpu2.commitStats0.numInsts 480000000\n"
            "system.cpu2.numCycles 200000000\n"
            "system.cpu3.commitStats0.numInsts 470000000\n"
            "system.cpu3.numCycles 200000000\n",
            encoding="utf-8",
        )
        write_json(self.r1 / "status.json", {
            "schema_version": 2,
            "state": "success",
            "instruction_window_scope": "cpu0",
        })
        for point, method, frequency in (
                (self.fixed, "fixed-bin", 1.4),
                (self.clip, "clip3d", 1.1)):
            self._write_point(point, method, frequency)
        self._write_source_result()

    def _write_point(self, point: Path, method: str, frequency: float) -> None:
        vector = {
            "schema_version": 1,
            "equation": 6,
            "components_cycles": {
                "l1i_cacti": 2,
                "l1d_cacti": 3,
                "l2_cacti": 12,
                "l2_arbitration": 3,
                "tsv": 2,
                "l1_pipeline": 1,
                "layout_wire": 3,
            },
            "critical_l1d_to_l2_cycles": 24,
            "gem5_overrides": deepcopy(self.overrides),
            "gem5_args": list(self.gem5_args),
            "layout": str((point / "hotspot/layout.json").resolve()),
            "layout_delays": {
                "tsv_hops": 1,
                "wire_cycles": 3,
                "wire_cycles_unrounded": 2.8,
                "maximum_wire_cycles": 4,
                "maximum_wire_cycles_unrounded": 3.6,
            },
            "wire_cycle_aggregation_for_r2": "traffic-weighted",
            "paper_parameters": [
                "Ncores-1 arbitration = 3",
                "2 cycles/TSV x 1",
                "L1 pipeline cycles = 1",
            ],
            "reproduction_assumptions": ["fixture preserves the complete vector shape"],
        }
        write_json(point / "r2_latency.json", vector)
        write_json(point / "run_config.json", {
            "schema_version": 1,
            "source": str(self.config_path.resolve()),
            "layout_method": method,
            "config": self.config,
        })
        write_json(point / "modules.json", {
            "schema_version": 2,
            "source_r1": str(self.r1.resolve()),
            "architecture": deepcopy(self.metadata),
            "ipc1": 2.75,
            "gamma": 0.2,
            "totals": {
                "dynamic_power_w": 80.0,
                "leakage_power_w": 20.0,
                "total_power_w": 100.0,
                "area_mm2": 45.0,
            },
            "modules": [],
        })
        write_json(point / "hotspot/thermal_result.json", {
            "schema_version": 1,
            "command": ["hotspot", "-c", "hotspot.config"],
            "return_code": 0,
            "power_trace": str((point / "hotspot/power.ptrace").resolve()),
            "steady_file": str((point / "hotspot/steady.txt").resolve()),
            "grid_steady_file": str((point / "hotspot/grid.steady.txt").resolve()),
            "tmax_k": 407.525,
            "tmax_c": 134.375,
            "peak_unit": "core0",
            "ambient_c": 25.0,
            "r_convec_k_per_w": 5.0,
            "sample_count": 1,
        })
        write_json(point / "performance.json", {
            "schema_version": 1,
            "equation": 13,
            "gamma": 0.2,
            "leakage_power_w": 20.0,
            "dynamic_power_w": 80.0,
            "tmax_f0_c": 134.375,
            "ambient_c": 25.0,
            "tsafe_c": 95.0,
            "f0_ghz": 2.0,
            "fmin_ghz": 0.4,
            "unclamped_solution_ghz": frequency,
            "sustainable_frequency_ghz": frequency,
            "estimated_tmax_at_fmin_c": 64.375,
            "thermal_feasible_at_fmin": True,
            "floor_power_scale": 0.36,
            "state": "thermally_limited",
            "ipc1": 2.75,
            "bips1_thermal": 2.75 * frequency,
        })
        write_json(point / "pipeline_summary.json", {
            "schema_version": 2,
            "r1": str(self.r1.resolve()),
            "output": str(point.resolve()),
            "experiment": self.config["name"],
            "workload": "fft",
            "l1d_size": "32kB",
            "l2_size": "512kB",
            "layout_method": method,
            "layout_mode": method,
            "gamma": 0.2,
            "tmax_c": 134.375,
            "sustainable_frequency_ghz": frequency,
            "ipc1": 2.75,
            "bips1_thermal": 2.75 * frequency,
            "ipc2": None,
            "bips2": None,
            "r2_source": None,
            "stage_seconds": {
                "mcpat": 1.0,
                "cacti": 1.0,
                "module_model": 1.0,
                "layout_and_hotspot": 1.0,
                "frequency_and_latency": 1.0,
            },
            "total_pipeline_seconds": 5.0,
            "artifacts": {
                "config": str((point / "run_config.json").resolve()),
                "modules": str((point / "modules.json").resolve()),
                "thermal": str((point / "hotspot/thermal_result.json").resolve()),
                "performance": str((point / "performance.json").resolve()),
                "r2_latency": str((point / "r2_latency.json").resolve()),
                "r2_result": None,
            },
        })

    def _identity(self, latency_path: Path) -> dict:
        metadata = read_json(self.r1 / "r1_metadata.json")
        return {
            "r1_directory": str(self.r1.resolve()),
            "r1_metadata_sha256": sha256_file(self.r1 / "r1_metadata.json"),
            "r1_stats_sha256": sha256_file(self.r1 / "stats.txt"),
            "latency_vector": str(latency_path.resolve()),
            "latency_sha256": sha256_file(latency_path),
            "instruction_window_scope": metadata["instruction_window_scope"],
            "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
            "measure_insts_cpu0": metadata["measure_insts_cpu0"],
        }

    def _write_source_result(self) -> None:
        output = self.fixed / "gem5_r2"
        result_path = output / "r2_result.json"
        identity = self._identity(self.fixed / "r2_latency.json")
        command = [
            "/opt/gem5.opt", "--listener-mode=off", f"--outdir={output.resolve()}",
            "/opt/clip_r1.py", "--stage", "R2", "--workload", "fft",
            "--l1i-size", "32kB", "--l1d-size", "32kB",
            "--l2-size", "512kB", "--warmup-insts", "100000000",
            "--measure-insts", "500000000", "--instruction-window-scope",
            identity["instruction_window_scope"], *self.gem5_args,
            "--options", "--threads 4",
        ]
        (output / "stats.txt").parent.mkdir(parents=True, exist_ok=True)
        (output / "stats.txt").write_text(
            "---------- Begin Simulation Statistics ----------\n"
            "system.cpu0.commitStats0.numInsts 162500000\n"
            "system.cpu0.numCycles 200000000\n"
            "system.cpu1.commitStats0.numInsts 162500000\n"
            "system.cpu1.numCycles 200000000\n"
            "system.cpu2.commitStats0.numInsts 162500000\n"
            "system.cpu2.numCycles 200000000\n"
            "system.cpu3.commitStats0.numInsts 162500000\n"
            "system.cpu3.numCycles 200000000\n",
            encoding="utf-8",
        )
        write_json(result_path, {
            "schema_version": 3,
            **identity,
            "command": command,
            "ipc2": 3.25,
            "per_core": [
                {"core": core, "instructions": 162500000,
                 "cycles": 200000000, "ipc": 0.8125}
                for core in range(4)
            ],
            "stats": str((output / "stats.txt").resolve()),
            "stats_sha256": sha256_file(output / "stats.txt"),
            "elapsed_seconds": 17.5,
        })
        write_json(output / "status.json", {
            "schema_version": 3,
            "state": "success",
            **identity,
            "r2_result": str(result_path.resolve()),
            "r2_result_sha256": sha256_file(result_path),
            "ipc2": 3.25,
            "stats": str((output / "stats.txt").resolve()),
            "stats_sha256": sha256_file(output / "stats.txt"),
            "return_code": 0,
            "command": command,
        })
        summary = read_json(self.fixed / "pipeline_summary.json")
        summary.update({
            "ipc2": 3.25,
            "bips2": 3.25 * 1.4,
            "r2_source": str(result_path.resolve()),
        })
        summary["stage_seconds"]["gem5_r2"] = 17.5
        summary["total_pipeline_seconds"] = 22.5
        summary["artifacts"]["r2_result"] = str(result_path.resolve())
        write_json(self.fixed / "pipeline_summary.json", summary)

    def _rewrite_result(self, mutate) -> None:
        result_path = self.fixed / "gem5_r2/r2_result.json"
        result = read_json(result_path)
        mutate(result)
        write_json(result_path, result)
        status_path = self.fixed / "gem5_r2/status.json"
        status = read_json(status_path)
        status["r2_result_sha256"] = sha256_file(result_path)
        write_json(status_path, status)

    def test_local_successful_cache_accepts_matching_latency_and_r1_identity(self):
        """Changing any requested local cache identity must make acceptance fail."""
        from workflow.r2.run_r2 import run, validate_local_result

        output = self.fixed / "gem5_r2"
        decision = validate_local_result(self.r1, self.fixed / "r2_latency.json", output)

        self.assertTrue(decision["accepted"])
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=AssertionError("compatible cache must resume")):
            result = run(self.r1, self.fixed / "r2_latency.json", output)
        self.assertEqual(result["ipc2"], 3.25)

    def test_local_successful_cache_rejects_another_latency_path_and_requires_rerun(self):
        """Status=success alone must never resume a result produced for another vector."""
        from workflow.r2.run_r2 import run, validate_local_result

        other = self.root / "another/r2_latency.json"
        write_json(other, read_json(self.fixed / "r2_latency.json"))
        output = self.fixed / "gem5_r2"

        decision = validate_local_result(self.r1, other, output)
        self.assertFalse(decision["accepted"])
        self.assertTrue(any("latency_vector" in reason for reason in decision["reasons"]))
        with self.assertRaisesRegex(ValueError, "--rerun"):
            run(self.r1, other, output)

    def test_local_success_status_with_missing_result_requires_explicit_rerun(self):
        """A missing result cannot turn a declared successful cache into an implicit rerun."""
        from workflow.r2.run_r2 import run

        output = self.fixed / "gem5_r2"
        (output / "r2_result.json").unlink()

        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=AssertionError("stale success must fail closed")):
            with self.assertRaisesRegex(ValueError, "--rerun"):
                run(self.r1, self.fixed / "r2_latency.json", output)

    def test_local_result_with_missing_success_status_requires_explicit_rerun(self):
        """A surviving successful result must not be overwritten when status is missing."""
        from workflow.r2.run_r2 import run

        output = self.fixed / "gem5_r2"
        (output / "status.json").unlink()

        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=AssertionError("unvalidated result must fail closed")):
            with self.assertRaisesRegex(ValueError, "--rerun"):
                run(self.r1, self.fixed / "r2_latency.json", output)

    def test_explicit_rerun_bypasses_malformed_cached_status(self):
        """The existing --rerun contract replaces cache artifacts without reading status."""
        from workflow.r2.run_r2 import run

        output = self.fixed / "gem5_r2"
        (output / "status.json").write_text("{malformed", encoding="utf-8")

        def complete_gem5(command, **_kwargs):
            (output / "stats.txt").write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.cpu0.commitStats0.numInsts 162500000\n"
                "system.cpu0.numCycles 200000000\n"
                "system.cpu1.commitStats0.numInsts 162500000\n"
                "system.cpu1.numCycles 200000000\n"
                "system.cpu2.commitStats0.numInsts 162500000\n"
                "system.cpu2.numCycles 200000000\n"
                "system.cpu3.commitStats0.numInsts 162500000\n"
                "system.cpu3.numCycles 200000000\n",
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, stdout="gem5 rerun fixture completed\n")

        with patch("workflow.r2.run_r2.subprocess.run", side_effect=complete_gem5):
            result = run(
                self.r1, self.fixed / "r2_latency.json", output, rerun=True
            )

        self.assertEqual(result["ipc2"], 3.25)
        self.assertEqual(read_json(output / "status.json")["state"], "success")

    def test_successful_rerun_rejects_stale_stats_when_gem5_writes_none(self):
        """A zero gem5 exit cannot bind old stats to a newly requested vector."""
        from workflow.r2.run_r2 import run

        output = self.fixed / "gem5_r2"
        vector = read_json(self.fixed / "r2_latency.json")
        vector["gem5_overrides"]["xbar_forward_latency"] = 9
        vector["gem5_args"][21] = "9"
        write_json(self.fixed / "r2_latency.json", vector)

        with patch("workflow.r2.run_r2.subprocess.run", return_value=CompletedProcess(
                ["gem5.opt"], 0, stdout="gem5 returned without stats\n")):
            with self.assertRaisesRegex(ValueError, "fresh stats"):
                run(self.r1, self.fixed / "r2_latency.json", output, rerun=True)

        self.assertEqual(read_json(output / "status.json")["state"], "failed")

    def test_local_all_core_cache_enforces_recorded_measurement_minimum(self):
        """A resumable cache must satisfy the same all-core window as a fresh run."""
        from workflow.r2.run_r2 import validate_local_result

        metadata = read_json(self.r1 / "r1_metadata.json")
        metadata["instruction_window_scope"] = "all-cores"
        write_json(self.r1 / "r1_metadata.json", metadata)
        self._write_source_result()

        decision = validate_local_result(
            self.r1, self.fixed / "r2_latency.json", self.fixed / "gem5_r2"
        )

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("measurement minimum" in reason
                            for reason in decision["reasons"]), decision["reasons"])

    def test_reuse_rejects_missing_source_r2_stats(self):
        """Result JSON alone cannot prove IPC2 when its measured gem5 stats disappeared."""
        from workflow.r2.reuse_result import validate_reuse

        (self.fixed / "gem5_r2/stats.txt").unlink()

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("R2 stats" in reason for reason in decision["reasons"]),
                        decision["reasons"])

    def test_reuse_rejects_result_ipc_that_disagrees_with_source_r2_stats(self):
        """Rehashing edited result JSON cannot make an unmeasured IPC2 auditable."""
        from workflow.r2.reuse_result import validate_reuse

        self._rewrite_result(lambda result: result.update(ipc2=9.0))
        status = read_json(self.fixed / "gem5_r2/status.json")
        status["ipc2"] = 9.0
        write_json(self.fixed / "gem5_r2/status.json", status)

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("IPC2 differs" in reason for reason in decision["reasons"]),
                        decision["reasons"])

    def test_new_r2_result_and_status_record_schema3_provenance(self):
        """A newly measured cache must persist every identity used by resume validation."""
        from workflow.r2.run_r2 import run, validate_local_result

        output = self.clip / "fresh_gem5_r2"

        def complete_gem5(_command, **_kwargs):
            (output / "stats.txt").write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.cpu0.commitStats0.numInsts 500000000\n"
                "system.cpu0.numCycles 200000000\n"
                "system.cpu1.commitStats0.numInsts 10\n"
                "system.cpu1.numCycles 200000000\n"
                "system.cpu2.commitStats0.numInsts 10\n"
                "system.cpu2.numCycles 200000000\n"
                "system.cpu3.commitStats0.numInsts 10\n"
                "system.cpu3.numCycles 200000000\n",
                encoding="utf-8",
            )
            return CompletedProcess(_command, 0, stdout="gem5 fixture completed\n")

        with patch("workflow.r2.run_r2.subprocess.run", side_effect=complete_gem5):
            result = run(self.r1, self.clip / "r2_latency.json", output)

        status = read_json(output / "status.json")
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(status["schema_version"], 3)
        for key, value in self._identity(self.clip / "r2_latency.json").items():
            self.assertEqual(result[key], value)
            self.assertEqual(status[key], value)
        self.assertEqual(status["r2_result_sha256"], sha256_file(output / "r2_result.json"))
        self.assertEqual(result["stats_sha256"], sha256_file(output / "stats.txt"))
        self.assertEqual(status["stats_sha256"], sha256_file(output / "stats.txt"))
        self.assertTrue(validate_local_result(
            self.r1, self.clip / "r2_latency.json", output
        )["accepted"])

    def test_fresh_r2_rejects_fractional_instruction_and_cycle_counters(self):
        """Fresh measurement must not truncate non-integral gem5 counters."""
        from workflow.r2.run_r2 import run

        for label, counter, value in (
                ("instructions", "system.cpu1.commitStats0.numInsts", "10.5"),
                ("cycles", "system.cpu1.numCycles", "200000000.5")):
            with self.subTest(label=label):
                output = self.clip / f"fractional-{label}"

                def complete_gem5(command, **_kwargs):
                    records = {
                        f"system.cpu{core}.commitStats0.numInsts": "10"
                        for core in range(4)
                    }
                    records.update({
                        f"system.cpu{core}.numCycles": "200000000"
                        for core in range(4)
                    })
                    records["system.cpu0.commitStats0.numInsts"] = "500000000"
                    records[counter] = value
                    text = "---------- Begin Simulation Statistics ----------\n" + "".join(
                        f"{name} {number}\n" for name, number in records.items()
                    )
                    (output / "stats.txt").write_text(text, encoding="utf-8")
                    return CompletedProcess(command, 0, stdout="fractional fixture\n")

                with patch("workflow.r2.run_r2.subprocess.run",
                           side_effect=complete_gem5):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        run(self.r1, self.clip / "r2_latency.json", output)

    def test_fresh_r2_rejects_stats_replaced_between_parse_and_hash(self):
        """A fresh result must parse and hash one stable gem5 stats snapshot."""
        import workflow.r2.run_r2 as run_r2

        output = self.clip / "replaced-stats"
        stats_path = output / "stats.txt"

        def complete_gem5(command, **_kwargs):
            stats_path.write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.cpu0.commitStats0.numInsts 500000000\n"
                "system.cpu0.numCycles 200000000\n"
                "system.cpu1.commitStats0.numInsts 10\n"
                "system.cpu1.numCycles 200000000\n"
                "system.cpu2.commitStats0.numInsts 10\n"
                "system.cpu2.numCycles 200000000\n"
                "system.cpu3.commitStats0.numInsts 10\n"
                "system.cpu3.numCycles 200000000\n",
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, stdout="replace fixture\n")

        original_parse = run_r2.parse_gem5_stats
        original_read_bytes = Path.read_bytes
        replaced = False

        def replace_stats() -> None:
            nonlocal replaced
            if not replaced:
                replaced = True
                stats_path.write_text(
                    stats_path.read_text(encoding="utf-8").replace(
                        "system.cpu1.commitStats0.numInsts 10",
                        "system.cpu1.commitStats0.numInsts 999",
                    ),
                    encoding="utf-8",
                )

        def replace_after_parse(path, *args, **kwargs):
            value = original_parse(path, *args, **kwargs)
            if Path(path).resolve() == stats_path.resolve():
                replace_stats()
            return value

        def replace_after_read_bytes(path):
            value = original_read_bytes(path)
            if Path(path).resolve() == stats_path.resolve():
                replace_stats()
            return value

        with patch("workflow.r2.run_r2.subprocess.run", side_effect=complete_gem5), \
                patch("workflow.r2.run_r2.parse_gem5_stats",
                      side_effect=replace_after_parse), \
                patch.object(Path, "read_bytes", replace_after_read_bytes):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                run_r2.run(self.r1, self.clip / "r2_latency.json", output)

    def test_fresh_r2_rechecks_stats_after_result_publication(self):
        """Successful status must not commit stats changed while writing the result."""
        import workflow.r2.run_r2 as run_r2

        output = self.clip / "stats-changed-during-result"
        stats_path = output / "stats.txt"
        result_path = output / "r2_result.json"

        def complete_gem5(command, **_kwargs):
            stats_path.write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.cpu0.commitStats0.numInsts 500000000\n"
                "system.cpu0.numCycles 200000000\n"
                "system.cpu1.commitStats0.numInsts 10\n"
                "system.cpu1.numCycles 200000000\n"
                "system.cpu2.commitStats0.numInsts 10\n"
                "system.cpu2.numCycles 200000000\n"
                "system.cpu3.commitStats0.numInsts 10\n"
                "system.cpu3.numCycles 200000000\n",
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, stdout="result race fixture\n")

        real_publish = run_r2.write_json
        replaced = False

        def replace_stats_after_result(path, value):
            nonlocal replaced
            result = real_publish(path, value)
            if Path(path).resolve() == result_path.resolve() and not replaced:
                replaced = True
                stats_path.write_text(
                    stats_path.read_text(encoding="utf-8").replace(
                        "system.cpu1.commitStats0.numInsts 10",
                        "system.cpu1.commitStats0.numInsts 999",
                    ),
                    encoding="utf-8",
                )
            return result

        with patch("workflow.r2.run_r2.subprocess.run", side_effect=complete_gem5), \
                patch("workflow.r2.run_r2.write_json",
                      side_effect=replace_stats_after_result):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                run_r2.run(self.r1, self.clip / "r2_latency.json", output)

        self.assertEqual(read_json(output / "status.json")["state"], "failed")

    def test_local_cache_rejects_tampered_latency_command_segment(self):
        """Result/status hashes cannot legitimize a command with changed override values."""
        from workflow.r2.run_r2 import validate_local_result

        self._rewrite_result(
            lambda result: result["command"].__setitem__(-1, "99")
        )

        decision = validate_local_result(
            self.r1, self.fixed / "r2_latency.json", self.fixed / "gem5_r2"
        )

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("command" in reason for reason in decision["reasons"]),
                        decision["reasons"])

    def test_local_cache_rejects_success_status_with_nonzero_return_code(self):
        """A state label cannot override a recorded nonzero gem5 return code."""
        from workflow.r2.run_r2 import validate_local_result

        status = read_json(self.fixed / "gem5_r2/status.json")
        status["return_code"] = 7
        write_json(self.fixed / "gem5_r2/status.json", status)

        decision = validate_local_result(
            self.r1, self.fixed / "r2_latency.json", self.fixed / "gem5_r2"
        )

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("return_code" in reason for reason in decision["reasons"]),
                        decision["reasons"])

    def test_reuse_rejects_one_different_override(self):
        """Matching aggregate latency cannot hide one changed gem5 override."""
        from workflow.r2.reuse_result import validate_reuse

        vector = read_json(self.clip / "r2_latency.json")
        vector["gem5_overrides"]["xbar_forward_latency"] = 9
        vector["gem5_args"][21] = "9"
        write_json(self.clip / "r2_latency.json", vector)

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("gem5_overrides" in reason for reason in decision["reasons"]))

    def test_reuse_rejects_noncanonical_gem5_args_even_when_overrides_match(self):
        """Stored args must be the complete ordered rendering of their own overrides."""
        from workflow.r2.reuse_result import validate_reuse

        for point in (self.fixed, self.clip):
            with self.subTest(point=point.name):
                vector = read_json(point / "r2_latency.json")
                malformed = deepcopy(vector)
                malformed["gem5_args"][-1] = "99"
                write_json(point / "r2_latency.json", malformed)

                decision = validate_reuse(
                    self.fixed, self.clip, self.r1, self.config_path
                )

                self.assertFalse(decision["accepted"])
                self.assertTrue(any("gem5_args" in reason for reason in decision["reasons"]))
                write_json(point / "r2_latency.json", vector)

    def test_reuse_rejects_each_identity_and_source_provenance_mismatch(self):
        """No architecture, R1, window, config, status, or result mismatch may reuse IPC2."""
        from workflow.r2.reuse_result import validate_reuse

        def architecture_mismatch():
            summary = read_json(self.clip / "pipeline_summary.json")
            summary["l2_size"] = "1024kB"
            write_json(self.clip / "pipeline_summary.json", summary)

        def r1_directory_mismatch():
            summary = read_json(self.clip / "pipeline_summary.json")
            summary["r1"] = str((self.root / "other-r1").resolve())
            write_json(self.clip / "pipeline_summary.json", summary)

        def scope_mismatch():
            self._rewrite_result(
                lambda result: result.update(instruction_window_scope="all-cores")
            )

        def config_mismatch():
            run_config = read_json(self.clip / "run_config.json")
            run_config["config"]["physical"]["r_convec_k_per_w"] = 4.0
            write_json(self.clip / "run_config.json", run_config)

        def status_mismatch():
            status = read_json(self.fixed / "gem5_r2/status.json")
            status["state"] = "failed"
            write_json(self.fixed / "gem5_r2/status.json", status)

        def result_provenance_mismatch():
            self._rewrite_result(
                lambda result: result.update(latency_vector=str(
                    (self.root / "stale/r2_latency.json").resolve()
                ))
            )

        for label, mutate, expected in (
                ("architecture", architecture_mismatch, "architecture"),
                ("r1-directory", r1_directory_mismatch, "R1 directory"),
                ("instruction-scope", scope_mismatch, "instruction_window_scope"),
                ("config", config_mismatch, "embedded config"),
                ("status", status_mismatch, "status"),
                ("result-provenance", result_provenance_mismatch, "latency_vector")):
            with self.subTest(label=label):
                self._write_point(self.fixed, "fixed-bin", 1.4)
                self._write_point(self.clip, "clip3d", 1.1)
                self._write_source_result()
                mutate()

                decision = validate_reuse(
                    self.fixed, self.clip, self.r1, self.config_path
                )

                self.assertFalse(decision["accepted"])
                self.assertTrue(any(expected in reason for reason in decision["reasons"]),
                                decision["reasons"])

    def test_reuse_rejects_mutated_target_modules_and_thermal_inputs(self):
        """Target architecture, gamma, and thermal identity must be validated inputs."""
        from workflow.r2.reuse_result import validate_reuse

        def architecture_mismatch():
            modules = read_json(self.clip / "modules.json")
            modules["architecture"]["l2_size"] = "1024kB"
            write_json(self.clip / "modules.json", modules)

        def gamma_mismatch():
            modules = read_json(self.clip / "modules.json")
            modules["gamma"] = 0.35
            write_json(self.clip / "modules.json", modules)

        def thermal_mismatch():
            thermal = read_json(self.clip / "hotspot/thermal_result.json")
            thermal["ambient_c"] = 30.0
            write_json(self.clip / "hotspot/thermal_result.json", thermal)

        def thermal_path_mismatch():
            thermal = read_json(self.clip / "hotspot/thermal_result.json")
            thermal["power_trace"] = str(
                (self.fixed / "hotspot/power.ptrace").resolve()
            )
            write_json(self.clip / "hotspot/thermal_result.json", thermal)

        def thermal_temperature_mismatch():
            thermal = read_json(self.clip / "hotspot/thermal_result.json")
            thermal["tmax_c"] = 80.0
            thermal["tmax_k"] = 353.15
            write_json(self.clip / "hotspot/thermal_result.json", thermal)

        for label, mutate, expected in (
                ("architecture", architecture_mismatch, "modules architecture"),
                ("gamma", gamma_mismatch, "gamma"),
                ("thermal", thermal_mismatch, "thermal ambient"),
                ("thermal-path", thermal_path_mismatch, "power_trace"),
                ("thermal-temperature", thermal_temperature_mismatch,
                 "thermal tmax_c")):
            with self.subTest(label=label):
                self._write_point(self.clip, "clip3d", 1.1)
                mutate()

                decision = validate_reuse(
                    self.fixed, self.clip, self.r1, self.config_path
                )

                self.assertFalse(decision["accepted"])
                self.assertTrue(any(expected in reason for reason in decision["reasons"]),
                                decision["reasons"])

    def test_reuse_rejects_incoherent_target_performance_and_summary(self):
        """Every non-R2 performance field must derive from one target input set."""
        from workflow.r2.reuse_result import validate_reuse

        def change_modules_ipc1():
            modules = read_json(self.clip / "modules.json")
            modules["ipc1"] = 3.0
            write_json(self.clip / "modules.json", modules)

        def change_performance(field, value):
            def mutate():
                performance = read_json(self.clip / "performance.json")
                performance[field] = value
                write_json(self.clip / "performance.json", performance)
            return mutate

        def change_summary(field, value):
            def mutate():
                summary = read_json(self.clip / "pipeline_summary.json")
                summary[field] = value
                write_json(self.clip / "pipeline_summary.json", summary)
            return mutate

        cases = [("modules-ipc1", change_modules_ipc1, "performance ipc1")]
        cases.extend(
            (f"performance-{field}", change_performance(field, value), field)
            for field, value in (
                ("schema_version", 2),
                ("equation", 12),
                ("gamma", 0.3),
                ("leakage_power_w", 21.0),
                ("dynamic_power_w", 81.0),
                ("tmax_f0_c", 130.0),
                ("ambient_c", 30.0),
                ("tsafe_c", 90.0),
                ("f0_ghz", 1.8),
                ("fmin_ghz", 0.5),
                ("unclamped_solution_ghz", 1.2),
                ("sustainable_frequency_ghz", 1.2),
                ("estimated_tmax_at_fmin_c", 70.0),
                ("thermal_feasible_at_fmin", False),
                ("floor_power_scale", 0.4),
                ("state", "thermal_headroom"),
                ("ipc1", 3.0),
                ("bips1_thermal", 3.0),
            )
        )
        cases.extend(
            (f"summary-{field}", change_summary(field, value), field)
            for field, value in (
                ("gamma", 0.9),
                ("sustainable_frequency_ghz", 1.2),
                ("ipc1", 3.0),
                ("bips1_thermal", 3.0),
            )
        )

        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self._write_point(self.clip, "clip3d", 1.1)
                mutate()

                decision = validate_reuse(
                    self.fixed, self.clip, self.r1, self.config_path
                )

                self.assertFalse(decision["accepted"])
                self.assertTrue(any(expected in reason for reason in decision["reasons"]),
                                decision["reasons"])

    def test_reuse_rejects_input_replaced_during_validation_snapshot(self):
        """Parsed vector bytes and recorded hashes must come from one stable snapshot."""
        import workflow.r2.reuse_result as reuse_result

        original_read = reuse_result._read_object
        replaced = False

        def replace_after_read(path, reasons, label, *args, **kwargs):
            nonlocal replaced
            value = original_read(path, reasons, label, *args, **kwargs)
            if label == "target latency vector" and not replaced:
                replaced = True
                changed = read_json(path)
                changed["critical_l1d_to_l2_cycles"] = 999
                write_json(path, changed)
            return value

        with patch("workflow.r2.reuse_result._read_object",
                   side_effect=replace_after_read):
            decision = reuse_result.validate_reuse(
                self.fixed, self.clip, self.r1, self.config_path
            )

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("changed during validation" in reason
                            for reason in decision["reasons"]), decision["reasons"])

    def test_attach_evaluation_failure_leaves_no_accepted_or_partial_outputs(self):
        """Evaluator failure must preserve authoritative target files and no marker."""
        from workflow.r2.reuse_result import attach_reused_result

        performance_before = (self.clip / "performance.json").read_bytes()
        summary_before = (self.clip / "pipeline_summary.json").read_bytes()

        with patch("workflow.r2.reuse_result.evaluate",
                   side_effect=RuntimeError("injected evaluator failure")):
            with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual((self.clip / "performance.json").read_bytes(), performance_before)
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_attach_summary_publish_failure_rolls_back_performance_and_marker(self):
        """A failed second publication cannot expose half-attached target state."""
        import workflow.r2.reuse_result as reuse_result

        performance_before = (self.clip / "performance.json").read_bytes()
        summary_before = (self.clip / "pipeline_summary.json").read_bytes()
        real_publish = reuse_result.write_json
        failed = False

        def fail_summary_once(path, value):
            nonlocal failed
            if Path(path).resolve() == (self.clip / "pipeline_summary.json").resolve() \
                    and not failed:
                failed = True
                raise OSError("injected summary publication failure")
            return real_publish(path, value)

        with patch("workflow.r2.reuse_result.write_json", side_effect=fail_summary_once):
            with self.assertRaisesRegex(OSError, "summary publication"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual((self.clip / "performance.json").read_bytes(), performance_before)
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_rollback_preserves_summary_replaced_before_this_attempt_publishes_it(self):
        """A performance write does not grant rollback ownership of the summary."""
        import workflow.r2.reuse_result as reuse_result

        performance_before = (self.clip / "performance.json").read_bytes()
        concurrent_summary = {"concurrent": "summary-before-publication"}
        real_publish = reuse_result.write_json
        interrupted = False

        def replace_summary_then_fail(path, value):
            nonlocal interrupted
            if Path(path).resolve() == (self.clip / "pipeline_summary.json").resolve() \
                    and not interrupted:
                interrupted = True
                real_publish(path, concurrent_summary)
                raise OSError("injected pre-summary publication failure")
            return real_publish(path, value)

        with patch("workflow.r2.reuse_result.write_json",
                   side_effect=replace_summary_then_fail):
            with self.assertRaisesRegex(OSError, "pre-summary publication"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual((self.clip / "performance.json").read_bytes(), performance_before)
        self.assertEqual(
            read_json(self.clip / "pipeline_summary.json"), concurrent_summary
        )

    def test_attach_rechecks_inputs_immediately_before_acceptance_marker(self):
        """An input changed during output publication must abort the final commit marker."""
        import workflow.r2.reuse_result as reuse_result

        performance_before = (self.clip / "performance.json").read_bytes()
        summary_before = (self.clip / "pipeline_summary.json").read_bytes()
        real_publish = reuse_result.write_json
        changed = False

        def change_input_after_summary(path, value):
            nonlocal changed
            result = real_publish(path, value)
            if Path(path).resolve() == (self.clip / "pipeline_summary.json").resolve() \
                    and not changed:
                changed = True
                vector = read_json(self.clip / "r2_latency.json")
                vector["critical_l1d_to_l2_cycles"] = 999
                real_publish(self.clip / "r2_latency.json", vector)
            return result

        with patch("workflow.r2.reuse_result.write_json",
                   side_effect=change_input_after_summary):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual((self.clip / "performance.json").read_bytes(), performance_before)
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_attach_rejects_outputs_replaced_before_acceptance_marker(self):
        """The marker must bind the exact performance and summary just published."""
        import workflow.r2.reuse_result as reuse_result

        real_publish = reuse_result.write_json
        replaced = False

        def replace_outputs_after_summary(path, value):
            nonlocal replaced
            result = real_publish(path, value)
            if Path(path).resolve() == (self.clip / "pipeline_summary.json").resolve() \
                    and not replaced:
                replaced = True
                real_publish(self.clip / "performance.json", {"tampered": True})
                real_publish(self.clip / "pipeline_summary.json", {"tampered": True})
            return result

        with patch("workflow.r2.reuse_result.write_json",
                   side_effect=replace_outputs_after_summary):
            with self.assertRaisesRegex(OSError, "published outputs changed"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual(read_json(self.clip / "performance.json"), {"tampered": True})
        self.assertEqual(
            read_json(self.clip / "pipeline_summary.json"), {"tampered": True}
        )

    def test_attach_checks_outputs_after_final_immutable_scan(self):
        """Output replacement during the final scan must precede the last hash check."""
        import workflow.r2.reuse_result as reuse_result

        original_scan = reuse_result._record_changed_snapshots
        replaced = False

        def replace_outputs_after_scan(snapshots, reasons, ignored=None):
            nonlocal replaced
            original_scan(snapshots, reasons, ignored)
            if ignored and not replaced:
                replaced = True
                write_json(self.clip / "performance.json", {"tampered": True})
                write_json(self.clip / "pipeline_summary.json", {"tampered": True})

        with patch("workflow.r2.reuse_result._record_changed_snapshots",
                   side_effect=replace_outputs_after_scan):
            with self.assertRaisesRegex(OSError, "published outputs changed"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual(read_json(self.clip / "performance.json"), {"tampered": True})
        self.assertEqual(
            read_json(self.clip / "pipeline_summary.json"), {"tampered": True}
        )

    def test_rollback_restores_owned_summary_but_preserves_replaced_performance(self):
        """Per-output CAS restores only bytes still owned by this attempt."""
        import workflow.r2.reuse_result as reuse_result

        summary_before = (self.clip / "pipeline_summary.json").read_bytes()
        concurrent_performance = {"concurrent": "performance-after-publication"}
        original_scan = reuse_result._record_changed_snapshots
        replaced = False

        def replace_performance_after_scan(snapshots, reasons, ignored=None):
            nonlocal replaced
            original_scan(snapshots, reasons, ignored)
            if ignored and not replaced:
                replaced = True
                write_json(self.clip / "performance.json", concurrent_performance)

        with patch("workflow.r2.reuse_result._record_changed_snapshots",
                   side_effect=replace_performance_after_scan):
            with self.assertRaisesRegex(OSError, "published outputs changed"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual(
            read_json(self.clip / "performance.json"), concurrent_performance
        )
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_concurrent_attachments_serialize_validation_through_rollback(self):
        """One point cannot admit another attachment inside its transaction."""
        import threading
        import workflow.r2.reuse_result as reuse_result

        real_evaluate = reuse_result.evaluate
        first_entered = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        call_guard = threading.Lock()
        errors = []
        call_count = 0

        def controlled_evaluate(*args, **kwargs):
            nonlocal call_count
            with call_guard:
                call_count += 1
                call_number = call_count
            if call_number == 1:
                first_entered.set()
                if not release_first.wait(5):
                    raise RuntimeError("timed out waiting to release first attachment")
            else:
                second_entered.set()
            return real_evaluate(*args, **kwargs)

        def attach(started=None):
            if started is not None:
                started.set()
            try:
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )
            except BaseException as error:
                errors.append(error)

        with patch("workflow.r2.reuse_result.evaluate",
                   side_effect=controlled_evaluate):
            first = threading.Thread(target=attach)
            second = threading.Thread(target=attach, args=(second_started,))
            first.start()
            self.assertTrue(first_entered.wait(2))
            second.start()
            self.assertTrue(second_started.wait(2))
            overlapped = second_entered.wait(0.5)
            release_first.set()
            first.join(5)
            second.join(5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(overlapped)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(errors, [])

    def test_evaluator_failure_does_not_clobber_concurrent_authoritative_update(self):
        """Rollback must not restore outputs this attempt never published."""
        import workflow.r2.reuse_result as reuse_result

        concurrent_performance = {"concurrent": "performance"}
        concurrent_summary = {"concurrent": "summary"}

        def update_then_fail(*_args, **_kwargs):
            write_json(self.clip / "performance.json", concurrent_performance)
            write_json(self.clip / "pipeline_summary.json", concurrent_summary)
            raise RuntimeError("injected concurrent evaluator failure")

        with patch("workflow.r2.reuse_result.evaluate", side_effect=update_then_fail):
            with self.assertRaisesRegex(RuntimeError, "concurrent evaluator failure"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual(read_json(self.clip / "performance.json"), concurrent_performance)
        self.assertEqual(read_json(self.clip / "pipeline_summary.json"), concurrent_summary)

    def test_failed_reattach_preserves_previous_complete_attachment(self):
        """A failed retry must not orphan an already valid committed attachment."""
        import workflow.r2.reuse_result as reuse_result

        reuse_result.attach_reused_result(
            self.fixed, self.clip, self.r1, self.config_path
        )
        artifact_before = (self.clip / "r2_reuse.json").read_bytes()
        performance_before = (self.clip / "performance.json").read_bytes()
        summary_before = (self.clip / "pipeline_summary.json").read_bytes()

        with patch("workflow.r2.reuse_result.evaluate",
                   side_effect=RuntimeError("injected retry failure")):
            with self.assertRaisesRegex(RuntimeError, "injected retry failure"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertEqual((self.clip / "r2_reuse.json").read_bytes(), artifact_before)
        self.assertEqual((self.clip / "performance.json").read_bytes(), performance_before)
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_failed_reattach_does_not_restore_marker_for_tampered_summary(self):
        """A prior marker is reusable only when it binds both current outputs."""
        import workflow.r2.reuse_result as reuse_result

        reuse_result.attach_reused_result(
            self.fixed, self.clip, self.r1, self.config_path
        )
        summary = read_json(self.clip / "pipeline_summary.json")
        summary["bips2"] += 1.0
        write_json(self.clip / "pipeline_summary.json", summary)
        summary_before = (self.clip / "pipeline_summary.json").read_bytes()

        with patch("workflow.r2.reuse_result.evaluate",
                   side_effect=RuntimeError("injected retry failure")):
            with self.assertRaisesRegex(RuntimeError, "injected retry failure"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())
        self.assertEqual((self.clip / "pipeline_summary.json").read_bytes(), summary_before)

    def test_failed_reattach_does_not_restore_marker_for_new_source_provenance(self):
        """A prior marker cannot certify a different newly validated source R2."""
        import workflow.r2.reuse_result as reuse_result

        reuse_result.attach_reused_result(
            self.fixed, self.clip, self.r1, self.config_path
        )
        output = self.fixed / "gem5_r2"
        stats_path = output / "stats.txt"
        stats_path.write_text(
            "---------- Begin Simulation Statistics ----------\n" + "".join(
                f"system.cpu{core}.commitStats0.numInsts 150000000\n"
                f"system.cpu{core}.numCycles 200000000\n"
                for core in range(4)
            ),
            encoding="utf-8",
        )
        result_path = output / "r2_result.json"
        result = read_json(result_path)
        result["ipc2"] = 3.0
        result["per_core"] = [
            {"core": core, "instructions": 150000000,
             "cycles": 200000000, "ipc": 0.75}
            for core in range(4)
        ]
        result["stats_sha256"] = sha256_file(stats_path)
        write_json(result_path, result)
        status_path = output / "status.json"
        status = read_json(status_path)
        status["ipc2"] = 3.0
        status["stats_sha256"] = sha256_file(stats_path)
        status["r2_result_sha256"] = sha256_file(result_path)
        write_json(status_path, status)
        fixed_summary = read_json(self.fixed / "pipeline_summary.json")
        fixed_summary["ipc2"] = 3.0
        fixed_summary["bips2"] = 4.2
        write_json(self.fixed / "pipeline_summary.json", fixed_summary)
        self.assertTrue(reuse_result.validate_reuse(
            self.fixed, self.clip, self.r1, self.config_path
        )["accepted"])

        with patch("workflow.r2.reuse_result.evaluate",
                   side_effect=RuntimeError("injected new-source retry failure")):
            with self.assertRaisesRegex(RuntimeError, "new-source retry failure"):
                reuse_result.attach_reused_result(
                    self.fixed, self.clip, self.r1, self.config_path
                )

        self.assertFalse((self.clip / "r2_reuse.json").exists())

    def test_exact_reuse_writes_provenance_and_recomputes_clip_performance(self):
        """An accepted reuse attaches source IPC2 without fabricating a target gem5 run."""
        from workflow.r2.reuse_result import attach_reused_result, validate_reuse

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)
        self.assertTrue(decision["accepted"], decision["reasons"])

        summary = attach_reused_result(
            self.fixed, self.clip, self.r1, self.config_path
        )

        source_result = self.fixed / "gem5_r2/r2_result.json"
        artifact_path = self.clip / "r2_reuse.json"
        artifact = read_json(artifact_path)
        performance = read_json(self.clip / "performance.json")
        self.assertEqual(summary["ipc2"], 3.25)
        self.assertAlmostEqual(summary["bips2"], 3.25 * 1.1)
        self.assertEqual(summary["r2_source"], str(source_result.resolve()))
        self.assertTrue(summary["r2_reused"])
        self.assertEqual(summary["r2_reuse_artifact"], str(artifact_path.resolve()))
        self.assertNotIn("gem5_r2", summary["stage_seconds"])
        self.assertFalse((self.clip / "gem5_r2/r2_result.json").exists())
        self.assertEqual(performance["ipc2"], 3.25)
        self.assertAlmostEqual(performance["bips2"], 3.25 * 1.1)
        self.assertEqual(artifact["decision"], "accepted")
        self.assertEqual(artifact["source_result"]["path"], str(source_result.resolve()))
        self.assertEqual(artifact["source_result"]["sha256"], sha256_file(source_result))
        self.assertEqual(artifact["source_stats"]["path"],
                         str((self.fixed / "gem5_r2/stats.txt").resolve()))
        self.assertEqual(artifact["source_stats"]["sha256"],
                         sha256_file(self.fixed / "gem5_r2/stats.txt"))
        self.assertEqual(artifact["source_ipc2"], 3.25)
        self.assertEqual(artifact["canonical_architecture_key"], {
            "workload": "fft", "l1d_size": "32kB", "l2_size": "512kB",
        })
        self.assertEqual(artifact["config"]["sha256"], sha256_file(self.config_path))
        self.assertEqual(artifact["r1"]["metadata_sha256"],
                         sha256_file(self.r1 / "r1_metadata.json"))
        self.assertEqual(artifact["source_latency"]["sha256"],
                         sha256_file(self.fixed / "r2_latency.json"))
        self.assertEqual(artifact["target_latency"]["sha256"],
                         sha256_file(self.clip / "r2_latency.json"))
        self.assertEqual(artifact["target_inputs"]["modules"]["sha256"],
                         sha256_file(self.clip / "modules.json"))
        self.assertEqual(artifact["target_inputs"]["thermal"]["sha256"],
                         sha256_file(self.clip / "hotspot/thermal_result.json"))
        self.assertEqual(artifact["target_outputs"]["performance"]["sha256"],
                         sha256_file(self.clip / "performance.json"))
        self.assertEqual(artifact["target_outputs"]["sustainable_frequency_ghz"], 1.1)
        self.assertAlmostEqual(artifact["target_outputs"]["bips2"], 3.25 * 1.1)


class PairedR2RunnerTests(unittest.TestCase):
    """Run real paired state transitions around only the expensive boundaries."""

    def setUp(self) -> None:
        self.fixture = StrictR2ReuseTests(
            "test_local_successful_cache_accepts_matching_latency_and_r1_identity"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.r1_root = self.root / "canonical-r1"
        self.fixed_root = self.root / "fixed"
        self.clip_root = self.root / "clip"
        self.config_path = self.fixture.config_path
        self.status_root = self.root / "paired-status"
        from workflow.r1_catalog import ArchitectureKey
        self.key = ArchitectureKey("fft", "32kB", "512kB")

    @staticmethod
    def _complete_gem5(command, **_kwargs):
        output = Path(next(arg.split("=", 1)[1] for arg in command
                           if arg.startswith("--outdir=")))
        (output / "stats.txt").parent.mkdir(parents=True, exist_ok=True)
        (output / "stats.txt").write_text(
            "---------- Begin Simulation Statistics ----------\n"
            "system.cpu0.commitStats0.numInsts 162500000\n"
            "system.cpu0.numCycles 200000000\n"
            "system.cpu1.commitStats0.numInsts 162500000\n"
            "system.cpu1.numCycles 200000000\n"
            "system.cpu2.commitStats0.numInsts 162500000\n"
            "system.cpu2.numCycles 200000000\n"
            "system.cpu3.commitStats0.numInsts 162500000\n"
            "system.cpu3.numCycles 200000000\n",
            encoding="utf-8",
        )
        return CompletedProcess(command, 0, stdout="gem5 fixture completed\n")

    def _make_vectors_unequal(self) -> None:
        from workflow.r2.run_r2 import canonical_gem5_args
        vector_path = self.fixture.clip / "r2_latency.json"
        vector = read_json(vector_path)
        vector["gem5_overrides"]["xbar_forward_latency"] = 9
        vector["gem5_args"] = canonical_gem5_args(vector["gem5_overrides"])
        write_json(vector_path, vector)

    def _ordered_boundaries(self, events):
        import workflow.r2.attach_result as attach_result
        import workflow.r2.reuse_result as reuse_result
        import workflow.r2.run_r2 as run_r2

        real_run = run_r2.run
        real_attach = attach_result.attach
        real_reuse_attach = reuse_result.attach_reused_result

        def ordered_run(r1_dir, latency_path, output_dir, *args, **kwargs):
            branch = "fixed" if Path(latency_path).parent == self.fixture.fixed else "clip"
            events.append(f"{branch}-run")
            return real_run(r1_dir, latency_path, output_dir, *args, **kwargs)

        def ordered_attach(point_dir):
            branch = "fixed" if Path(point_dir) == self.fixture.fixed else "clip"
            events.append(f"{branch}-attach")
            return real_attach(point_dir)

        def ordered_reuse(*args, **kwargs):
            events.append("reuse-attach")
            return real_reuse_attach(*args, **kwargs)

        return (
            patch("workflow.r2.run_paired_sweep.run_r2.run", side_effect=ordered_run),
            patch("workflow.r2.run_paired_sweep.attach_result.attach",
                  side_effect=ordered_attach),
            patch("workflow.r2.run_paired_sweep.reuse_result.attach_reused_result",
                  side_effect=ordered_reuse),
        )

    def test_equal_vectors_run_fixed_first_and_record_one_physical_run(self):
        """Choosing reuse before fixed attachment would violate pair ordering."""
        from workflow.r2.run_paired_sweep import run_pair

        events = []
        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        with run_patch, attach_patch, reuse_patch:
            result = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "success")
        self.assertEqual(result["physical_r2_runs"], 1)
        self.assertTrue(result["clip3d_reused_fixed_r2"])
        self.assertEqual(events, ["fixed-run", "fixed-attach", "reuse-attach"])
        self.assertEqual(
            result["pair_status"],
            str((self.status_root / self.key.relative_path() /
                 "pair_status.json").resolve()),
        )

    def test_default_status_root_is_a_sibling_of_fixed_root(self):
        """A direct call must not put scheduler state inside a layout root."""
        from workflow.r2.run_paired_sweep import run_pair

        result = run_pair(
            self.key, self.r1_root, self.fixed_root, self.clip_root,
            self.config_path,
        )

        expected = self.fixed_root.parent / "paired_r2_status" / (
            self.key.relative_path()
        ) / "pair_status.json"
        self.assertEqual(result["pair_status"], str(expected.resolve()))
        self.assertTrue(expected.is_file())

    def test_unequal_vectors_run_and_attach_a_separate_clip_result(self):
        """An override mismatch must never be relabeled as fixed-result reuse."""
        from workflow.r2.run_paired_sweep import run_pair

        self._make_vectors_unequal()
        events = []
        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5), \
                run_patch, attach_patch, reuse_patch:
            result = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "success")
        self.assertEqual(result["physical_r2_runs"], 2)
        self.assertFalse(result["clip3d_reused_fixed_r2"])
        self.assertEqual(
            events, ["fixed-run", "fixed-attach", "clip-run", "clip-attach"]
        )

    def test_completed_compatible_pair_is_skipped_on_second_invocation(self):
        """A validated successful pair must not repeat either attachment boundary."""
        from workflow.r2.run_paired_sweep import run_pair

        first = run_pair(
            self.key, self.r1_root, self.fixed_root, self.clip_root,
            self.config_path, status_root=self.status_root,
        )
        with patch("workflow.r2.run_paired_sweep.run_r2.run",
                   side_effect=AssertionError("completed fixed must skip")), \
                patch("workflow.r2.run_paired_sweep.attach_result.attach",
                      side_effect=AssertionError("completed attach must skip")), \
                patch("workflow.r2.run_paired_sweep.reuse_result.attach_reused_result",
                      side_effect=AssertionError("completed reuse must skip")):
            second = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(second, first)

    def test_completed_pair_with_tampered_reuse_marker_is_reattached(self):
        """Marker existence alone must not make a damaged reused pair skippable."""
        from workflow.r2.run_paired_sweep import run_pair

        run_pair(
            self.key, self.r1_root, self.fixed_root, self.clip_root,
            self.config_path, status_root=self.status_root,
        )
        marker_path = self.fixture.clip / "r2_reuse.json"
        marker = read_json(marker_path)
        marker["source_ipc2"] = 99.0
        write_json(marker_path, marker)
        events = []
        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        with run_patch, attach_patch, reuse_patch:
            repaired = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(repaired["state"], "success")
        self.assertEqual(events, ["reuse-attach"])
        self.assertEqual(read_json(marker_path)["source_ipc2"], 3.25)

    def test_fresh_reuse_rejects_a_marker_corrupted_after_attachment(self):
        """A fresh pair cannot succeed before its marker binds validated outputs."""
        from workflow.r2.run_paired_sweep import run_pair
        import workflow.r2.reuse_result as reuse_result

        real_attach = reuse_result.attach_reused_result

        def attach_then_corrupt(*args, **kwargs):
            summary = real_attach(*args, **kwargs)
            marker_path = self.fixture.clip / "r2_reuse.json"
            marker = read_json(marker_path)
            marker["source_ipc2"] = 99.0
            write_json(marker_path, marker)
            return summary

        with patch("workflow.r2.run_paired_sweep.reuse_result.attach_reused_result",
                   side_effect=attach_then_corrupt):
            result = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "failed")
        self.assertIn("reuse attachment", result["error"])

    def test_fresh_reuse_rejects_corrupt_live_outputs_even_if_marker_rehashed(self):
        """Marker claims cannot redefine live reused IPC2/BIPS2 evidence."""
        from workflow.r2.run_paired_sweep import run_pair
        import workflow.r2.reuse_result as reuse_result

        real_attach = reuse_result.attach_reused_result

        def attach_then_forge_marker(*args, **kwargs):
            summary = real_attach(*args, **kwargs)
            performance_path = self.fixture.clip / "performance.json"
            summary_path = self.fixture.clip / "pipeline_summary.json"
            marker_path = self.fixture.clip / "r2_reuse.json"
            performance = read_json(performance_path)
            performance["bips2"] = 999.0
            write_json(performance_path, performance)
            summary = read_json(summary_path)
            summary["bips2"] = 999.0
            write_json(summary_path, summary)
            marker = read_json(marker_path)
            marker["target_outputs"]["performance"]["sha256"] = sha256_file(
                performance_path
            )
            marker["target_outputs"]["summary"]["sha256"] = sha256_file(
                summary_path
            )
            marker["target_outputs"]["bips2"] = 999.0
            write_json(marker_path, marker)
            return summary

        with patch("workflow.r2.run_paired_sweep.reuse_result.attach_reused_result",
                   side_effect=attach_then_forge_marker):
            result = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "failed")
        self.assertIn("reuse attachment", result["error"])

    def test_failed_clip_stage_resumes_after_the_fixed_checkpoint(self):
        """A CLIP interruption must preserve and validate completed fixed work."""
        from workflow.r2.run_paired_sweep import run_pair

        self._make_vectors_unequal()
        events = []
        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        failed_once = False

        import workflow.r2.run_r2 as run_r2
        real_run = run_r2.run

        def fail_first_clip(r1_dir, latency_path, output_dir, *args, **kwargs):
            nonlocal failed_once
            branch = "fixed" if Path(latency_path).parent == self.fixture.fixed else "clip"
            events.append(f"{branch}-run")
            if branch == "clip" and not failed_once:
                failed_once = True
                raise RuntimeError("injected CLIP interruption")
            return real_run(r1_dir, latency_path, output_dir, *args, **kwargs)

        with patch("workflow.r2.run_paired_sweep.run_r2.run",
                   side_effect=fail_first_clip), attach_patch, reuse_patch:
            failed = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )
        self.assertEqual(failed["state"], "failed")
        self.assertTrue(failed["fixed_complete"])
        events.clear()

        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5), \
                run_patch, attach_patch, reuse_patch:
            resumed = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(resumed["state"], "success")
        self.assertEqual(events, ["clip-run", "clip-attach"])

    def test_separate_clip_rejects_incoherent_attached_performance(self):
        """Local provenance cannot certify BIPS2 that disagrees with performance."""
        from workflow.r2.run_paired_sweep import run_pair
        import workflow.r2.attach_result as attach_result

        self._make_vectors_unequal()
        real_attach = attach_result.attach

        def attach_then_corrupt(point_dir):
            summary = real_attach(point_dir)
            if Path(point_dir) == self.fixture.clip:
                performance_path = Path(point_dir) / "performance.json"
                performance = read_json(performance_path)
                performance["bips2"] = 999.0
                write_json(performance_path, performance)
            return summary

        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5), \
                patch("workflow.r2.run_paired_sweep.attach_result.attach",
                      side_effect=attach_then_corrupt):
            result = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "failed")
        self.assertIn("local attachment", result["error"])

    def test_completed_separate_clip_with_corrupt_performance_is_reattached(self):
        """Resume validation must bind local performance before skipping a pair."""
        from workflow.r2.run_paired_sweep import run_pair

        self._make_vectors_unequal()
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5):
            first = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )
        self.assertEqual(first["state"], "success")
        performance_path = self.fixture.clip / "performance.json"
        performance = read_json(performance_path)
        performance["bips2"] = 999.0
        write_json(performance_path, performance)

        events = []
        run_patch, attach_patch, reuse_patch = self._ordered_boundaries(events)
        with run_patch, attach_patch, reuse_patch:
            repaired = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(repaired["state"], "success")
        self.assertEqual(events, ["clip-run", "clip-attach"])
        self.assertAlmostEqual(read_json(performance_path)["bips2"], 3.575)

    def test_sweep_retains_a_failed_pair_while_another_completes(self):
        """One returned failure must not cancel or erase a completed pair."""
        from concurrent.futures import Future
        from workflow.experiments.balanced50 import load_selection, selection_keys
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        keys = selection_keys(load_selection(selection_path))[:2]
        status_root = self.status_root

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _function, key, *_args, **_kwargs):
                future = Future()
                state = "failed" if key == keys[0] else "success"
                future.set_result({
                    "schema_version": 1,
                    "state": state,
                    "key": {
                        "workload": key.workload,
                        "l1d_size": key.l1d_size,
                        "l2_size": key.l2_size,
                    },
                    "clip3d_reused_fixed_r2": state == "success",
                    "physical_r2_runs": 1 if state == "success" else 0,
                    "pair_status": str(
                        status_root / key.relative_path() / "pair_status.json"
                    ),
                })
                return future

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value={"mode": "resume"}):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root, selection_path,
                self.config_path, self.status_root, jobs=2, limit=2,
            )

        self.assertEqual(result["selected_pair_count"], 2)
        self.assertEqual(result["completed_pair_count"], 1)
        self.assertEqual(result["failed_pair_count"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual([pair["state"] for pair in result["pairs"]],
                         ["failed", "success"])

    def test_future_exception_is_retained_and_status_rewrites_after_each_result(self):
        """A crashed worker must become one failed pair without hiding progress."""
        from concurrent.futures import Future
        from workflow.experiments.balanced50 import load_selection, selection_keys
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        keys = selection_keys(load_selection(selection_path))[:2]
        status_root = self.status_root

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _function, key, *_args, **_kwargs):
                future = Future()
                if key == keys[0]:
                    future.set_exception(RuntimeError("worker process crashed"))
                else:
                    future.set_result({
                        "schema_version": 1,
                        "state": "success",
                        "key": {"workload": key.workload,
                                "l1d_size": key.l1d_size,
                                "l2_size": key.l2_size},
                        "clip3d_reused_fixed_r2": True,
                        "physical_r2_runs": 1,
                        "pair_status": str(
                            status_root / key.relative_path() / "pair_status.json"
                        ),
                    })
                return future

        real_write = paired.write_json
        experiment_snapshots = []

        def capture_status(path, value):
            real_write(path, value)
            if Path(path).resolve() == (status_root / "status.json").resolve():
                experiment_snapshots.append(deepcopy(value))

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value={"mode": "resume"}), \
                patch.object(paired, "write_json", side_effect=capture_status), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root, selection_path,
                self.config_path, status_root, jobs=2, limit=2,
            )

        failed_path = status_root / keys[0].relative_path() / "pair_status.json"
        self.assertIn("worker process crashed", read_json(failed_path)["error"])
        self.assertEqual(result["completed_pair_count"], 1)
        self.assertEqual(result["failed_pair_count"], 1)
        self.assertEqual(len(experiment_snapshots), 4)
        self.assertEqual(
            [snapshot["completed_pair_count"] + snapshot["failed_pair_count"]
             for snapshot in experiment_snapshots],
            [0, 1, 2, 2],
        )

    def test_existing_r2_adapter_uses_task5_local_and_reuse_decisions(self):
        """Resume preflight must not invent a weaker scheduler provenance path."""
        import workflow.r2.run_paired_sweep as paired
        from workflow.r2.attach_result import attach
        from workflow.r2.reuse_result import attach_reused_result
        from workflow.r2.run_r2 import run

        adapter = paired._existing_r2_validator(
            self.r1_root, self.fixed_root, self.clip_root, self.config_path
        )
        fixed = adapter("fixed-bin", self.key, self.fixture.fixed)
        self.assertTrue(fixed["accepted"], fixed["reasons"])

        attach_reused_result(
            self.fixture.fixed, self.fixture.clip, self.fixture.r1,
            self.config_path,
        )
        reused = adapter("clip3d", self.key, self.fixture.clip)
        self.assertTrue(reused["accepted"], reused["reasons"])
        self.assertIn("artifact", reused)

        (self.fixture.clip / "r2_reuse.json").unlink()
        self.fixture._write_point(self.fixture.clip, "clip3d", 1.1)
        self._make_vectors_unequal()
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5):
            run(
                self.fixture.r1, self.fixture.clip / "r2_latency.json",
                self.fixture.clip / "gem5_r2",
            )
        attach(self.fixture.clip)
        local_clip = adapter("clip3d", self.key, self.fixture.clip)
        self.assertTrue(local_clip["accepted"], local_clip["reasons"])
        self.assertIn("result", local_clip)

    def test_equal_override_preflight_marks_missing_reuse_marker_repairable(self):
        """Missing attachment state must not misclassify an equal vector as local."""
        import workflow.r2.run_paired_sweep as paired
        from workflow.r2.reuse_result import attach_reused_result

        attach_reused_result(
            self.fixture.fixed, self.fixture.clip, self.fixture.r1,
            self.config_path,
        )
        (self.fixture.clip / "r2_reuse.json").unlink()
        adapter = paired._existing_r2_validator(
            self.r1_root, self.fixed_root, self.clip_root, self.config_path
        )

        decision = adapter("clip3d", self.key, self.fixture.clip)

        self.assertTrue(decision["accepted"], decision.get("reasons"))
        self.assertTrue(decision["repair_required"])
        self.assertFalse(decision["scientific_evidence_accepted"])
        self.assertEqual(decision["branch"], "reuse")

    def test_mismatched_override_preflight_ignores_stale_reuse_marker(self):
        """A stale marker must not turn a mismatched vector into reuse."""
        import workflow.r2.run_paired_sweep as paired
        from workflow.r2.attach_result import attach
        from workflow.r2.run_r2 import run

        self._make_vectors_unequal()
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5):
            run(
                self.fixture.r1, self.fixture.clip / "r2_latency.json",
                self.fixture.clip / "gem5_r2",
            )
        attach(self.fixture.clip)
        write_json(self.fixture.clip / "r2_reuse.json", {"stale": True})
        adapter = paired._existing_r2_validator(
            self.r1_root, self.fixed_root, self.clip_root, self.config_path
        )

        decision = adapter("clip3d", self.key, self.fixture.clip)

        self.assertTrue(decision["accepted"], decision.get("reasons"))
        self.assertEqual(decision["branch"], "local")
        self.assertTrue(decision["repair_required"])
        self.assertFalse(decision["scientific_evidence_accepted"])

    def test_mismatched_local_pair_clears_stale_reuse_attachment_state(self):
        """The local branch must remove stale reuse fields before certification."""
        from workflow.r2.run_paired_sweep import run_pair
        from workflow.r2.attach_result import attach
        from workflow.r2.run_r2 import run

        self._make_vectors_unequal()
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5):
            run(
                self.fixture.r1, self.fixture.clip / "r2_latency.json",
                self.fixture.clip / "gem5_r2",
            )
        attach(self.fixture.clip)
        summary_path = self.fixture.clip / "pipeline_summary.json"
        summary = read_json(summary_path)
        summary["r2_reused"] = True
        summary["r2_reuse_artifact"] = str(
            (self.fixture.clip / "r2_reuse.json").resolve()
        )
        summary["artifacts"]["r2_reuse"] = summary["r2_reuse_artifact"]
        write_json(summary_path, summary)
        write_json(self.fixture.clip / "r2_reuse.json", {"stale": True})

        result = run_pair(
            self.key, self.r1_root, self.fixed_root, self.clip_root,
            self.config_path, status_root=self.status_root,
        )

        self.assertEqual(result["state"], "success")
        repaired = read_json(summary_path)
        self.assertNotIn("r2_reused", repaired)
        self.assertNotIn("r2_reuse_artifact", repaired)
        self.assertNotIn("r2_reuse", repaired["artifacts"])

    def test_rerun_preflight_marks_stale_metrics_for_replacement(self):
        """Explicit rerun must bypass old scientific attachment trust."""
        import workflow.r2.run_paired_sweep as paired

        result_path = self.fixture.fixed / "gem5_r2/r2_result.json"
        result = read_json(result_path)
        result["latency_sha256"] = "0" * 64
        write_json(result_path, result)
        adapter = paired._existing_r2_validator(
            self.r1_root, self.fixed_root, self.clip_root, self.config_path,
            rerun_requested=True,
        )

        decision = adapter("fixed-bin", self.key, self.fixture.fixed)

        self.assertTrue(decision["accepted"])
        self.assertTrue(decision["repair_required"])
        self.assertTrue(decision["rerun_requested"])
        self.assertFalse(decision["scientific_evidence_accepted"])

    def test_rerun_preflight_rejects_an_unknown_method(self):
        """Rerun replacement must not broaden the accepted method set."""
        import workflow.r2.run_paired_sweep as paired

        adapter = paired._existing_r2_validator(
            self.r1_root, self.fixed_root, self.clip_root, self.config_path,
            rerun_requested=True,
        )

        decision = adapter("unknown", self.key, self.fixture.fixed)

        self.assertFalse(decision["accepted"])
        self.assertIn("unknown method unknown", decision["reasons"])

    def test_missing_marker_sweep_preflight_reaches_pair_repair(self):
        """A sweep must repair eligible reuse attachment state in its worker."""
        from concurrent.futures import Future
        import workflow.r2.run_paired_sweep as paired

        first = paired.run_pair(
            self.key, self.r1_root, self.fixed_root, self.clip_root,
            self.config_path, status_root=self.status_root,
        )
        self.assertEqual(first["state"], "success")
        marker_path = self.fixture.clip / "r2_reuse.json"
        marker_path.unlink()
        decisions = []

        class InlineExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args, **kwargs):
                future = Future()
                future.set_result(function(*args, **kwargs))
                return future

        def validate_roots(*_args, **kwargs):
            decision = kwargs["existing_r2_validator"](
                "clip3d", self.key, self.fixture.clip
            )
            decisions.append(decision)
            if decision.get("accepted") is not True:
                raise ValueError("repairable reuse was rejected")
            return {"mode": "resume"}

        with patch.object(paired, "load_selection", return_value={}), \
                patch.object(paired, "selection_keys", return_value=[self.key]), \
                patch.object(paired, "validate_layout_roots",
                             side_effect=validate_roots), \
                patch.object(paired, "ProcessPoolExecutor", InlineExecutor), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root,
                self.root / "selection.json", self.config_path,
                self.status_root, limit=1,
            )

        self.assertTrue(decisions[0]["repair_required"])
        self.assertEqual(result["completed_pair_count"], 1)
        self.assertEqual(result["pairs"][0]["state"], "success")
        self.assertTrue(marker_path.is_file())

    def test_rerun_sweep_replaces_an_incompatible_successful_cache(self):
        """Rerun preflight must permit and the worker must replace stale R2."""
        from concurrent.futures import Future
        import workflow.r2.run_paired_sweep as paired
        from workflow.r2.run_r2 import validate_local_result

        result_path = self.fixture.fixed / "gem5_r2/r2_result.json"
        stale = read_json(result_path)
        stale["latency_sha256"] = "0" * 64
        write_json(result_path, stale)
        decisions = []

        class InlineExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args, **kwargs):
                future = Future()
                future.set_result(function(*args, **kwargs))
                return future

        def validate_roots(*_args, **kwargs):
            decision = kwargs["existing_r2_validator"](
                "fixed-bin", self.key, self.fixture.fixed
            )
            decisions.append(decision)
            if decision.get("accepted") is not True:
                raise ValueError("rerun replacement was rejected")
            return {"mode": "resume"}

        with patch.object(paired, "load_selection", return_value={}), \
                patch.object(paired, "selection_keys", return_value=[self.key]), \
                patch.object(paired, "validate_layout_roots",
                             side_effect=validate_roots), \
                patch.object(paired, "ProcessPoolExecutor", InlineExecutor), \
                patch("workflow.r2.run_r2.subprocess.run",
                      side_effect=self._complete_gem5), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root,
                self.root / "selection.json", self.config_path,
                self.status_root, rerun=True, limit=1,
            )

        self.assertTrue(decisions[0]["rerun_requested"])
        self.assertEqual(result["pairs"][0]["state"], "success")
        local = validate_local_result(
            self.fixture.r1, self.fixture.fixed / "r2_latency.json",
            self.fixture.fixed / "gem5_r2",
        )
        self.assertTrue(local["accepted"], local["reasons"])
        self.assertNotEqual(read_json(result_path)["latency_sha256"], "0" * 64)

    def test_cli_rejects_nonpositive_jobs(self):
        """Zero workers must fail argument parsing before scheduling anything."""
        from workflow.r2.run_paired_sweep import main

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--r1-root", "r1", "--fixed-root", "fixed",
                  "--clip-root", "clip", "--selection", "selection.json",
                  "--config", "config.json", "--status-root", "status",
                  "--jobs", "0"])

    def test_scheduler_values_cross_a_real_process_pool_boundary(self):
        """Architecture keys and status roots must remain worker-picklable."""
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                _paired_process_smoke, self.key, self.status_root
            ).result(timeout=5)

        self.assertEqual(result["key"], {
            "workload": "fft", "l1d_size": "32kB", "l2_size": "512kB",
        })
        self.assertEqual(
            result["pair_status"],
            str((self.status_root / self.key.relative_path() /
                 "pair_status.json").resolve()),
        )

    def test_cli_rejects_nonpositive_limit(self):
        """A zero smoke limit must fail parsing rather than schedule no work."""
        from workflow.r2.run_paired_sweep import main

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--r1-root", "r1", "--fixed-root", "fixed",
                  "--clip-root", "clip", "--selection", "selection.json",
                  "--config", "config.json", "--status-root", "status",
                  "--limit", "0"])

    def test_limit_one_passes_only_the_first_manifest_entry(self):
        """A smoke limit must preserve manifest order rather than completion order."""
        from concurrent.futures import Future
        from workflow.experiments.balanced50 import load_selection, selection_keys
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        first = selection_keys(load_selection(selection_path))[0]
        observed = []
        status_root = self.status_root

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _function, key, *_args, **_kwargs):
                observed.append(key)
                future = Future()
                future.set_result({
                    "schema_version": 1,
                    "state": "success",
                    "key": {"workload": key.workload,
                            "l1d_size": key.l1d_size, "l2_size": key.l2_size},
                    "clip3d_reused_fixed_r2": True,
                    "physical_r2_runs": 1,
                    "pair_status": str(
                        status_root / key.relative_path() / "pair_status.json"
                    ),
                })
                return future

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value={"mode": "resume"}):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root, selection_path,
                self.config_path, self.status_root, limit=1,
            )

        self.assertEqual(observed, [first])
        self.assertTrue(result["limited_run"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["selected_pair_count"], 1)

    def test_complete_fifty_pair_status_records_fifty_plus_k_runs(self):
        """A complete summary must count one fixed run plus K separate CLIP runs."""
        from concurrent.futures import Future
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        submitted = 0
        status_root = self.status_root

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _function, key, *_args, **_kwargs):
                nonlocal submitted
                separate = submitted < 7
                submitted += 1
                future = Future()
                future.set_result({
                    "schema_version": 1,
                    "state": "success",
                    "key": {"workload": key.workload,
                            "l1d_size": key.l1d_size, "l2_size": key.l2_size},
                    "clip3d_reused_fixed_r2": not separate,
                    "physical_r2_runs": 2 if separate else 1,
                    "pair_status": str(
                        status_root / key.relative_path() / "pair_status.json"
                    ),
                })
                return future

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value={"mode": "resume"}), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root, selection_path,
                self.config_path, self.status_root, jobs=4,
            )

        self.assertTrue(result["complete"])
        self.assertEqual(result["selected_pair_count"], 50)
        self.assertEqual(result["completed_pair_count"], 50)
        self.assertEqual(result["separate_clip_r2_count"], 7)
        self.assertEqual(result["reuse_pair_count"], 43)
        self.assertEqual(result["physical_r2_runs"], 57)
        self.assertEqual(len(result["pairs"]), 50)

    def test_inconsistent_fifty_pair_results_never_mark_complete(self):
        """Success labels alone cannot override key/branch/run-count invariants."""
        from workflow.experiments.balanced50 import load_selection, selection_keys
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        keys = selection_keys(load_selection(selection_path))
        base = {
            key: {
                "schema_version": 1,
                "state": "success",
                "key": {"workload": key.workload,
                        "l1d_size": key.l1d_size, "l2_size": key.l2_size},
                "clip3d_reused_fixed_r2": True,
                "physical_r2_runs": 1,
            }
            for key in keys
        }
        cases = {}
        wrong_key = deepcopy(base)
        wrong_key[keys[-1]]["key"] = deepcopy(wrong_key[keys[0]]["key"])
        cases["duplicate-result-key"] = wrong_key
        missing_branch = deepcopy(base)
        missing_branch[keys[-1]].pop("clip3d_reused_fixed_r2")
        cases["missing-branch"] = missing_branch
        wrong_runs = deepcopy(base)
        wrong_runs[keys[-1]]["physical_r2_runs"] = 2
        cases["wrong-physical-count"] = wrong_runs
        malformed_runs = deepcopy(base)
        malformed_runs[keys[-1]]["physical_r2_runs"] = "one"
        cases["malformed-physical-count"] = malformed_runs
        offsetting_runs = deepcopy(base)
        offsetting_runs[keys[0]]["clip3d_reused_fixed_r2"] = False
        offsetting_runs[keys[0]]["physical_r2_runs"] = 1
        offsetting_runs[keys[1]]["physical_r2_runs"] = 2
        cases["offsetting-per-pair-counts"] = offsetting_runs

        for label, results in cases.items():
            with self.subTest(label=label):
                status = paired._experiment_status(
                    keys, results, self.status_root, {"mode": "resume"},
                    False, 1.0,
                )
                self.assertFalse(status["complete"])

    def test_limited_cli_rejects_an_inconsistent_success_record(self):
        """A smoke subset succeeds only when every returned pair is valid."""
        import workflow.r2.run_paired_sweep as paired

        inconsistent = {
            "complete": False,
            "limited_run": True,
            "selected_results_valid": False,
            "selected_pair_count": 1,
            "completed_pair_count": 1,
            "failed_pair_count": 0,
            "pairs": [{"state": "success", "key": {"workload": "wrong"}}],
        }
        with patch.object(paired, "run_sweep", return_value=inconsistent):
            exit_code = paired.main([
                "--r1-root", "r1", "--fixed-root", "fixed",
                "--clip-root", "clip", "--selection", "selection.json",
                "--config", "config.json", "--status-root", "status",
                "--limit", "1",
            ])

        self.assertEqual(exit_code, 1)

    def test_normal_cli_is_nonzero_while_any_of_fifty_is_incomplete(self):
        """Subset success cannot make a full 50-pair invocation exit zero."""
        import workflow.r2.run_paired_sweep as paired

        selection_path = Path(__file__).resolve().parents[1] / (
            "configs/experiments/balanced50_traffic_weighted.json"
        )
        incomplete = {
            "complete": False, "limited_run": False,
            "selected_pair_count": 50, "completed_pair_count": 49,
            "failed_pair_count": 1, "pairs": [],
        }
        with patch.object(paired, "run_sweep", return_value=incomplete):
            exit_code = paired.main([
                "--r1-root", str(self.r1_root),
                "--fixed-root", str(self.fixed_root),
                "--clip-root", str(self.clip_root),
                "--selection", str(selection_path),
                "--config", str(self.config_path),
                "--status-root", str(self.status_root),
            ])

        self.assertEqual(exit_code, 1)

if __name__ == "__main__":
    unittest.main()
