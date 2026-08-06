from __future__ import annotations

import tempfile
import unittest
import sys
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


if __name__ == "__main__":
    unittest.main()
