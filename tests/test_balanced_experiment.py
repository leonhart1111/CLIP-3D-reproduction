from __future__ import annotations

import tempfile
import unittest
import sys
import shutil
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


NATIVE_PREFLIGHT = {
    "mode": "resume",
    "fixed": {"physical_model_authority": "McPAT 1.3 embedded CACTI-P"},
    "clip3d": {"physical_model_authority": "McPAT 1.3 embedded CACTI-P"},
}


class AtomicJsonPublicationTests(unittest.TestCase):
    """Shared JSON publication must not collide or leave partial state."""

    def test_write_json_does_not_reuse_another_writers_legacy_temp(self):
        """A deterministic sibling temp must remain owned by its original writer."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            destination = root / "status.json"
            legacy_temp = root / "status.json.tmp"
            legacy_temp.write_text("other writer\n", encoding="utf-8")

            write_json(destination, {"state": "success"})

            self.assertEqual(read_json(destination), {"state": "success"})
            self.assertTrue(legacy_temp.is_file())
            self.assertEqual(
                legacy_temp.read_text(encoding="utf-8"), "other writer\n"
            )

    def test_write_json_serialization_failure_leaves_no_partial_temp(self):
        """A failed serialization must preserve the old file and directory contents."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            destination = root / "status.json"
            write_json(destination, {"state": "old"})
            before = {path.name: path.read_bytes() for path in root.iterdir()}

            with self.assertRaises(TypeError):
                write_json(destination, {"not_json": object()})

            self.assertEqual(read_json(destination), {"state": "old"})
            self.assertEqual(
                {path.name: path.read_bytes() for path in root.iterdir()}, before
            )

    def test_attachment_byte_publication_never_exposes_a_torn_destination(self):
        """Readers must observe either the old attachment JSON or the complete new JSON."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from workflow.r2.attach_result import _publish_bytes

        with tempfile.TemporaryDirectory() as name:
            destination = Path(name) / "performance.json"
            old = b'{"state": "old"}\n'
            new = b'{"state": "new", "padding": "0123456789"}\n'
            destination.write_bytes(old)
            half_written = threading.Event()
            finish_write = threading.Event()
            real_write_bytes = Path.write_bytes

            def slow_destination_write(path, data):
                if Path(path).resolve() != destination.resolve():
                    return real_write_bytes(path, data)
                with Path(path).open("wb") as stream:
                    stream.write(data[:len(data) // 2])
                    stream.flush()
                    half_written.set()
                    if not finish_write.wait(2):
                        raise TimeoutError("reader never sampled partial publication")
                    stream.write(data[len(data) // 2:])
                return len(data)

            with patch.object(Path, "write_bytes", new=slow_destination_write), \
                    ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_publish_bytes, destination, new)
                exposed = None
                if half_written.wait(0.2):
                    exposed = destination.read_bytes()
                    finish_write.set()
                published_digest = future.result(timeout=2)

            if exposed is None:
                exposed = destination.read_bytes()
            self.assertIn(exposed, (old, new))
            self.assertEqual(destination.read_bytes(), new)
            self.assertEqual(published_digest, sha256_file(destination))


class CanonicalFixtureTests(unittest.TestCase):
    def write_canonical_plan(self, root: Path, experiment: Path,
                             profile: str = "paper") -> dict:
        grid = read_json(experiment)
        jobs = []
        for workload in grid["workloads"]:
            for l1d_size in grid["l1d_sizes"]:
                for l2_size in grid["l2_sizes"]:
                    output_dir = (
                        root / workload / f"l1d_{l1d_size}" / f"l2_{l2_size}"
                    )
                    jobs.append({
                        "workload": workload,
                        "l1d_size": l1d_size,
                        "l2_size": l2_size,
                        "profile": profile,
                        "output_dir": str(output_dir.resolve()),
                    })
        plan = {
            "experiment": grid.get("name", "fixture"),
            "profile": profile,
            "job_count": len(jobs),
            "jobs": jobs,
        }
        write_json(root / "planned_jobs.json", plan)
        return plan

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
        self.write_canonical_plan(root, experiment)
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
    def test_audit_requires_planned_jobs_file(self):
        """Canonical artifacts alone cannot certify an absent execution plan."""
        from workflow.analysis.audit_r1 import audit

        root, experiment = self.make_grid_fixture()
        (root / "planned_jobs.json").unlink()

        result = audit(root, root.parent / "audit.json", experiment, "paper",
                       expected_points=4)

        self.assertFalse(result["complete"])
        self.assertFalse(result["canonical_plan_valid"])
        self.assertIn("missing planned_jobs.json", result["canonical_plan_errors"])

    def test_audit_rejects_duplicate_jobs_even_when_job_count_matches(self):
        """A duplicated key cannot stand in for another canonical plan entry."""
        from workflow.analysis.audit_r1 import audit

        root, experiment = self.make_grid_fixture()
        plan = read_json(root / "planned_jobs.json")
        plan["jobs"][1] = deepcopy(plan["jobs"][0])
        write_json(root / "planned_jobs.json", plan)

        result = audit(root, root.parent / "audit.json", experiment, "paper",
                       expected_points=4)

        self.assertFalse(result["complete"])
        self.assertFalse(result["canonical_plan_valid"])
        self.assertTrue(any("ordered canonical job" in error
                            for error in result["canonical_plan_errors"]))

    def test_audit_rejects_quarantined_plan_output(self):
        """A canonical key cannot redirect execution into a quarantine directory."""
        from workflow.analysis.audit_r1 import audit

        root, experiment = self.make_grid_fixture()
        plan = read_json(root / "planned_jobs.json")
        plan["jobs"][0]["output_dir"] = str(
            (root / "fft/l1d_16kB/l2_128kB.corrupt_duplicate_fixture").resolve()
        )
        write_json(root / "planned_jobs.json", plan)

        result = audit(root, root.parent / "audit.json", experiment, "paper",
                       expected_points=4)

        self.assertFalse(result["complete"])
        self.assertTrue(any("output_dir" in error
                            for error in result["canonical_plan_errors"]))

    def test_audit_rejects_missing_extra_reordered_or_wrong_profile_jobs(self):
        """Count-preserving plan edits cannot change the exact canonical order."""
        from workflow.analysis.audit_r1 import audit

        mutations = {
            "missing": lambda plan: plan["jobs"].pop(),
            "extra": lambda plan: plan["jobs"].append(deepcopy(plan["jobs"][-1])),
            "reordered": lambda plan: plan["jobs"].reverse(),
            "wrong-profile": lambda plan: plan["jobs"][0].__setitem__(
                "profile", "smoke"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root, experiment = self.make_grid_fixture()
                plan = read_json(root / "planned_jobs.json")
                mutate(plan)
                write_json(root / "planned_jobs.json", plan)

                result = audit(
                    root, root.parent / f"audit-{label}.json", experiment,
                    "paper", expected_points=4,
                )

                self.assertFalse(result["complete"])
                self.assertFalse(result["canonical_plan_valid"])

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
            self.write_canonical_plan(output_root / profile, experiment, profile)
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
        from tests.test_mcpat_native_cache import (
            write_native_physical_fixture,
            write_native_r1_fixture,
        )
        from workflow.run_lifting_sweep import required_artifacts

        for artifact in required_artifacts:
            write_json(output / artifact, {})
        binary = output / "mcpat/test-binary"
        binary.write_bytes(b"strict McPAT test binary\n")
        r1 = output / ".source-r1"
        l1d_part = next(
            (part for part in reversed(output.parts) if part.startswith("l1d_")),
            "l1d_32kB",
        )
        l2_part = next(
            (part for part in reversed(output.parts) if part.startswith("l2_")),
            "l2_512kB",
        )
        l1d_index = output.parts.index(l1d_part) if l1d_part in output.parts else -1
        metadata = {
            "workload": output.parts[l1d_index - 1] if l1d_index > 0 else "fft",
            "num_cores": 4,
            "cpu_clock": "2GHz", "l1i_size": "32kB",
            "l1d_size": l1d_part.removeprefix("l1d_"),
            "l2_size": l2_part.removeprefix("l2_"),
            "l1_associativity": 2, "l2_associativity": 8,
            "cache_line_bytes": 64,
        }
        write_native_r1_fixture(r1, metadata)
        mcpat, _model = write_native_physical_fixture(
            output, r1, binary, metadata,
        )
        mcpat_output = output / "mcpat/mcpat.out"
        write_json(output / "run_config.json", {"config": config})
        write_json(output / "pipeline_summary.json", {
            "r1": str(r1.resolve()),
            "layout_method": layout_method,
            "cooling": {"r_convec_k_per_w": 5.0},
            "ipc2": None,
            "bips2": None,
            "stage_seconds": {},
            "cache_authority": "McPAT 1.3 embedded CACTI-P",
            "mcpat_provenance": mcpat["provenance"],
            "artifacts": {
                "mcpat_json": str((output / "mcpat/mcpat.json").resolve()),
                "mcpat_output": str(mcpat_output.resolve()),
                "mcpat_binary": str(binary.resolve()),
            },
            "artifact_sha256": {
                "mcpat_json": sha256_file(output / "mcpat/mcpat.json"),
                "mcpat_output": mcpat["provenance"]["hashes"]["output_sha256"],
                "mcpat_binary": mcpat["provenance"]["hashes"]["binary_sha256"],
            },
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
        from tests.test_mcpat_native_cache import (
            write_native_physical_fixture,
            write_native_r1_fixture,
        )
        from workflow.r1_catalog import expected_keys
        from workflow.run_lifting_sweep import (
            CLIP3D_REQUIRED_ARTIFACTS,
            required_artifacts,
        )

        config_path = (config_path or self.config_path).resolve()
        config = config if config is not None else self.config
        fixed_root = self.root / "fixed"
        clip_root = self.root / "clip"
        mcpat_binary = self.root / "tools/mcpat"
        mcpat_binary.parent.mkdir(parents=True, exist_ok=True)
        mcpat_binary.write_bytes(b"strict patched McPAT fixture\n")
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
                    "stage_seconds": {},
                })
                fixed = self.grid["fixed_architecture"]
                r1 = self.root / "native-r1" / key.relative_path()
                metadata = {
                    "workload": key.workload,
                    "num_cores": fixed["cores"],
                    "cpu_clock": fixed["clock"],
                    "l1i_size": fixed["l1i_size"],
                    "l1d_size": key.l1d_size,
                    "l2_size": key.l2_size,
                    "l1_associativity": fixed["l1_associativity"],
                    "l2_associativity": fixed["l2_associativity"],
                    "cache_line_bytes": fixed["cache_line_bytes"],
                }
                write_native_r1_fixture(r1, metadata)
                mcpat, model = write_native_physical_fixture(
                    point, r1, mcpat_binary, metadata,
                )
                mcpat_output = point / "mcpat/mcpat.out"
                summary = read_json(point / "pipeline_summary.json")
                summary.update({
                    "r1": str(r1.resolve()),
                    "cache_authority": "McPAT 1.3 embedded CACTI-P",
                    "mcpat_provenance": mcpat["provenance"],
                    "communication_profile": model["communication_profile"],
                    "artifacts": {
                        "mcpat_json": str((point / "mcpat/mcpat.json").resolve()),
                        "mcpat_output": str(mcpat_output.resolve()),
                        "mcpat_binary": str(mcpat_binary.resolve()),
                    },
                    "artifact_sha256": {
                        "mcpat_json": sha256_file(point / "mcpat/mcpat.json"),
                        "mcpat_output": mcpat["provenance"]["hashes"]["output_sha256"],
                        "mcpat_binary": mcpat["provenance"]["hashes"]["binary_sha256"],
                    },
                })
                write_json(point / "pipeline_summary.json", summary)
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
        self.assertEqual(manifest["physical_model"], {
            "cache_characterization": "local-cacti-schema-v2",
            "non_cache_area": "unmodified-mcpat",
            "cache_area": "unmodified-local-cacti",
            "global_area_scaling": "none",
            "historical_scaled_layouts_reusable": False,
        })
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

    def test_batch_sweep_rejects_native_roots_from_different_r1_before_mutation(self):
        """Batch admission binds every physical point to the supplied R1 root."""
        import workflow.r2.run_paired_sweep as paired

        def snapshot(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
            }

        fixed_root, clip_root = self._write_layout_roots()
        alternate_r1_root = self.root / "alternate-r1"
        shutil.copytree(self.root / "native-r1", alternate_r1_root)
        status_root = self.root / "paired-status"
        before = {
            "fixed": snapshot(fixed_root),
            "clip": snapshot(clip_root),
        }

        with patch.object(
            paired, "ProcessPoolExecutor",
            side_effect=AssertionError("wrong-R1 roots must not submit workers"),
        ) as executor, patch.object(
            paired.run_r2, "run",
            side_effect=AssertionError("wrong-R1 roots must not execute R2"),
        ), patch.object(
            paired.attach_result, "attach",
            side_effect=AssertionError("wrong-R1 roots must not attach R2"),
        ), self.assertRaisesRegex(ValueError, "R1"):
            paired.run_sweep(
                alternate_r1_root, fixed_root, clip_root, self.manifest_path,
                self.config_path, status_root,
            )

        executor.assert_not_called()
        self.assertEqual(snapshot(fixed_root), before["fixed"])
        self.assertEqual(snapshot(clip_root), before["clip"])
        self.assertFalse(status_root.exists())

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

    def test_preflight_rejects_historical_scaled_geometry_after_config_repin(self):
        """A new config hash must not make scaled historical layouts reusable."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        modules = read_json(point / "modules.json")
        modules["area_provenance"]["global_scaling"] = "legacy 150 mm2 calibration"
        write_json(point / "modules.json", modules)

        with self.assertRaisesRegex(ValueError, "area provenance.*McPAT-native"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_non_granular_native_module_model(self):
        """A native McPAT artifact cannot legitimize an aggregate module model."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = clip_root / keys[0].relative_path()
        modules = read_json(point / "modules.json")
        modules["module_schema"]["requires_granular_cores"] = False
        write_json(point / "modules.json", modules)

        with self.assertRaisesRegex(ValueError, "strict granular"):
            validate_layout_roots(fixed_root, clip_root, keys, self.config_path,
                                  selection=manifest)

    def test_preflight_rejects_historical_cacti_only_point(self):
        """Standalone characterization cannot replace missing native McPAT evidence."""
        from workflow.experiments.balanced50 import validate_layout_roots

        manifest, keys = self._selection()
        fixed_root, clip_root = self._write_layout_roots()
        point = fixed_root / keys[0].relative_path()
        (point / "mcpat/mcpat.json").unlink()
        write_json(point / "cacti/cacti_characterization.json", {
            "schema_version": 2, "characterization_id": "c" * 64,
        })

        with self.assertRaisesRegex(ValueError, "missing mcpat/mcpat.json"):
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
            "cpu_clock": "2GHz",
            "l1_associativity": 2,
            "l2_associativity": 8,
            "cache_line_bytes": 64,
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
        with (self.r1 / "stats.txt").open("a", encoding="utf-8") as stream:
            for core in range(4):
                stream.write(
                    f"system.l2.demandAccesses::cpu{core}.data {100 + core}\n"
                )
        write_json(self.r1 / "status.json", {
            "schema_version": 2,
            "state": "success",
            "instruction_window_scope": "cpu0",
        })
        self.mcpat_binary = self.root / "tools/mcpat"
        self.mcpat_binary.parent.mkdir(parents=True)
        self.mcpat_binary.write_bytes(b"strict patched McPAT R2 fixture\n")
        for point, method, frequency in (
                (self.fixed, "fixed-bin", 1.4),
                (self.clip, "clip3d", 1.1)):
            self._write_point(point, method, frequency)
        self._write_source_result()

    def _write_point(self, point: Path, method: str, frequency: float) -> None:
        ipc1 = 9.7
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
            "ipc1": ipc1,
            "gamma": 0.2,
            "totals": {
                "dynamic_power_w": 80.0,
                "leakage_power_w": 20.0,
                "total_power_w": 100.0,
                "area_mm2": 45.0,
            },
            "modules": [{
                "name": "fixture-total",
                "dynamic_power_w": 80.0,
                "leakage_power_w": 20.0,
                "total_power_w": 100.0,
            }],
        })
        from tests.test_mcpat_native_cache import write_native_physical_fixture
        from workflow.r2.build_latency_vector import build_vector

        mcpat, model = write_native_physical_fixture(
            point, self.r1, self.mcpat_binary, self.metadata,
        )
        build_vector(
            point / "modules.json", point / "r2_latency.json",
            tsv_hops=1, wire_cycles=3,
        )
        ipc1 = model["ipc1"]
        gamma = float(model["gamma"])
        target_tmax_c = 25.0 + 70.0 / (
            gamma + (1.0 - gamma) * frequency / 2.0
        )
        target_tmax_k = target_tmax_c + 273.15
        hotspot = point / "hotspot"
        hotspot.mkdir(parents=True, exist_ok=True)
        (hotspot / "steady.txt").write_text(
            f"core0 {target_tmax_k:.15g}\n", encoding="utf-8"
        )
        write_json(hotspot / "hotspot_manifest.json", {
            "schema_version": 1,
            "ambient_c": 25.0,
            "r_convec_k_per_w": 5.0,
        })
        write_json(point / "hotspot/thermal_result.json", {
            "schema_version": 1,
            "command": ["hotspot", "-c", "hotspot.config"],
            "return_code": 0,
            "power_trace": str((point / "hotspot/power.ptrace").resolve()),
            "steady_file": str((point / "hotspot/steady.txt").resolve()),
            "grid_steady_file": str((point / "hotspot/grid.steady.txt").resolve()),
            "tmax_k": target_tmax_k,
            "tmax_c": target_tmax_c,
            "peak_unit": "core0",
            "ambient_c": 25.0,
            "r_convec_k_per_w": 5.0,
            "sample_count": 1,
        })
        from workflow.thermal.sustainable_frequency import derive
        performance = derive(
            model, read_json(point / "hotspot/thermal_result.json"),
            2.0, 0.4, 95.0, 25.0,
        )
        write_json(point / "performance.json", performance)
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
            "gamma": model["gamma"],
            "tmax_c": target_tmax_c,
            "sustainable_frequency_ghz": performance[
                "sustainable_frequency_ghz"
            ],
            "ipc1": ipc1,
            "bips1_thermal": performance["bips1_thermal"],
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
        summary_path = point / "pipeline_summary.json"
        summary = read_json(summary_path)
        summary.update({
            "schema_version": 3,
            "module_count": len(model["modules"]),
            "total_power_w": model["totals"]["total_power_w"],
            "power_provenance": model["power_provenance"],
            "area_provenance": model["area_provenance"],
            "cache_authority": model["cache_authority"],
            "mcpat_provenance": model["mcpat_provenance"],
            "communication_profile": model["communication_profile"],
            "gamma": model["gamma"],
            "ipc1": model["ipc1"],
            "tmax_c": performance["tmax_f0_c"],
            "sustainable_frequency_ghz": performance[
                "sustainable_frequency_ghz"
            ],
            "bips1_thermal": performance["bips1_thermal"],
        })
        summary["stage_seconds"].pop("cacti", None)
        mcpat_output = point / "mcpat/mcpat.out"
        summary["artifacts"].update({
            "mcpat_json": str((point / "mcpat/mcpat.json").resolve()),
            "mcpat_output": str(mcpat_output.resolve()),
            "mcpat_binary": str(self.mcpat_binary.resolve()),
        })
        summary["artifact_sha256"] = {
            "mcpat_json": sha256_file(point / "mcpat/mcpat.json"),
            "mcpat_output": mcpat["provenance"]["hashes"]["output_sha256"],
            "mcpat_binary": mcpat["provenance"]["hashes"]["binary_sha256"],
        }
        write_json(summary_path, summary)

    def _identity(self, latency_path: Path) -> dict:
        metadata = read_json(self.r1 / "r1_metadata.json")
        return {
            "r1_directory": str(self.r1.resolve()),
            "r1_metadata_sha256": sha256_file(self.r1 / "r1_metadata.json"),
            "r1_stats_sha256": sha256_file(self.r1 / "stats.txt"),
            "latency_vector": str(latency_path.resolve()),
            "latency_sha256": sha256_file(latency_path),
            "instruction_window_scope": metadata.get(
                "instruction_window_scope", "cpu0"
            ),
            "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
            "measure_insts_cpu0": metadata["measure_insts_cpu0"],
        }

    def _write_source_result(self) -> None:
        output = self.fixed / "gem5_r2"
        result_path = output / "r2_result.json"
        identity = self._identity(self.fixed / "r2_latency.json")
        gem5_args = read_json(self.fixed / "r2_latency.json")["gem5_args"]
        command = [
            "/opt/gem5.opt", "--listener-mode=off", f"--outdir={output.resolve()}",
            "/opt/clip_r1.py", "--stage", "R2", "--workload", "fft",
            "--l1i-size", "32kB", "--l1d-size", "32kB",
            "--l2-size", "512kB", "--warmup-insts", "100000000",
            "--measure-insts", "500000000", "--instruction-window-scope",
            identity["instruction_window_scope"], *gem5_args,
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

    def test_local_attach_rejects_legacy_r1_and_module_scope_omission(self):
        """Legacy scope evidence remains readable but cannot be relabeled native."""
        from workflow.r2.attach_result import attach

        self.metadata.pop("instruction_window_scope")
        write_json(self.r1 / "r1_metadata.json", self.metadata)
        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["architecture"].pop("instruction_window_scope")
        write_json(modules_path, modules)
        self._write_source_result()

        with self.assertRaisesRegex(ValueError, "McPAT-native|module"):
            attach(self.fixed)

    def test_local_attach_rejects_explicit_scope_mismatch(self):
        """Compatibility for an absent key must not accept all-cores evidence."""
        from workflow.r2.attach_result import attach

        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["architecture"]["instruction_window_scope"] = "all-cores"
        write_json(modules_path, modules)

        with self.assertRaisesRegex(ValueError, "instruction_window_scope"):
            attach(self.fixed)

    def test_local_attach_rejects_explicit_unsupported_scope(self):
        """Self-consistent local evidence cannot introduce a new scope domain."""
        from workflow.r2.attach_result import attach

        self.metadata["instruction_window_scope"] = "bogus"
        write_json(self.r1 / "r1_metadata.json", self.metadata)
        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["architecture"]["instruction_window_scope"] = "bogus"
        write_json(modules_path, modules)
        self._write_source_result()

        with self.assertRaisesRegex(ValueError, "instruction_window_scope"):
            attach(self.fixed)

    def test_local_attach_rejects_explicit_nonstring_scope(self):
        """JSON collection scopes must be rejected rather than raising TypeError."""
        from workflow.r2.attach_result import attach

        self.metadata["instruction_window_scope"] = ["cpu0"]
        write_json(self.r1 / "r1_metadata.json", self.metadata)
        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["architecture"]["instruction_window_scope"] = ["cpu0"]
        write_json(modules_path, modules)
        self._write_source_result()

        with self.assertRaisesRegex(ValueError, "instruction_window_scope"):
            attach(self.fixed)

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

    def test_local_attach_recomputes_every_derived_summary_field(self):
        """Local attachment publishes one coherent physical/R2 summary snapshot."""
        from workflow.r2.attach_result import attach

        performance_path = self.fixed / "performance.json"
        performance = read_json(performance_path)
        performance.update({
            "gamma": 0.9,
            "tmax_f0_c": 999.0,
            "sustainable_frequency_ghz": 9.0,
            "ipc1": 99.0,
            "bips1_thermal": 891.0,
            "ipc2": 3.25,
            "bips2": 29.25,
        })
        write_json(performance_path, performance)
        summary_path = self.fixed / "pipeline_summary.json"
        summary = read_json(summary_path)
        summary.update({
            "gamma": 0.9,
            "tmax_c": 999.0,
            "sustainable_frequency_ghz": 9.0,
            "ipc1": 99.0,
            "bips1_thermal": 891.0,
            "ipc2": 99.0,
            "bips2": 891.0,
            "r2_source": str((self.root / "forged-result.json").resolve()),
            "total_pipeline_seconds": 999.0,
        })
        summary["stage_seconds"]["gem5_r2"] = 999.0
        for artifact in (
                "config", "modules", "thermal", "performance", "r2_latency",
                "r2_result"):
            summary["artifacts"][artifact] = str(
                (self.root / f"forged-{artifact}").resolve()
            )
        write_json(summary_path, summary)

        attached = attach(self.fixed)
        published = read_json(performance_path)

        for summary_field, performance_field in (
                ("gamma", "gamma"), ("tmax_c", "tmax_f0_c"),
                ("sustainable_frequency_ghz", "sustainable_frequency_ghz"),
                ("ipc1", "ipc1"), ("bips1_thermal", "bips1_thermal"),
                ("ipc2", "ipc2"), ("bips2", "bips2")):
            self.assertEqual(attached[summary_field], published[performance_field])
        result_path = (self.fixed / "gem5_r2/r2_result.json").resolve()
        self.assertEqual(attached["r2_source"], str(result_path))
        expected_artifacts = {
            "config": str((self.fixed / "run_config.json").resolve()),
            "modules": str((self.fixed / "modules.json").resolve()),
            "thermal": str(
                (self.fixed / "hotspot/thermal_result.json").resolve()
            ),
            "performance": str((self.fixed / "performance.json").resolve()),
            "r2_latency": str((self.fixed / "r2_latency.json").resolve()),
            "r2_result": str(result_path),
        }
        for field, value in expected_artifacts.items():
            self.assertEqual(attached["artifacts"][field], value)
        self.assertEqual(
            attached["artifacts"]["mcpat_json"],
            str((self.fixed / "mcpat/mcpat.json").resolve()),
        )
        self.assertEqual(
            attached["artifacts"]["mcpat_output"],
            str((self.fixed / "mcpat/mcpat.out").resolve()),
        )
        self.assertEqual(
            attached["artifacts"]["mcpat_binary"],
            str(self.mcpat_binary.resolve()),
        )
        self.assertNotIn("cacti", attached["artifacts"])
        self.assertEqual(attached["stage_seconds"]["gem5_r2"], 17.5)
        self.assertEqual(
            attached["total_pipeline_seconds"],
            sum(attached["stage_seconds"].values()),
        )

    def test_local_attach_rejects_invalid_module_and_thermal_inputs(self):
        """Attachment cannot normalize corrupt architecture or HotSpot provenance."""
        from workflow.r2.attach_result import attach

        cases = {
            "module-architecture": (
                self.fixed / "modules.json",
                lambda value: value["architecture"].__setitem__("l2_size", "1MB"),
            ),
            "module-gamma": (
                self.fixed / "modules.json",
                lambda value: value.__setitem__("gamma", 0.9),
            ),
            "thermal-temperature": (
                self.fixed / "hotspot/thermal_result.json",
                lambda value: value.__setitem__("tmax_k", 999.0),
            ),
            "thermal-provenance": (
                self.fixed / "hotspot/thermal_result.json",
                lambda value: value.__setitem__(
                    "power_trace", str((self.root / "forged.ptrace").resolve())
                ),
            ),
        }
        originals = {path: path.read_bytes() for path, _mutate in cases.values()}
        for label, (path, mutate) in cases.items():
            with self.subTest(label=label):
                for original_path, data in originals.items():
                    original_path.write_bytes(data)
                value = read_json(path)
                mutate(value)
                write_json(path, value)
                with self.assertRaises(ValueError):
                    attach(self.fixed)

    def test_local_attach_rejects_coherent_ipc1_not_derived_from_r1_stats(self):
        """Attachment must not legitimize an IPC1 forged across all derived files."""
        from workflow.r2.attach_result import attach

        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["ipc1"] = 99.0
        write_json(modules_path, modules)

        with self.assertRaisesRegex(ValueError, "IPC1|R1 stats"):
            attach(self.fixed)

    def test_local_attach_rejects_coherent_tmax_not_derived_from_steady_output(self):
        """Attachment must not legitimize a thermal peak forged above HotSpot output."""
        from workflow.r2.attach_result import attach

        thermal_path = self.fixed / "hotspot/thermal_result.json"
        thermal = read_json(thermal_path)
        thermal["tmax_c"] = 999.0
        thermal["tmax_k"] = 1272.15
        write_json(thermal_path, thermal)

        with self.assertRaisesRegex(ValueError, "steady|thermal|Tmax"):
            attach(self.fixed)

    def test_local_attach_rejects_thermal_identity_not_derived_from_steady_output(self):
        """Peak name and sample count must describe the captured steady samples."""
        from workflow.r2.attach_result import attach

        thermal_path = self.fixed / "hotspot/thermal_result.json"
        original = thermal_path.read_bytes()
        for field, value in (("peak_unit", "forged"), ("sample_count", 99)):
            with self.subTest(field=field):
                thermal_path.write_bytes(original)
                thermal = read_json(thermal_path)
                thermal[field] = value
                write_json(thermal_path, thermal)
                with self.assertRaisesRegex(ValueError, "steady|peak|sample"):
                    attach(self.fixed)

    def test_local_attach_rejects_hotspot_manifest_not_bound_to_config(self):
        """Thermal JSON cannot remain certified beside a conflicting manifest."""
        from workflow.r2.attach_result import attach

        manifest_path = self.fixed / "hotspot/hotspot_manifest.json"
        manifest = read_json(manifest_path)
        manifest["ambient_c"] = 30.0
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "manifest|ambient"):
            attach(self.fixed)

    def test_local_attach_rejects_power_totals_not_derived_from_module_records(self):
        """Attachment must not legitimize totals forged above unchanged modules."""
        from workflow.r2.attach_result import attach

        modules_path = self.fixed / "modules.json"
        modules = read_json(modules_path)
        modules["totals"].update({
            "dynamic_power_w": 160.0,
            "leakage_power_w": 40.0,
            "total_power_w": 200.0,
        })
        modules["gamma"] = 0.2
        write_json(modules_path, modules)

        with self.assertRaisesRegex(ValueError, "power|module"):
            attach(self.fixed)

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

    def test_local_attachment_rejects_changed_mcpat_binary_hash(self):
        """A live binary replacement cannot retain corrected attachment authority."""
        from workflow.r2.attach_result import attach

        self.mcpat_binary.write_bytes(b"replaced McPAT binary\n")

        with self.assertRaisesRegex(ValueError, "McPAT|binary"):
            attach(self.fixed)

    def test_reuse_rejects_changed_native_record_and_mcpat_hash_identities(self):
        """Equal latency overrides cannot hide changed McPAT-native identities."""
        from workflow.r2.reuse_result import validate_reuse

        def replace_record_id(modules: dict) -> None:
            record = next(
                item for item in modules["embedded_cacti_p"]["records"]
                if item["cache"] == "l2"
            )
            record["record_id"] = "f" * 64

        def replace_output_hash(modules: dict) -> None:
            modules["mcpat_provenance"]["hashes"]["output_sha256"] = "e" * 64

        for label, mutate in (
                ("record-id", replace_record_id),
                ("output-hash", replace_output_hash)):
            with self.subTest(label=label):
                self._write_point(self.clip, "clip3d", 1.1)
                modules_path = self.clip / "modules.json"
                modules = read_json(modules_path)
                mutate(modules)
                write_json(modules_path, modules)

                decision = validate_reuse(
                    self.fixed, self.clip, self.r1, self.config_path,
                )

                self.assertFalse(decision["accepted"])
                self.assertTrue(any(
                    "McPAT" in reason or "cache" in reason
                    for reason in decision["reasons"]
                ), decision["reasons"])

    def test_override_schema_rejects_loose_types_keys_and_cycle_bounds(self):
        """Only complete positive uint64 integer latency overrides are renderable."""
        from workflow.r2.run_r2 import canonical_gem5_args

        invalid = {}
        fractional = deepcopy(self.overrides)
        fractional["xbar_forward_latency"] = 8.0
        invalid["float"] = fractional
        boolean = deepcopy(self.overrides)
        boolean["l1i_response_latency"] = True
        invalid["boolean"] = boolean
        zero = deepcopy(self.overrides)
        zero["l1d_response_latency"] = 0
        invalid["zero"] = zero
        overflow = deepcopy(self.overrides)
        overflow["l2_tag_latency"] = 1 << 64
        invalid["uint64-overflow"] = overflow
        missing = deepcopy(self.overrides)
        missing.pop("xbar_response_latency")
        invalid["missing-key"] = missing
        extra = deepcopy(self.overrides)
        extra["unexpected_latency"] = 1
        invalid["extra-key"] = extra

        for label, overrides in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                canonical_gem5_args(overrides)
        self.assertEqual(canonical_gem5_args(deepcopy(self.overrides)), self.gem5_args)

    def test_reuse_rejects_integer_float_equality_and_argument_spelling(self):
        """Equal numeric values with `8` versus `8.0` are different vectors."""
        from workflow.r2.reuse_result import validate_reuse

        vector_path = self.clip / "r2_latency.json"
        vector = read_json(vector_path)
        vector["gem5_overrides"]["xbar_forward_latency"] = 8.0
        vector["gem5_args"][21] = "8.0"
        write_json(vector_path, vector)

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("non-boolean integer" in reason
                            for reason in decision["reasons"]), decision["reasons"])

    def test_reuse_rejects_boolean_integer_equality(self):
        """Python's `True == 1` rule cannot make two latency vectors reusable."""
        from workflow.r2.reuse_result import validate_reuse

        vector_path = self.clip / "r2_latency.json"
        vector = read_json(vector_path)
        vector["gem5_overrides"]["l1i_response_latency"] = True
        vector["gem5_args"][5] = "True"
        write_json(vector_path, vector)

        decision = validate_reuse(self.fixed, self.clip, self.r1, self.config_path)

        self.assertFalse(decision["accepted"])
        self.assertTrue(any("non-boolean integer" in reason
                            for reason in decision["reasons"]), decision["reasons"])

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
        self.assertAlmostEqual(
            artifact["target_outputs"]["sustainable_frequency_ghz"], 1.1,
        )
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
        self._upgrade_to_native_physical_points()

    def _upgrade_to_native_physical_points(self) -> None:
        """Give direct-pair tests real Task-3/4 physical authority."""
        from tests.test_mcpat_native_cache import (
            native_mcpat_artifact,
            native_text,
        )
        from workflow.floorplan.build_module_model import build_model
        from workflow.r2 import attach_result
        from workflow.r2.build_latency_vector import build_vector
        from workflow.thermal.sustainable_frequency import derive

        metadata = read_json(self.fixture.r1 / "r1_metadata.json")
        metadata.update({
            "cpu_clock": metadata["clock"],
            "l1_associativity": 2,
            "l2_associativity": 8,
            "cache_line_bytes": 64,
        })
        write_json(self.fixture.r1 / "r1_metadata.json", metadata)
        with (self.fixture.r1 / "stats.txt").open("a", encoding="utf-8") as stream:
            for core in range(4):
                stream.write(
                    f"system.l2.demandAccesses::cpu{core}.data {100 + core}\n"
                )

        binary = self.root / "tools/mcpat"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"strict patched McPAT fixture\n")
        for point, method in (
                (self.fixture.fixed, "fixed-bin"),
                (self.fixture.clip, "clip3d")):
            mcpat_dir = point / "mcpat"
            mcpat_dir.mkdir(parents=True, exist_ok=True)
            (mcpat_dir / "input.xml").write_text(
                "<component/>", encoding="utf-8",
            )
            write_json(mcpat_dir / "mapping_report.json", {})
            mcpat_output = mcpat_dir / "mcpat.out"
            mcpat_output.write_text(native_text(), encoding="utf-8")
            mcpat = native_mcpat_artifact(binary, mcpat_output, metadata)
            mcpat_path = mcpat_dir / "mcpat.json"
            write_json(mcpat_path, mcpat)
            model = build_model(
                self.fixture.r1, mcpat_path, point / "modules.json",
                require_communication_profile=True,
            )
            layout_modules = deepcopy(model["modules"])
            for module in layout_modules:
                module["tier"] = (
                    1 if module["kind"] in ("l2", "interconnect") else 0
                )
            write_json(point / "hotspot/layout.json", {
                "modules": layout_modules,
            })
            build_vector(
                point / "modules.json", point / "r2_latency.json",
                tsv_hops=1, wire_cycles=3,
            )
            if method == "clip3d":
                write_json(point / "optimizer_report.json", {})
                write_json(point / "layout_selection.json", {})

            frequency = self.fixture.config["frequency"]
            thermal = read_json(point / "hotspot/thermal_result.json")
            performance = derive(
                model, thermal, frequency["f0_ghz"], frequency["fmin_ghz"],
                frequency["tsafe_c"], frequency["ambient_c"],
            )
            write_json(point / "performance.json", performance)

            summary_path = point / "pipeline_summary.json"
            summary = read_json(summary_path)
            summary.update({
                "r1": str(self.fixture.r1.resolve()),
                "module_count": len(model["modules"]),
                "total_power_w": model["totals"]["total_power_w"],
                "power_provenance": model["power_provenance"],
                "area_provenance": model["area_provenance"],
                "cache_authority": model["cache_authority"],
                "mcpat_provenance": model["mcpat_provenance"],
                "communication_profile": model["communication_profile"],
                "gamma": model["gamma"],
                "tmax_c": performance["tmax_f0_c"],
                "sustainable_frequency_ghz": performance[
                    "sustainable_frequency_ghz"
                ],
                "ipc1": performance["ipc1"],
                "bips1_thermal": performance["bips1_thermal"],
            })
            summary["stage_seconds"].pop("cacti", None)
            summary["artifacts"].update({
                "mcpat_xml": str((mcpat_dir / "input.xml").resolve()),
                "mcpat_json": str(mcpat_path.resolve()),
                "mcpat_output": str(mcpat_output.resolve()),
                "mcpat_binary": str(binary.resolve()),
            })
            summary["artifact_sha256"] = {
                "mcpat_json": sha256_file(mcpat_path),
                "mcpat_output": mcpat["provenance"]["hashes"]["output_sha256"],
                "mcpat_binary": mcpat["provenance"]["hashes"]["binary_sha256"],
            }
            write_json(summary_path, summary)

        self.fixture._write_source_result()
        attach_result.attach(self.fixture.fixed)

    def test_paired_sweep_requires_native_physical_preflight_authority(self):
        """A historical preflight result cannot authorize corrected R2 pairing."""
        from workflow.r2.run_paired_sweep import _require_native_preflight

        with self.assertRaisesRegex(ValueError, "McPAT-native"):
            _require_native_preflight({"mode": "resume"})
        accepted = {
            "mode": "resume",
            "fixed": {
                "physical_model_authority": "McPAT 1.3 embedded CACTI-P",
            },
            "clip3d": {
                "physical_model_authority": "McPAT 1.3 embedded CACTI-P",
            },
        }
        self.assertIs(_require_native_preflight(accepted), accepted)

    def test_direct_pair_rejects_legacy_roots_before_any_mutation(self):
        """A public single-pair call cannot bypass native physical preflight."""
        from workflow.r2.run_paired_sweep import run_pair

        def snapshot(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
            }

        for point in (self.fixture.fixed, self.fixture.clip):
            modules = read_json(point / "modules.json")
            modules["schema_version"] = 2
            write_json(point / "modules.json", modules)
            summary = read_json(point / "pipeline_summary.json")
            summary["stage_seconds"]["cacti"] = 1.0
            write_json(point / "pipeline_summary.json", summary)

        before = {
            "fixed": snapshot(self.fixed_root),
            "clip": snapshot(self.clip_root),
        }
        with patch(
            "workflow.r2.run_paired_sweep.run_r2.run",
            side_effect=AssertionError("legacy roots must not execute R2"),
        ), patch(
            "workflow.r2.run_paired_sweep.attach_result.attach",
            side_effect=AssertionError("legacy roots must not attach R2"),
        ), self.assertRaisesRegex(ValueError, "McPAT-native"):
            run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(snapshot(self.fixed_root), before["fixed"])
        self.assertEqual(snapshot(self.clip_root), before["clip"])
        self.assertFalse(self.status_root.exists())

    def test_direct_pair_rejects_native_points_from_different_r1(self):
        """Direct admission binds physical evidence to the caller's R1 point."""
        from workflow.r2.run_paired_sweep import run_pair

        def snapshot(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
            }

        alternate_r1_root = self.root / "alternate-r1"
        shutil.copytree(
            self.fixture.r1,
            alternate_r1_root / self.key.relative_path(),
        )
        for point in (self.fixture.fixed, self.fixture.clip):
            summary_path = point / "pipeline_summary.json"
            summary = read_json(summary_path)
            summary["ipc2"] = None
            summary["bips2"] = None
            write_json(summary_path, summary)

        before = {
            "fixed": snapshot(self.fixed_root),
            "clip": snapshot(self.clip_root),
        }
        with patch(
            "workflow.r2.run_paired_sweep.run_r2.run",
            side_effect=AssertionError("wrong-R1 roots must not execute R2"),
        ), patch(
            "workflow.r2.run_paired_sweep.attach_result.attach",
            side_effect=AssertionError("wrong-R1 roots must not attach R2"),
        ), self.assertRaisesRegex(ValueError, "R1"):
            run_pair(
                self.key, alternate_r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(snapshot(self.fixed_root), before["fixed"])
        self.assertEqual(snapshot(self.clip_root), before["clip"])
        self.assertFalse(self.status_root.exists())

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

    def test_scheduler_rejects_type_loose_vector_before_branch_selection(self):
        """Pair branching must use the same strict vector contract as gem5/reuse."""
        from workflow.r2.run_paired_sweep import _validated_vectors

        vector_path = self.fixture.clip / "r2_latency.json"
        vector = read_json(vector_path)
        vector["gem5_overrides"]["xbar_forward_latency"] = 8.0
        vector["gem5_args"][21] = "8.0"
        write_json(vector_path, vector)

        with self.assertRaisesRegex(ValueError, "non-boolean integer"):
            _validated_vectors(self.fixture.fixed, self.fixture.clip)

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

    def test_completed_local_pair_rejects_a_different_selected_config_source(self):
        """A self-coherent local attachment must still bind the selected config."""
        from workflow.r2.run_paired_sweep import run_pair

        self._make_vectors_unequal()
        with patch("workflow.r2.run_r2.subprocess.run",
                   side_effect=self._complete_gem5):
            first = run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )
        self.assertEqual(first["state"], "success")
        run_config_path = self.fixture.clip / "run_config.json"
        run_config = read_json(run_config_path)
        run_config["source"] = str((self.root / "different-config.json").resolve())
        write_json(run_config_path, run_config)
        status_path = self.status_root / self.key.relative_path() / "pair_status.json"
        status_before = status_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "run_config source"):
            run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(status_path.read_bytes(), status_before)

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

    def test_local_resume_rejects_all_physical_and_derived_mutations(self):
        """One physical validator binds modules, thermal, performance, and summary."""
        from workflow.r2.attach_result import attach
        from workflow.r2.run_paired_sweep import _validate_local_attachment

        attach(self.fixture.fixed)
        point = self.fixture.fixed
        paths = {
            "modules": point / "modules.json",
            "thermal": point / "hotspot/thermal_result.json",
            "performance": point / "performance.json",
            "summary": point / "pipeline_summary.json",
        }
        originals = {name: path.read_bytes() for name, path in paths.items()}

        def module_ipc1(values):
            values["modules"]["ipc1"] = 99.0

        def module_gamma(values):
            values["modules"]["gamma"] = 0.9

        def module_architecture(values):
            values["modules"]["architecture"]["l1d_size"] = "128kB"

        def thermal_tmax(values):
            values["thermal"]["tmax_c"] = 999.0
            values["thermal"]["tmax_k"] = 1272.15

        def thermal_provenance(values):
            values["thermal"]["steady_file"] = str(
                (self.root / "forged.steady").resolve()
            )

        def performance_frequency(values):
            values["performance"]["sustainable_frequency_ghz"] = 9.0
            values["performance"]["bips2"] = 3.25 * 9.0
            values["summary"]["sustainable_frequency_ghz"] = 9.0
            values["summary"]["bips2"] = 3.25 * 9.0

        def performance_bips(values):
            values["performance"]["sustainable_frequency_ghz"] = 8.0
            values["performance"]["bips2"] = 26.0
            values["summary"]["sustainable_frequency_ghz"] = 8.0
            values["summary"]["bips2"] = 26.0

        def summary_tmax(values):
            values["summary"]["tmax_c"] = 999.0

        def summary_gamma(values):
            values["summary"]["gamma"] = 0.9

        def summary_ipc1(values):
            values["summary"]["ipc1"] = 99.0
            values["summary"]["bips1_thermal"] = 99.0 * values[
                "summary"
            ]["sustainable_frequency_ghz"]

        cases = {
            "module-ipc1": module_ipc1,
            "module-gamma": module_gamma,
            "module-architecture": module_architecture,
            "thermal-tmax": thermal_tmax,
            "thermal-provenance": thermal_provenance,
            "performance-frequency": performance_frequency,
            "performance-bips": performance_bips,
            "summary-tmax": summary_tmax,
            "summary-gamma": summary_gamma,
            "summary-ipc1": summary_ipc1,
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                for name, data in originals.items():
                    paths[name].write_bytes(data)
                values = {name: read_json(path) for name, path in paths.items()}
                mutate(values)
                for name, value in values.items():
                    write_json(paths[name], value)

                self.assertIsNone(
                    _validate_local_attachment(self.fixture.r1, point, "fixed-bin")
                )

        for name, data in originals.items():
            paths[name].write_bytes(data)
        self.assertIsNotNone(
            _validate_local_attachment(self.fixture.r1, point, "fixed-bin")
        )

    def test_scheduler_metrics_use_the_shared_validator_snapshot(self):
        """A post-validation summary replacement cannot supply scheduler metrics."""
        import workflow.r2.run_paired_sweep as paired
        from workflow.r2.attach_result import attach, validate_local_attachment

        attach(self.fixture.fixed)
        decision = validate_local_attachment(
            self.fixture.fixed, self.fixture.r1, self.config_path
        )
        self.assertTrue(decision["accepted"], decision["reasons"])
        expected_bips2 = decision["summary"]["bips2"]
        summary_path = self.fixture.fixed / "pipeline_summary.json"
        replaced = read_json(summary_path)
        replaced["bips2"] = 999.0
        write_json(summary_path, replaced)

        with patch.object(
                paired.attach_result, "validate_local_attachment",
                return_value=decision):
            validated = paired._validate_local_attachment(
                self.fixture.r1, self.fixture.fixed, "fixed-bin",
                self.config_path,
            )

        self.assertIsNotNone(validated)
        self.assertEqual(validated[1]["bips2"], expected_bips2)

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
        repaired_performance = read_json(performance_path)
        self.assertAlmostEqual(
            repaired_performance["bips2"],
            repaired_performance["ipc2"]
            * repaired_performance["sustainable_frequency_ghz"],
        )

    def test_sweep_lock_fails_fast_before_preflight_or_worker_launch(self):
        """A second sweep cannot inspect or mutate roots owned by an active sweep."""
        import fcntl
        import os
        import workflow.r2.run_paired_sweep as paired

        lock_path = self.fixed_root / ".paired-r2-sweep.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            with patch.object(paired, "load_selection", return_value={}), \
                    patch.object(paired, "selection_keys", return_value=[]), \
                    patch.object(paired, "validate_layout_roots",
                                 return_value=NATIVE_PREFLIGHT), \
                    self.assertRaisesRegex(RuntimeError, "sweep.*active|lock"):
                paired.run_sweep(
                    self.r1_root, self.fixed_root, self.clip_root,
                    self.root / "selection.json", self.config_path,
                    self.status_root,
                )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_sweep_lock_remains_held_while_pair_workers_run(self):
        """The sweep lock must enclose worker execution and status publication."""
        from concurrent.futures import Future
        import fcntl
        import os
        import workflow.r2.run_paired_sweep as paired

        observations = []
        publication_observations = []
        status_root = self.status_root
        lock_paths = [
            self.fixed_root / ".paired-r2-sweep.lock",
            self.clip_root / ".paired-r2-sweep.lock",
        ]

        def lock_is_held(path):
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                return False
            finally:
                os.close(descriptor)

        class InspectingExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _function, key, *_args, **_kwargs):
                observations.append(all(lock_is_held(path) for path in lock_paths))
                pair_path = status_root / key.relative_path() / "pair_status.json"
                record = {
                    "schema_version": 1,
                    "state": "failed",
                    "key": {"workload": key.workload,
                            "l1d_size": key.l1d_size,
                            "l2_size": key.l2_size},
                    "pair_status": str(pair_path.resolve()),
                    "fixed_complete": False,
                    "physical_r2_runs": 0,
                    "error": "intentional fixture failure",
                }
                write_json(pair_path, record)
                future = Future()
                future.set_result(record)
                return future

        real_write = paired.write_json

        def observe_publication(path, value):
            if Path(path).resolve() == (status_root / "status.json").resolve():
                publication_observations.append(
                    all(lock_is_held(lock_path) for lock_path in lock_paths)
                )
            return real_write(path, value)

        with patch.object(paired, "load_selection", return_value={}), \
                patch.object(paired, "selection_keys", return_value=[self.key]), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(paired, "ProcessPoolExecutor", InspectingExecutor), \
                patch.object(paired, "write_json", side_effect=observe_publication), \
                redirect_stdout(StringIO()):
            paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root,
                self.root / "selection.json", self.config_path,
                self.status_root, limit=1,
            )

        self.assertEqual(observations, [True])
        self.assertTrue(publication_observations)
        self.assertTrue(all(publication_observations))

    def test_pair_lock_serializes_before_cached_status_validation(self):
        """A second direct pair attempt must wait before trusting shared status."""
        from concurrent.futures import ThreadPoolExecutor
        import fcntl
        import os
        import threading
        import workflow.r2.run_paired_sweep as paired

        pair_dir = self.status_root / self.key.relative_path()
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_path = pair_dir / "pair_status.json"
        previous = {"schema_version": 1, "state": "success", "sentinel": True}
        write_json(pair_path, previous)
        lock_path = self.fixture.fixed / ".paired-r2-pair.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        attempted = threading.Event()
        validation_started = threading.Event()

        def invoke():
            attempted.set()
            return paired.run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        def validate(*_args, **_kwargs):
            validation_started.set()
            return True

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            with patch.object(paired, "_validate_completed", side_effect=validate):
                future = executor.submit(invoke)
                self.assertTrue(attempted.wait(1))
                reached_while_locked = validation_started.wait(0.2)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                self.assertTrue(validation_started.wait(1))
                result = future.result(timeout=2)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                executor.shutdown(wait=True)

        self.assertFalse(reached_while_locked)
        self.assertEqual(result, previous)

    def test_pair_lock_identity_is_shared_across_status_roots(self):
        """Different status destinations cannot permit overlapping physical work."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import workflow.r2.run_paired_sweep as paired

        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        active = 0
        maximum_active = 0

        def inspect_body(*_args, status_root=None, **_kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                if Path(status_root).name == "status-a":
                    first_entered.set()
                    self.assertTrue(release_first.wait(2))
                else:
                    second_entered.set()
                return {"state": "success", "status_root": str(status_root)}
            finally:
                active -= 1

        with patch.object(paired, "_run_pair_locked", side_effect=inspect_body), \
                ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                paired.run_pair, self.key, self.r1_root, self.fixed_root,
                self.clip_root, self.config_path,
                status_root=self.root / "status-a",
            )
            self.assertTrue(first_entered.wait(1))
            second = executor.submit(
                paired.run_pair, self.key, self.r1_root, self.fixed_root,
                self.clip_root, self.config_path,
                status_root=self.root / "status-b",
            )
            overlapped = second_entered.wait(0.2)
            release_first.set()
            first.result(timeout=2)
            second.result(timeout=2)

        self.assertFalse(overlapped)
        self.assertEqual(maximum_active, 1)

    def test_direct_pair_waits_until_physical_sweep_lock_is_released(self):
        """A direct pair cannot mutate evidence during final sweep publication."""
        from concurrent.futures import ThreadPoolExecutor
        import fcntl
        import os
        import threading
        import workflow.r2.run_paired_sweep as paired

        lock_path = self.fixed_root / ".paired-r2-sweep.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        attempted = threading.Event()
        entered = threading.Event()

        def invoke():
            attempted.set()
            return paired.run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        def inspect_body(*_args, **_kwargs):
            entered.set()
            return {"state": "success"}

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            with patch.object(paired, "_run_pair_locked",
                              side_effect=inspect_body):
                future = executor.submit(invoke)
                self.assertTrue(attempted.wait(1))
                reached_while_locked = entered.wait(0.2)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                self.assertTrue(entered.wait(1))
                future.result(timeout=2)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                executor.shutdown(wait=True)

        self.assertFalse(reached_while_locked)

    def test_pair_lock_precedes_each_physical_attachment_lock(self):
        """Local and reuse attachment boundaries must run inside the pair lock."""
        import fcntl
        import os
        import workflow.r2.run_paired_sweep as paired

        lock_path = self.fixture.fixed / ".paired-r2-pair.lock"
        observations = []
        real_local_attach = paired.attach_result.attach
        real_reuse_attach = paired.reuse_result.attach_reused_result

        def pair_lock_is_held():
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                return False
            finally:
                os.close(descriptor)

        def inspect_local(*args, **kwargs):
            observations.append(("local", pair_lock_is_held()))
            return real_local_attach(*args, **kwargs)

        def inspect_reuse(*args, **kwargs):
            observations.append(("reuse", pair_lock_is_held()))
            return real_reuse_attach(*args, **kwargs)

        with patch.object(paired.attach_result, "attach",
                          side_effect=inspect_local), \
                patch.object(paired.reuse_result, "attach_reused_result",
                             side_effect=inspect_reuse):
            result = paired.run_pair(
                self.key, self.r1_root, self.fixed_root, self.clip_root,
                self.config_path, status_root=self.status_root,
            )

        self.assertEqual(result["state"], "success")
        self.assertEqual(observations, [("local", True), ("reuse", True)])

    def test_clear_reuse_attachment_holds_the_physical_point_lock(self):
        """Clearing reuse state is one serialized physical-point mutation."""
        from concurrent.futures import ThreadPoolExecutor
        import fcntl
        import os
        import threading
        import workflow.r2.run_paired_sweep as paired

        descriptor = os.open(
            self.fixture.clip, os.O_RDONLY | os.O_DIRECTORY
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        attempted = threading.Event()
        mutation_started = threading.Event()
        real_read = paired.read_json

        def observe_read(path):
            mutation_started.set()
            return real_read(path)

        def invoke():
            attempted.set()
            paired._clear_reuse_attachment(self.fixture.clip)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            with patch.object(paired, "read_json", side_effect=observe_read):
                future = executor.submit(invoke)
                self.assertTrue(attempted.wait(1))
                reached_while_locked = mutation_started.wait(0.2)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                self.assertTrue(mutation_started.wait(1))
                future.result(timeout=2)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                executor.shutdown(wait=True)

        self.assertFalse(reached_while_locked)

    def test_final_status_rejects_worker_return_after_persisted_tamper(self):
        """A worker's return value cannot certify evidence changed on disk."""
        from concurrent.futures import Future
        import workflow.r2.run_paired_sweep as paired

        status_root = self.status_root

        class TamperingExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args, **kwargs):
                returned = function(*args, **kwargs)
                pair_path = Path(returned["pair_status"])
                persisted = read_json(pair_path)
                persisted["fixed_bips2"] = 999.0
                write_json(pair_path, persisted)
                future = Future()
                future.set_result(deepcopy(returned))
                return future

        with patch.object(paired, "load_selection", return_value={}), \
                patch.object(paired, "selection_keys", return_value=[self.key]), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(paired, "ProcessPoolExecutor", TamperingExecutor), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root,
                self.root / "selection.json", self.config_path,
                status_root, limit=1,
            )

        pair_path = status_root / self.key.relative_path() / "pair_status.json"
        self.assertEqual(read_json(pair_path)["state"], "success")
        self.assertFalse(result["selected_results_valid"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["pairs"][0]["state"], "failed")
        self.assertEqual(result["completion_revalidation"], {
            "performed": True,
            "accepted_pair_count": 0,
            "rejected_pair_count": 1,
            "rejections": [{
                "key": {"workload": "fft", "l1d_size": "32kB",
                        "l2_size": "512kB"},
                "pair_status": str(pair_path.resolve()),
                "reason": "persisted success record failed live evidence validation",
            }],
        })

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
                pair_path = status_root / key.relative_path() / "pair_status.json"
                record = {
                    "schema_version": 1,
                    "state": state,
                    "key": {
                        "workload": key.workload,
                        "l1d_size": key.l1d_size,
                        "l2_size": key.l2_size,
                    },
                    "clip3d_reused_fixed_r2": state == "success",
                    "physical_r2_runs": 1 if state == "success" else 0,
                    "pair_status": str(pair_path.resolve()),
                }
                write_json(pair_path, record)
                future.set_result(record)
                return future

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(
                    paired, "_validate_completed",
                    side_effect=lambda record, *_args: (
                        isinstance(record, dict)
                        and record.get("state") == "success"
                    ),
                ):
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
                    pair_path = (
                        status_root / key.relative_path() / "pair_status.json"
                    )
                    record = {
                        "schema_version": 1,
                        "state": "success",
                        "key": {"workload": key.workload,
                                "l1d_size": key.l1d_size,
                                "l2_size": key.l2_size},
                        "clip3d_reused_fixed_r2": True,
                        "physical_r2_runs": 1,
                        "pair_status": str(pair_path.resolve()),
                    }
                    write_json(pair_path, record)
                    future.set_result(record)
                return future

        real_write = paired.write_json
        experiment_snapshots = []

        def capture_status(path, value):
            real_write(path, value)
            if Path(path).resolve() == (status_root / "status.json").resolve():
                experiment_snapshots.append(deepcopy(value))

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(
                    paired, "_validate_completed",
                    side_effect=lambda record, *_args: (
                        isinstance(record, dict)
                        and record.get("state") == "success"
                    ),
                ), \
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
            return NATIVE_PREFLIGHT

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
            return NATIVE_PREFLIGHT

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
                pair_path = status_root / key.relative_path() / "pair_status.json"
                record = {
                    "schema_version": 1,
                    "state": "success",
                    "key": {"workload": key.workload,
                            "l1d_size": key.l1d_size, "l2_size": key.l2_size},
                    "clip3d_reused_fixed_r2": True,
                    "physical_r2_runs": 1,
                    "pair_status": str(pair_path.resolve()),
                }
                write_json(pair_path, record)
                future = Future()
                future.set_result(record)
                return future

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(
                    paired, "_validate_completed",
                    side_effect=lambda record, *_args: (
                        isinstance(record, dict)
                        and record.get("state") == "success"
                    ),
                ):
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
        config_path = self.config_path
        r1_root = self.r1_root
        fixed_root = self.fixed_root
        clip_root = self.clip_root

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
                pair_path = (
                    status_root / key.relative_path() / "pair_status.json"
                )
                record = {
                    "schema_version": 1,
                    "state": "success",
                    "phase": "complete",
                    "key": {"workload": key.workload,
                            "l1d_size": key.l1d_size,
                            "l2_size": key.l2_size},
                    "config": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "r1_directory": str(
                        (r1_root / key.relative_path()).resolve()
                    ),
                    "fixed_point": str(
                        (fixed_root / key.relative_path()).resolve()
                    ),
                    "clip3d_point": str(
                        (clip_root / key.relative_path()).resolve()
                    ),
                    "fixed_complete": True,
                    "fixed_ipc2": 2.0,
                    "fixed_bips2": 3.0,
                    "clip3d_ipc2": 2.0,
                    "clip3d_bips2": 3.0,
                    "clip3d_reused_fixed_r2": not separate,
                    "physical_r2_runs": 2 if separate else 1,
                    "pair_status": str(pair_path.resolve()),
                }
                write_json(pair_path, record)
                future = Future()
                future.set_result(record)
                return future

        def validate_persisted(record, key, paths, config_path):
            import fcntl
            import os

            lock_path = paths["fixed"] / ".paired-r2-pair.lock"
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pair_lock_held = True
                else:
                    pair_lock_held = False
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            return (
                pair_lock_held
                and record == read_json(paths["status"])
                and record.get("state") == "success"
                and record.get("key") == {
                    "workload": key.workload,
                    "l1d_size": key.l1d_size,
                    "l2_size": key.l2_size,
                }
                and record.get("config") == str(config_path)
                and record.get("pair_status") == str(paths["status"])
            )

        real_write = paired.write_json
        experiment_snapshots = []

        def capture_experiment_status(path, value):
            real_write(path, value)
            if Path(path).resolve() == (status_root / "status.json").resolve():
                experiment_snapshots.append(deepcopy(value))

        with patch.object(paired, "ProcessPoolExecutor", ImmediateExecutor), \
                patch.object(paired, "validate_layout_roots",
                             return_value=NATIVE_PREFLIGHT), \
                patch.object(paired, "_validate_completed",
                             side_effect=validate_persisted) as validate_completed, \
                patch.object(paired, "write_json",
                             side_effect=capture_experiment_status), \
                redirect_stdout(StringIO()):
            result = paired.run_sweep(
                self.r1_root, self.fixed_root, self.clip_root, selection_path,
                self.config_path, self.status_root, jobs=4,
            )

        self.assertTrue(experiment_snapshots[-1]["complete"])
        self.assertFalse(any(
            snapshot["complete"] for snapshot in experiment_snapshots[:-1]
        ))
        self.assertEqual(validate_completed.call_count, 50)
        self.assertTrue(result["complete"])
        self.assertEqual(result["selected_pair_count"], 50)
        self.assertEqual(result["completed_pair_count"], 50)
        self.assertEqual(result["separate_clip_r2_count"], 7)
        self.assertEqual(result["reuse_pair_count"], 43)
        self.assertEqual(result["physical_r2_runs"], 57)
        self.assertEqual(len(result["pairs"]), 50)
        self.assertEqual(result["completion_revalidation"], {
            "performed": True,
            "accepted_pair_count": 50,
            "rejected_pair_count": 0,
            "rejections": [],
        })

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

class PairedAggregationTests(unittest.TestCase):
    """Exercise strict paired reporting with a complete tracked 50-point sample."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.r1_root = self.root / "canonical-r1"
        self.fixed_root = self.root / "fixed_bin"
        self.clip_root = self.root / "clip3d"
        self.status_root = self.root / "paired_r2_status"
        project = Path(__file__).resolve().parents[1]
        self.selection_path = project / "configs/experiments/balanced50_traffic_weighted.json"
        self.config_path = project / (
            "configs/experiments/"
            "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
            "traffic_weighted_exploratory.json"
        )
        self.config = read_json(self.config_path)
        self.classification = self.config["experiment_classification"]
        self.mcpat_binary = self.root / "tools/mcpat"
        self.mcpat_binary.parent.mkdir(parents=True)
        self.mcpat_binary.write_bytes(b"strict patched McPAT report fixture\n")

    def _write_point(self, root: Path, key: dict, method: str, ipc2: float,
                     tmax_c: float) -> Path:
        from workflow.r2 import run_r2
        from workflow.thermal.sustainable_frequency import derive

        r1 = self.r1_root / key["workload"] / f"l1d_{key['l1d_size']}" / (
            f"l2_{key['l2_size']}"
        )
        r1.mkdir(parents=True, exist_ok=True)
        metadata = {
            **key,
            "l1i_size": "32kB",
            "num_cores": 4,
            "cpu_clock": "2GHz",
            "clock": "2GHz",
            "l1_associativity": 2,
            "l2_associativity": 8,
            "cache_line_bytes": 64,
            "instruction_window_scope": "cpu0",
            "warmup_insts_cpu0": 0,
            "measure_insts_cpu0": 100,
            "command": [],
        }
        write_json(r1 / "r1_metadata.json", metadata)
        (r1 / "stats.txt").write_text(
            "---------- Begin Simulation Statistics ----------\n"
            + "".join(
                f"system.cpu{core}.commitStats0.numInsts 100\n"
                f"system.cpu{core}.numCycles 50\n"
                f"system.l2.demandAccesses::cpu{core}.data {100 + core}\n"
                for core in range(4)
            ), encoding="utf-8",
        )
        point = root / key["workload"] / f"l1d_{key['l1d_size']}" / (
            f"l2_{key['l2_size']}"
        )
        point.mkdir(parents=True, exist_ok=True)
        from tests.test_mcpat_native_cache import write_native_physical_fixture
        from workflow.r2.build_latency_vector import build_vector

        mcpat, modules = write_native_physical_fixture(
            point, r1, self.mcpat_binary, metadata,
        )
        vector = build_vector(
            point / "modules.json", point / "r2_latency.json",
            tsv_hops=1, wire_cycles=3,
        )
        vector["critical_l1d_to_l2_cycles"] = (
            17 if method == "fixed-bin" else 19
        )
        write_json(point / "r2_latency.json", vector)
        ipc1 = modules["ipc1"]
        hotspot = point / "hotspot"
        hotspot.mkdir(parents=True, exist_ok=True)
        peak_k = tmax_c + 273.15
        (hotspot / "steady.txt").write_text(
            f"core0 {peak_k:.17g}\n", encoding="utf-8"
        )
        write_json(hotspot / "hotspot_manifest.json", {
            "schema_version": 1,
            "ambient_c": self.config["frequency"]["ambient_c"],
            "r_convec_k_per_w": self.config["physical"]["r_convec_k_per_w"],
        })
        thermal = {
            "schema_version": 1,
            "return_code": 0,
            "power_trace": str((point / "hotspot/power.ptrace").resolve()),
            "steady_file": str((point / "hotspot/steady.txt").resolve()),
            "grid_steady_file": str((point / "hotspot/grid.steady.txt").resolve()),
            "tmax_k": peak_k,
            "tmax_c": tmax_c,
            "peak_unit": "core0",
            "sample_count": 1,
            "ambient_c": self.config["frequency"]["ambient_c"],
            "r_convec_k_per_w": self.config["physical"]["r_convec_k_per_w"],
        }
        performance = derive(
            modules, thermal, **{
                "f0_ghz": self.config["frequency"]["f0_ghz"],
                "fmin_ghz": self.config["frequency"]["fmin_ghz"],
                "tsafe_c": self.config["frequency"]["tsafe_c"],
                "ambient_c": self.config["frequency"]["ambient_c"],
            }, ipc2=ipc2,
        )
        frequency = performance["sustainable_frequency_ghz"]
        bips2 = performance["bips2"]
        local_result = point / "gem5_r2/r2_result.json"
        local_result.parent.mkdir(exist_ok=True)
        r2_stats = local_result.parent / "stats.txt"
        instructions = int(ipc2 * 25)
        r2_stats.write_text(
            "---------- Begin Simulation Statistics ----------\n"
            + "".join(
                f"system.cpu{core}.commitStats0.numInsts {instructions}\n"
                f"system.cpu{core}.numCycles 100\n"
                for core in range(4)
            ), encoding="utf-8",
        )
        provenance = run_r2._provenance(r1, point / "r2_latency.json")
        per_core = [
            {"core": core, "instructions": instructions, "cycles": 100,
             "ipc": instructions / 100}
            for core in range(4)
        ]
        command = [
            "/tmp/gem5", "--listener-mode=off",
            f"--outdir={local_result.parent.resolve()}", "/tmp/clip_r1.py",
            *run_r2._command_tail(metadata, vector),
        ]
        result = {
            "schema_version": 3, "command": command, **provenance,
            "ipc2": ipc2, "per_core": per_core,
            "stats": str(r2_stats.resolve()), "stats_sha256": sha256_file(r2_stats),
            "elapsed_seconds": 1.0,
        }
        write_json(local_result, result)
        write_json(point / "gem5_r2/status.json", {
            "schema_version": 3, "state": "success", "return_code": 0,
            "command": command, **provenance, "ipc2": ipc2,
            "stats": str(r2_stats.resolve()), "stats_sha256": sha256_file(r2_stats),
            "r2_result": str(local_result.resolve()),
            "r2_result_sha256": sha256_file(local_result),
        })
        write_json(point / "run_config.json", {
            "layout_method": method,
            "config": self.config,
            "source": str(self.config_path.resolve()),
        })
        write_json(point / "modules.json", modules)
        write_json(point / "hotspot/thermal_result.json", thermal)
        write_json(point / "performance.json", performance)
        write_json(point / "pipeline_summary.json", {
            "schema_version": 3,
            "layout_method": method,
            "experiment_classification": self.classification,
            "workload": key["workload"], "l1d_size": key["l1d_size"],
            "l2_size": key["l2_size"], "r1": str(r1.resolve()),
            "output": str(point.resolve()),
            "gamma": performance["gamma"],
            "tmax_c": tmax_c,
            "sustainable_frequency_ghz": frequency,
            "ipc1": performance["ipc1"],
            "bips1_thermal": performance["bips1_thermal"],
            "module_count": len(modules["modules"]),
            "total_power_w": modules["totals"]["total_power_w"],
            "power_provenance": modules["power_provenance"],
            "area_provenance": modules["area_provenance"],
            "cache_authority": modules["cache_authority"],
            "mcpat_provenance": modules["mcpat_provenance"],
            "communication_profile": modules["communication_profile"],
            "r2_critical_path_cycles": vector["critical_l1d_to_l2_cycles"],
            "ipc2": ipc2,
            "bips2": bips2,
            "r2_source": str(local_result.resolve()),
            "stage_seconds": {"layout": 1.0, "gem5_r2": 1.0},
            "total_pipeline_seconds": 2.0,
            "artifacts": {
                "config": str((point / "run_config.json").resolve()),
                "modules": str((point / "modules.json").resolve()),
                "thermal": str((point / "hotspot/thermal_result.json").resolve()),
                "performance": str((point / "performance.json").resolve()),
                "r2_latency": str((point / "r2_latency.json").resolve()),
                "r2_result": str(local_result.resolve()),
                "mcpat_json": str((point / "mcpat/mcpat.json").resolve()),
                "mcpat_output": str((point / "mcpat/mcpat.out").resolve()),
                "mcpat_binary": str(self.mcpat_binary.resolve()),
            },
            "artifact_sha256": {
                "mcpat_json": sha256_file(point / "mcpat/mcpat.json"),
                "mcpat_output": mcpat["provenance"]["hashes"][
                    "output_sha256"
                ],
                "mcpat_binary": mcpat["provenance"]["hashes"][
                    "binary_sha256"
                ],
            },
        })
        return point

    def _make_complete_fixture(self) -> tuple[Path, Path]:
        selection = read_json(self.selection_path)
        for index, key in enumerate(selection["points"]):
            fixed = self._write_point(
                self.fixed_root, key, "fixed-bin", 2.0,
                80.123456 + index * 0.1,
            )
            clip = self._write_point(
                self.clip_root, key, "clip3d", 2.4,
                81.654321 + index * 0.1,
            )
            fixed_summary = read_json(fixed / "pipeline_summary.json")
            clip_summary = read_json(clip / "pipeline_summary.json")
            status_path = self.status_root / key["workload"] / (
                f"l1d_{key['l1d_size']}"
            ) / f"l2_{key['l2_size']}" / "pair_status.json"
            write_json(status_path, {
                "schema_version": 1,
                "state": "success",
                "key": key,
                "config": str(self.config_path.resolve()),
                "config_sha256": sha256_file(self.config_path),
                "r1_directory": str((self.r1_root / key["workload"] /
                                     f"l1d_{key['l1d_size']}" /
                                     f"l2_{key['l2_size']}").resolve()),
                "fixed_point": str(fixed.resolve()),
                "clip3d_point": str(clip.resolve()),
                "fixed_vector_sha256": sha256_file(fixed / "r2_latency.json"),
                "clip3d_vector_sha256": sha256_file(clip / "r2_latency.json"),
                "fixed_ipc2": 2.0,
                "fixed_bips2": fixed_summary["bips2"],
                "clip3d_ipc2": 2.4,
                "clip3d_bips2": clip_summary["bips2"],
                "clip3d_reused_fixed_r2": False,
                "physical_r2_runs": 2,
                "pair_status": str(status_path.resolve()),
            })
        return self.root / "paired_results.csv", self.root / "paired_summary.json"

    def _attach_real_reuse_for_selected_pair(self) -> tuple[dict, Path]:
        """Install one Task-5 fixture under a selected key and commit real reuse."""
        from workflow.r2 import attach_result, reuse_result
        from workflow.thermal.sustainable_frequency import derive

        key = {"workload": "fft", "l1d_size": "64kB", "l2_size": "512kB"}
        relative = Path(key["workload"]) / f"l1d_{key['l1d_size']}" / (
            f"l2_{key['l2_size']}"
        )
        r1 = self.r1_root / relative
        fixed = self.fixed_root / relative
        clip = self.clip_root / relative
        clip_performance = derive(
            read_json(clip / "modules.json"),
            read_json(clip / "hotspot/thermal_result.json"),
            self.config["frequency"]["f0_ghz"],
            self.config["frequency"]["fmin_ghz"],
            self.config["frequency"]["tsafe_c"],
            self.config["frequency"]["ambient_c"],
        )
        write_json(clip / "performance.json", clip_performance)
        clip_summary = read_json(clip / "pipeline_summary.json")
        clip_summary.update({
            "ipc2": None, "bips2": None, "r2_source": None,
            "gamma": clip_performance["gamma"],
            "tmax_c": clip_performance["tmax_f0_c"],
            "sustainable_frequency_ghz": clip_performance[
                "sustainable_frequency_ghz"
            ],
            "ipc1": clip_performance["ipc1"],
            "bips1_thermal": clip_performance["bips1_thermal"],
        })
        clip_summary["stage_seconds"].pop("gem5_r2", None)
        clip_summary["artifacts"]["r2_result"] = None
        write_json(clip / "pipeline_summary.json", clip_summary)

        attach_result.attach(fixed)
        fixed_summary = read_json(fixed / "pipeline_summary.json")
        summary = reuse_result.attach_reused_result(fixed, clip, r1, self.config_path)
        pair_path = self.status_root / relative / "pair_status.json"
        pair = read_json(pair_path)
        pair.update({
            "r1_directory": str(r1.resolve()),
            "fixed_point": str(fixed.resolve()), "clip3d_point": str(clip.resolve()),
            "fixed_vector_sha256": sha256_file(fixed / "r2_latency.json"),
            "clip3d_vector_sha256": sha256_file(clip / "r2_latency.json"),
            "fixed_ipc2": fixed_summary["ipc2"], "fixed_bips2": fixed_summary["bips2"],
            "clip3d_ipc2": summary["ipc2"], "clip3d_bips2": summary["bips2"],
            "clip3d_reused_fixed_r2": True, "physical_r2_runs": 1,
        })
        write_json(pair_path, pair)
        return summary, clip / "r2_reuse.json"

    def test_report_uses_all_fifty_measured_pairs_and_preserves_temperature_precision(self):
        """Dropping a selected row or rounding report temperatures changes the result."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        result = summarize(self.fixed_root, self.clip_root, self.selection_path,
                           self.config_path, csv_path, json_path)

        self.assertEqual(result["point_count"], 50)
        self.assertEqual(set(result["workloads"]),
                         {"fft", "cholesky", "stream", "matmul", "stencil"})
        self.assertTrue(all(item["n"] == 10
                            for item in result["workloads"].values()))
        self.assertEqual(result["score_definition"],
                         "paired measured BIPS2=IPC2*f_sus; exact validated reuse allowed")
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["aggregate"]["arithmetic_mean_ratio"], 1.2)
        self.assertAlmostEqual(result["aggregate"]["geometric_mean_ratio"], 1.2)
        self.assertAlmostEqual(result["aggregate"]["median_percent_change"], 20.0)
        self.assertEqual(result["aggregate"]["wins"], 50)

        rows = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertIn("fixed_tmax_c", rows[0])
        self.assertIn("clip3d_tmax_c", rows[0])
        self.assertIn("fixed_wire_cycles", rows[0])
        self.assertIn("clip3d_vector_sha256", rows[0])
        self.assertIn("clip3d_reused_fixed_r2", rows[0])
        self.assertIn("absolute_bips2_difference", rows[0])
        self.assertIn("percent_change", rows[0])
        self.assertIn("80.123456", rows[1])
        self.assertIn("81.654321", rows[1])
        self.assertEqual(read_json(json_path), result)

    def test_report_accepts_the_runner_explicit_status_root(self):
        """Reporting must consume the same caller-selected status namespace."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        alternate = self.root / "alternate-paired-status"
        shutil.copytree(self.status_root, alternate)
        for key in read_json(self.selection_path)["points"]:
            status_path = alternate / key["workload"] / (
                f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
            )
            status = read_json(status_path)
            status["pair_status"] = str(status_path.resolve())
            write_json(status_path, status)

        result = summarize(
            self.fixed_root, self.clip_root, self.selection_path,
            self.config_path, csv_path, json_path, status_root=alternate,
        )

        self.assertTrue(result["complete"])
        self.assertEqual(result["status_root"], str(alternate.resolve()))

    def test_report_rejects_fabricated_summary_and_performance_temperature(self):
        """Matching derived files cannot override the target HotSpot evidence."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        key = read_json(self.selection_path)["points"][0]
        point = self.fixed_root / key["workload"] / (
            f"l1d_{key['l1d_size']}"
        ) / f"l2_{key['l2_size']}"
        from workflow.thermal.sustainable_frequency import derive

        thermal_path = point / "hotspot/thermal_result.json"
        thermal = read_json(thermal_path)
        thermal.update({"tmax_c": 999.0, "tmax_k": 1272.15})
        write_json(thermal_path, thermal)
        modules = read_json(point / "modules.json")
        old_performance = read_json(point / "performance.json")
        frequency = self.config["frequency"]
        performance = derive(
            modules, thermal, frequency["f0_ghz"], frequency["fmin_ghz"],
            frequency["tsafe_c"], frequency["ambient_c"],
            old_performance["ipc2"],
        )
        write_json(point / "performance.json", performance)
        summary = read_json(point / "pipeline_summary.json")
        summary.update({
            "gamma": performance["gamma"],
            "tmax_c": performance["tmax_f0_c"],
            "sustainable_frequency_ghz": performance[
                "sustainable_frequency_ghz"
            ],
            "ipc1": performance["ipc1"],
            "bips1_thermal": performance["bips1_thermal"],
            "ipc2": performance["ipc2"],
            "bips2": performance["bips2"],
        })
        write_json(point / "pipeline_summary.json", summary)
        status_path = self.status_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
        )
        status = read_json(status_path)
        status["fixed_bips2"] = performance["bips2"]
        write_json(status_path, status)

        with self.assertRaisesRegex(ValueError, "thermal|Tmax|physical"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, json_path)

    def test_report_rejects_a_local_snapshot_replaced_between_validation_layers(self):
        """Report metrics and the shared physical decision must bind one snapshot."""
        from workflow.analysis.summarize_paired_sweep import summarize
        import workflow.analysis.summarize_paired_sweep as reporting
        from workflow.thermal.sustainable_frequency import derive

        csv_path, json_path = self._make_complete_fixture()
        key = read_json(self.selection_path)["points"][0]
        point = self.clip_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}"
        )
        real_validate = reporting.attach_result.validate_local_attachment
        replaced = False

        def replace_before_shared_validation(point_dir, *args, **kwargs):
            nonlocal replaced
            if Path(point_dir).resolve() == point.resolve() and not replaced:
                replaced = True
                modules = read_json(point / "modules.json")
                thermal_path = point / "hotspot/thermal_result.json"
                thermal = read_json(thermal_path)
                thermal["tmax_c"] += 0.25
                thermal["tmax_k"] = thermal["tmax_c"] + 273.15
                write_json(thermal_path, thermal)
                (point / "hotspot/steady.txt").write_text(
                    f"{thermal['peak_unit']} {thermal['tmax_k']:.17g}\n",
                    encoding="utf-8",
                )
                old_performance = read_json(point / "performance.json")
                frequency = self.config["frequency"]
                performance = derive(
                    modules, thermal, frequency["f0_ghz"],
                    frequency["fmin_ghz"], frequency["tsafe_c"],
                    frequency["ambient_c"], old_performance["ipc2"],
                )
                write_json(point / "performance.json", performance)
                summary_path = point / "pipeline_summary.json"
                summary = read_json(summary_path)
                summary.update({
                    "gamma": performance["gamma"],
                    "tmax_c": performance["tmax_f0_c"],
                    "sustainable_frequency_ghz": performance[
                        "sustainable_frequency_ghz"
                    ],
                    "ipc1": performance["ipc1"],
                    "bips1_thermal": performance["bips1_thermal"],
                    "ipc2": performance["ipc2"],
                    "bips2": performance["bips2"],
                })
                write_json(summary_path, summary)
            return real_validate(point_dir, *args, **kwargs)

        with patch.object(
                reporting.attach_result, "validate_local_attachment",
                side_effect=replace_before_shared_validation), \
                self.assertRaisesRegex(ValueError, "snapshot|changed"):
            summarize(
                self.fixed_root, self.clip_root, self.selection_path,
                self.config_path, csv_path, json_path,
            )

    def test_report_rejects_rows_without_measured_bips2_or_with_proxy_marker(self):
        """A proxy or a missing measured score must not enter paired statistics."""
        from workflow.analysis.summarize_paired_sweep import summarize

        for label in ("missing", "proxy"):
            with self.subTest(label=label):
                csv_path, json_path = self._make_complete_fixture()
                key = read_json(self.selection_path)["points"][0]
                summary_path = self.fixed_root / key["workload"] / (
                    f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pipeline_summary.json"
                )
                summary = read_json(summary_path)
                if label == "missing":
                    summary.pop("bips2")
                else:
                    summary["bips2_proxy"] = True
                write_json(summary_path, summary)
                with self.assertRaisesRegex(ValueError, "measured|proxy"):
                    summarize(self.fixed_root, self.clip_root, self.selection_path,
                              self.config_path, csv_path, json_path)

    def test_report_records_an_absolute_bips2_difference_for_a_clip_loss(self):
        """A CLIP loss must not turn the absolute-difference column negative."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        key = read_json(self.selection_path)["points"][0]
        point = self.clip_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}"
        )
        clip_ipc2 = 1.6
        clip_frequency = read_json(point / "performance.json")[
            "sustainable_frequency_ghz"
        ]
        clip_bips2 = clip_ipc2 * clip_frequency
        for name in ("pipeline_summary.json", "performance.json"):
            payload = read_json(point / name)
            payload["ipc2"] = clip_ipc2
            payload["bips2"] = clip_bips2
            write_json(point / name, payload)
        r2_dir = point / "gem5_r2"
        r2_stats = r2_dir / "stats.txt"
        r2_stats.write_text(
            "---------- Begin Simulation Statistics ----------\n"
            + "".join(
                f"system.cpu{core}.commitStats0.numInsts 40\n"
                f"system.cpu{core}.numCycles 100\n"
                for core in range(4)
            ), encoding="utf-8",
        )
        result_path = r2_dir / "r2_result.json"
        result = read_json(result_path)
        result.update({
            "ipc2": clip_ipc2,
            "per_core": [
                {"core": core, "instructions": 40, "cycles": 100, "ipc": 0.4}
                for core in range(4)
            ],
            "stats_sha256": sha256_file(r2_stats),
        })
        write_json(result_path, result)
        r2_status = read_json(r2_dir / "status.json")
        r2_status.update({
            "ipc2": clip_ipc2,
            "stats_sha256": sha256_file(r2_stats),
            "r2_result_sha256": sha256_file(result_path),
        })
        write_json(r2_dir / "status.json", r2_status)
        status_path = self.status_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
        )
        status = read_json(status_path)
        status["clip3d_ipc2"] = clip_ipc2
        status["clip3d_bips2"] = clip_bips2
        write_json(status_path, status)

        result = summarize(self.fixed_root, self.clip_root, self.selection_path,
                           self.config_path, csv_path, json_path)

        fixed_bips2 = result["rows"][0]["fixed_bips2"]
        self.assertLess(clip_bips2, fixed_bips2)
        self.assertAlmostEqual(
            result["rows"][0]["absolute_bips2_difference"],
            fixed_bips2 - clip_bips2,
        )

    def test_report_rejects_a_local_r2_result_with_stale_vector_provenance(self):
        """A locally attached score must retain the Task-5 R1/vector identity."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        key = read_json(self.selection_path)["points"][0]
        status_path = self.fixed_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/gem5_r2/status.json"
        )
        status = read_json(status_path)
        status["latency_sha256"] = "0" * 64
        write_json(status_path, status)

        with self.assertRaisesRegex(ValueError, "local R2 provenance"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, json_path)

    def test_report_rejects_summary_architecture_that_disagrees_with_selected_key(self):
        """A valid R2 cache cannot relabel its selected workload or cache sizes."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        key = read_json(self.selection_path)["points"][0]
        summary_path = self.fixed_root / key["workload"] / (
            f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pipeline_summary.json"
        )
        summary = read_json(summary_path)
        summary["workload"] = "matmul"
        write_json(summary_path, summary)

        with self.assertRaisesRegex(ValueError, "architecture"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, json_path)

    def test_report_rejects_identical_csv_and_json_paths_before_publication(self):
        """A single path cannot safely hold two different report formats."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, _json_path = self._make_complete_fixture()
        csv_path.write_bytes(b"old-csv")
        with self.assertRaisesRegex(ValueError, "distinct"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, csv_path)
        self.assertEqual(csv_path.read_bytes(), b"old-csv")

    def test_report_rolls_back_both_outputs_when_second_publication_fails(self):
        """A failed JSON replacement must not leave a new CSV beside old JSON."""
        import workflow.analysis.summarize_paired_sweep as paired

        csv_path, json_path = self._make_complete_fixture()
        csv_path.write_bytes(b"old-csv")
        json_path.write_bytes(b"old-json")
        real_replace = paired._replace
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second publication failure")
            return real_replace(source, destination)

        with patch.object(paired, "_replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected second"):
                paired.summarize(self.fixed_root, self.clip_root,
                                 self.selection_path, self.config_path,
                                 csv_path, json_path)
        self.assertEqual(csv_path.read_bytes(), b"old-csv")
        self.assertEqual(json_path.read_bytes(), b"old-json")

    def test_report_normalizes_huge_metrics_and_rejects_zero_wire_cycles(self):
        """Overflowing metrics and zero-cycle vectors cannot become report rows."""
        from workflow.analysis.summarize_paired_sweep import summarize

        for label in ("huge", "zero-wire"):
            with self.subTest(label=label):
                csv_path, json_path = self._make_complete_fixture()
                key = read_json(self.selection_path)["points"][0]
                if label == "huge":
                    status_path = self.status_root / key["workload"] / (
                        f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
                    )
                    status = read_json(status_path)
                    old_limit = sys.get_int_max_str_digits()
                    try:
                        sys.set_int_max_str_digits(0)
                        status["fixed_ipc2"] = 10 ** 10000
                        write_json(status_path, status)
                    finally:
                        sys.set_int_max_str_digits(old_limit)
                else:
                    point = self.fixed_root / key["workload"] / (
                        f"l1d_{key['l1d_size']}/l2_{key['l2_size']}"
                    )
                    vector = read_json(point / "r2_latency.json")
                    vector["critical_l1d_to_l2_cycles"] = 0
                    write_json(point / "r2_latency.json", vector)
                    summary = read_json(point / "pipeline_summary.json")
                    summary["r2_critical_path_cycles"] = 0
                    write_json(point / "pipeline_summary.json", summary)
                    for name in ("r2_result.json", "status.json"):
                        r2_path = point / "gem5_r2" / name
                        r2 = read_json(r2_path)
                        r2["latency_sha256"] = sha256_file(point / "r2_latency.json")
                        write_json(r2_path, r2)
                    r2_status_path = point / "gem5_r2/status.json"
                    r2_status = read_json(r2_status_path)
                    r2_status["r2_result_sha256"] = sha256_file(
                        point / "gem5_r2/r2_result.json"
                    )
                    write_json(r2_status_path, r2_status)
                    pair_path = self.status_root / key["workload"] / (
                        f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
                    )
                    pair = read_json(pair_path)
                    pair["fixed_vector_sha256"] = sha256_file(point / "r2_latency.json")
                    write_json(pair_path, pair)
                with self.assertRaisesRegex(ValueError, "finite|wire|cannot read"):
                    summarize(self.fixed_root, self.clip_root, self.selection_path,
                              self.config_path, csv_path, json_path)

    def test_statistics_use_exact_win_tie_loss_and_even_median(self):
        """Paired statistics must retain signed changes and exact tie semantics."""
        from workflow.analysis.summarize_paired_sweep import _statistics

        rows = [
            {"ratio": 0.5, "percent_change": -50.0,
             "clip3d_reused_fixed_r2": False},
            {"ratio": 1.0, "percent_change": 0.0,
             "clip3d_reused_fixed_r2": True},
            {"ratio": 2.0, "percent_change": 100.0,
             "clip3d_reused_fixed_r2": True},
            {"ratio": 3.0, "percent_change": 200.0,
             "clip3d_reused_fixed_r2": False},
        ]

        result = _statistics(rows)

        self.assertAlmostEqual(result["arithmetic_mean_ratio"], 1.625)
        self.assertAlmostEqual(result["geometric_mean_ratio"], 3.0 ** 0.25)
        self.assertEqual(result["median_percent_change"], 50.0)
        self.assertEqual((result["wins"], result["ties"], result["losses"]),
                         (2, 1, 1))
        self.assertEqual(result["fixed_to_clip_reuse_count"], 2)
        self.assertEqual(result["separate_clip_r2_count"], 2)

    def test_report_accepts_committed_task5_reuse_and_rejects_marker_tampering(self):
        """A real reused row is accepted only while its Task-5 marker remains bound."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        summary, marker_path = self._attach_real_reuse_for_selected_pair()

        result = summarize(self.fixed_root, self.clip_root, self.selection_path,
                           self.config_path, csv_path, json_path)
        self.assertTrue(result["complete"])
        self.assertEqual(result["aggregate"]["fixed_to_clip_reuse_count"], 1)
        self.assertEqual(result["aggregate"]["separate_clip_r2_count"], 49)
        self.assertAlmostEqual(summary["ipc2"], 2.0)

        marker = read_json(marker_path)
        marker["source_ipc2"] = 99.0
        write_json(marker_path, marker)
        with self.assertRaisesRegex(ValueError, "reuse"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, json_path)

    def test_report_rejects_tampered_committed_reuse_output_hash(self):
        """A reuse marker's published-output hash must bind the live summary bytes."""
        from workflow.analysis.summarize_paired_sweep import summarize

        csv_path, json_path = self._make_complete_fixture()
        _summary, marker_path = self._attach_real_reuse_for_selected_pair()
        marker = read_json(marker_path)
        marker["target_outputs"]["summary"]["sha256"] = "0" * 64
        write_json(marker_path, marker)

        with self.assertRaisesRegex(ValueError, "reuse"):
            summarize(self.fixed_root, self.clip_root, self.selection_path,
                      self.config_path, csv_path, json_path)

    def test_report_rejects_duplicate_or_unbound_status_keys_and_mixed_config_hashes(self):
        """A status set that cannot be one selected/configured experiment is invalid."""
        from workflow.analysis.summarize_paired_sweep import summarize

        for label in ("duplicate", "config", "unbound"):
            with self.subTest(label=label):
                csv_path, json_path = self._make_complete_fixture()
                keys = read_json(self.selection_path)["points"]
                status_path = self.status_root / keys[-1]["workload"] / (
                    f"l1d_{keys[-1]['l1d_size']}/l2_{keys[-1]['l2_size']}/pair_status.json"
                )
                status = read_json(status_path)
                if label == "duplicate":
                    status["key"] = keys[0]
                elif label == "config":
                    status["config_sha256"] = "0" * 64
                else:
                    status["pair_status"] = str(self.root / "other-status.json")
                write_json(status_path, status)
                with self.assertRaisesRegex(ValueError, "duplicate|config|self-binding"):
                    summarize(self.fixed_root, self.clip_root, self.selection_path,
                              self.config_path, csv_path, json_path)

    def test_report_rejects_wrong_classification_and_unvalidated_reuse(self):
        """A non-canonical classification or markerless reuse cannot be reported."""
        from workflow.analysis.summarize_paired_sweep import summarize

        for label in ("classification", "reuse"):
            with self.subTest(label=label):
                csv_path, json_path = self._make_complete_fixture()
                key = read_json(self.selection_path)["points"][0]
                if label == "classification":
                    run_config_path = self.clip_root / key["workload"] / (
                        f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/run_config.json"
                    )
                    run_config = read_json(run_config_path)
                    run_config["config"]["experiment_classification"] = {
                        "non_formal": False
                    }
                    write_json(run_config_path, run_config)
                else:
                    status_path = self.status_root / key["workload"] / (
                        f"l1d_{key['l1d_size']}/l2_{key['l2_size']}/pair_status.json"
                    )
                    status = read_json(status_path)
                    status["clip3d_reused_fixed_r2"] = True
                    status["physical_r2_runs"] = 1
                    write_json(status_path, status)
                with self.assertRaisesRegex(ValueError, "classification|reuse"):
                    summarize(self.fixed_root, self.clip_root, self.selection_path,
                              self.config_path, csv_path, json_path)


if __name__ == "__main__":
    unittest.main()
