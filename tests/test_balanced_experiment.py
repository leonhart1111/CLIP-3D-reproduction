from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from workflow.common import read_json, write_json
from workflow.r1_catalog import (
    build_catalogue,
    canonical_directories,
    snapshot_canonical_artifacts,
)


class CanonicalFixtureTests(unittest.TestCase):
    def make_grid_fixture(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "r1"
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
            write_json(root / "planned_jobs.json", {"job_count": 4, "jobs": []})
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


if __name__ == "__main__":
    unittest.main()
