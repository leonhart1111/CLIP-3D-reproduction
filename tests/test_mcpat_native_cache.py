"""Behavior tests for the patched McPAT clean-build contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from workflow.mcpat.cache_metrics import parse_embedded_cacti_records


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch"
BUILD = ROOT / "scripts/build_mcpat.sh"
MCPAT_SOURCE = ROOT / "tools/src/mcpat"
MARKER = "CLIP_MCPAT_CACTI_P_V1"


def native_text() -> str:
    records = []
    for cache, core in (
        *(("l1i", index) for index in range(4)),
        *(("l1d", index) for index in range(4)),
        ("l2", -1),
    ):
        records.append(
            f"{MARKER} cache={cache} core={core} "
            "access_time_s=1.25000000000000000e-09 "
            "cycle_time_s=2.50000000000000000e-09 "
            "height_mm=4.00000000000000000e-01 "
            "width_mm=8.00000000000000000e-01 "
            "mcpat_version=1.3 model=embedded-cacti-p"
        )
    return "\n".join(records) + "\n"


def duplicate_l1i() -> str:
    return native_text() + native_text().splitlines()[0] + "\n"


def nan_l2() -> str:
    return native_text().replace(
        "access_time_s=1.25000000000000000e-09",
        "access_time_s=nan",
        1,
    )


def missing_core3_l1d() -> str:
    return "\n".join(
        line for line in native_text().splitlines()
        if not ("cache=l1d" in line and "core=3" in line)
    ) + "\n"


def noncanonical_l2_core() -> str:
    return native_text().replace("cache=l2 core=-1", "cache=l2 core=-2")


class EmbeddedCACTIParserTests(unittest.TestCase):
    def test_accepts_exact_four_core_record_set(self):
        records = parse_embedded_cacti_records(native_text(), expected_core_count=4)
        self.assertEqual(len(records), 9)
        self.assertEqual(
            {(record["cache"], record["core"]) for record in records},
            {
                *(("l1i", index) for index in range(4)),
                *(("l1d", index) for index in range(4)),
                ("l2", None),
            },
        )
        self.assertEqual(
            records[0]["record_id"],
            "800ce674d18a6e977cda6ef8d65e7731968e8cf60c34be8dcd2acb0bd47061ca",
        )

    def test_rejects_duplicate_nonfinite_and_missing_records(self):
        for broken in (
            duplicate_l1i(), nan_l2(), missing_core3_l1d(), noncanonical_l2_core()
        ):
            with self.assertRaises(ValueError):
                parse_embedded_cacti_records(broken, expected_core_count=4)


class McPATPatchTests(unittest.TestCase):
    def test_patch_applies_to_vendor_source(self):
        # Production mutation caught: an incompatible or incomplete McPAT patch.
        self.assertTrue(PATCH.is_file(), "McPAT patch must exist")
        forward = subprocess.run(
            ["patch", "-d", str(MCPAT_SOURCE), "-p1", "--dry-run", "--forward"],
            input=PATCH.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        reverse = subprocess.run(
            ["patch", "-d", str(MCPAT_SOURCE), "-p1", "--dry-run", "--reverse"],
            input=PATCH.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertIn(
            0, (forward.returncode, reverse.returncode),
            forward.stdout.decode() + reverse.stdout.decode(),
        )

    def test_build_script_clean_builds_marks_and_records_provenance(self):
        # Production mutation caught: skipping clean/apply/marker/provenance behavior.
        self.assertTrue(BUILD.is_file(), "McPAT build script must exist")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "mcpat"
            source.mkdir()
            (source / "state").write_text("unpatched\n")
            (source / "Makefile").write_text(
                "all:\n"
                "\ttest \"$$(cat state)\" = patched\n"
                f"\tprintf '{MARKER}\\n' > mcpat\n"
                "\tchmod +x mcpat\n"
                "\tprintf 'all\\n' >> build.log\n"
                "clean:\n"
                "\trm -f mcpat\n"
                "\tprintf 'clean\\n' >> build.log\n"
            )
            fixture_patch = tmp_path / "fixture.patch"
            fixture_patch.write_text(
                "--- a/state\n"
                "+++ b/state\n"
                "@@ -1 +1 @@\n"
                "-unpatched\n"
                "+patched\n"
            )
            provenance = tmp_path / "build/build_provenance.json"
            env = {
                "MCPAT_SOURCE_DIR": str(source),
                "MCPAT_PATCH_FILE": str(fixture_patch),
                "MCPAT_PROVENANCE_FILE": str(provenance),
            }
            first = subprocess.run(
                [str(BUILD)], cwd=ROOT, env={**os.environ, **env},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertEqual((source / "build.log").read_text().splitlines(), ["clean", "all"])
            self.assertIn(MARKER, (source / "mcpat").read_text())
            recorded = json.loads(provenance.read_text())
            self.assertEqual(recorded["patch_sha256"], hashlib.sha256(fixture_patch.read_bytes()).hexdigest())
            self.assertEqual(recorded["binary_sha256"], hashlib.sha256((source / "mcpat").read_bytes()).hexdigest())
            self.assertEqual(
                recorded["build_commands"],
                ["make clean CXX=g++ CC=gcc", "make CXX=g++ CC=gcc"],
            )
            self.assertIn("built_at_utc", recorded)

            second = subprocess.run(
                [str(BUILD)], cwd=ROOT, env={**os.environ, **env},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(
                (source / "build.log").read_text().splitlines(),
                ["clean", "all", "clean", "all"],
            )

    def test_default_build_produces_marked_host_binary(self):
        # Production mutation caught: restoring upstream's unsupported -m32 compiler default.
        active = subprocess.run(["pgrep", "-x", "mcpat"], stdout=subprocess.DEVNULL)
        self.assertNotEqual(active.returncode, 0, "do not rebuild while McPAT is running")
        result = subprocess.run(
            [str(BUILD)], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        marker = subprocess.run(
            ["strings", str(MCPAT_SOURCE / "mcpat")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(marker.returncode, 0, marker.stdout)
        self.assertIn(MARKER, marker.stdout)


if __name__ == "__main__":
    unittest.main()
