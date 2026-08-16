"""Behavior tests for the patched McPAT clean-build contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch"
BUILD = ROOT / "scripts/build_mcpat.sh"
MCPAT_SOURCE = ROOT / "tools/src/mcpat"
MARKER = "CLIP_MCPAT_CACTI_P_V1"


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
