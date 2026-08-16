"""Behavior tests for the patched McPAT clean-build contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from workflow.mcpat.cache_metrics import parse_embedded_cacti_records
from workflow.mcpat.gem5_to_mcpat import (
    component_by_id,
    convert,
    named_child,
)
from workflow.mcpat.run_mcpat import run_mcpat


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch"
BUILD = ROOT / "scripts/build_mcpat.sh"
MCPAT_SOURCE = ROOT / "tools/src/mcpat"
MARKER = "CLIP_MCPAT_CACTI_P_V1"


def native_text() -> str:
    def metrics(area: float, dynamic: float, indent: str = "  ") -> str:
        return (
            f"{indent}Area = {area} mm^2\n"
            f"{indent}Runtime Dynamic = {dynamic} W\n"
            f"{indent}Subthreshold Leakage = 0.1 W\n"
            f"{indent}Gate Leakage = 0.01 W\n"
        )

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
    sections = [
        "McPAT (version 1.3) results\n"
        "Technology 45 nm\nCore clock Rate(MHz) 2000\nProcessor:\n"
        + metrics(100.0, 10.0)
    ]
    separator = "*" * 40
    for _core in range(4):
        sections.append(
            "Core:\n" + metrics(20.0, 2.0)
            + "Instruction Fetch Unit:\n" + metrics(4.0, 0.4, "    ")
            + "Instruction Cache:\n" + metrics(1.0, 0.1, "      ")
            + "Renaming Unit:\n" + metrics(2.0, 0.2, "    ")
            + "Load Store Unit:\n" + metrics(4.0, 0.4, "    ")
            + "Data Cache:\n" + metrics(1.0, 0.1, "      ")
            + "Memory Management Unit:\n" + metrics(2.0, 0.2, "    ")
            + "Execution Unit:\n" + metrics(5.0, 0.5, "    ")
        )
    sections.append("L2\n" + metrics(8.0, 0.8))
    return (f"\n{separator}\n".join(sections) + "\n"
            + "\n".join(records) + "\n")


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


class McPATRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.r1 = self.root / "r1"
        self.out = self.root / "mcpat"
        self.r1.mkdir()
        metadata = {
            "num_cores": 4,
            "cpu_clock": "2GHz",
            "issue_width": 4,
            "rob_entries": 192,
            "cache_line_bytes": 64,
            "l1i_size": "16kB",
            "l1d_size": "32kB",
            "l2_size": "512kB",
            "l1_associativity": 2,
            "l2_associativity": 8,
        }
        (self.r1 / "r1_metadata.json").write_text(json.dumps(metadata))
        stats = []
        for core in range(4):
            stats.extend((
                f"system.cpu{core}.numCycles 1000\n",
                f"system.cpu{core}.commitStats0.numInsts 100\n",
                f"system.cpu{core}.commitStats0.numOps 100\n",
            ))
        (self.r1 / "stats.txt").write_text("".join(stats))
        self.binary = self.root / "bin" / "mcpat"
        self.binary.parent.mkdir()
        self.binary.write_text(MARKER + "\n")
        self.binary.chmod(0o755)

    def tearDown(self):
        self.temporary.cleanup()

    def test_runner_invokes_mcpat_once_and_writes_native_artifact(self):
        # Break caught: bypassing the one strict runner loses its canonical
        # native artifact or runs McPAT more than once.
        with patch(
            "workflow.mcpat.run_mcpat.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=native_text()),
        ) as run:
            result = run_mcpat(self.r1, self.out, {}, self.binary)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            result["embedded_cacti_p"]["authority"],
            "McPAT 1.3 embedded CACTI-P",
        )
        self.assertTrue((self.out / "input.xml").is_file())
        self.assertTrue((self.out / "mcpat.out").is_file())
        self.assertTrue((self.out / "mcpat.json").is_file())
        self.assertEqual(
            result["provenance"]["output_sha256"],
            hashlib.sha256((self.out / "mcpat.out").read_bytes()).hexdigest(),
        )

    def test_xml_cache_timings_are_only_optimization_constraints(self):
        # Break caught: accepting a standalone cache measurement and silently
        # presenting its timing as an XML-derived physical measurement.
        report = convert(self.r1, self.out / "input.xml")
        constraints = report["optimization_constraints"][
            "cache_latency_throughput_cycles"
        ]
        self.assertEqual(constraints["classification"], "not a measurement")
        self.assertEqual(constraints["throughput_cycles"], 10)
        self.assertEqual(constraints["latency_cycles"], 10)
        self.assertNotIn("cache_measurements", report)
        tree = ET.parse(self.out / "input.xml")
        system = component_by_id(tree.getroot(), "system")
        value = named_child(
            component_by_id(system, "system.core0.icache"),
            "param", "icache_config",
        ).get("value")
        self.assertEqual(value.split(",")[4:6], ["10", "10"])


if __name__ == "__main__":
    unittest.main()
