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

from workflow.cache_contract import mcpat_embedded_cache_record_identity
from workflow.floorplan.build_module_model import (
    apply_mcpat_cache_geometry,
    build_model,
    validate_mobility_contract,
)
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


def embedded_records(dimensions=None) -> list[dict]:
    """Build real parser-validated records with independently chosen shapes."""
    dimensions = dimensions or {}
    lines = []
    for cache, core in (
        *(("l1i", index) for index in range(4)),
        *(("l1d", index) for index in range(4)),
        ("l2", -1),
    ):
        width, height = dimensions.get((cache, None if core == -1 else core), (4.0, 2.0))
        lines.append(
            f"{MARKER} cache={cache} core={core} "
            "access_time_s=1.25000000000000000e-09 "
            "cycle_time_s=2.50000000000000000e-09 "
            f"height_mm={height:.17e} width_mm={width:.17e} "
            "mcpat_version=1.3 model=embedded-cacti-p"
        )
    return parse_embedded_cacti_records("\n".join(lines), expected_core_count=4)


def cache_module(kind="l2", core=None, area=6.0) -> dict:
    result = {
        "name": "shared_l2" if kind == "l2" else f"core{core}_{kind}",
        "kind": kind,
        "area_mm2": area,
        "dynamic_power_w": 1.0,
        "leakage_power_w": 0.25,
        "total_power_w": 1.25,
    }
    if core is not None:
        result["core"] = core
    return result


def granular_34_modules() -> list[dict]:
    modules = []
    for core in range(4):
        for suffix, kind in (
            ("ifu", "core_ifu"), ("rename", "core_rename"),
            ("lsu", "core_lsu"), ("mmu", "core_mmu"),
            ("exec", "core_exec"), ("other", "core_other"),
            ("l1i", "l1i"), ("l1d", "l1d"),
        ):
            modules.append({
                "name": f"core{core}_{suffix}", "kind": kind, "core": core,
                "area_mm2": 1.0, "dynamic_power_w": 0.5,
                "subthreshold_leakage_w": 0.08, "gate_leakage_w": 0.02,
                "leakage_power_w": 0.1, "total_power_w": 0.6,
            })
    modules.extend((
        {
            "name": "shared_l2", "kind": "l2", "area_mm2": 4.0,
            "dynamic_power_w": 1.0, "subthreshold_leakage_w": 0.16,
            "gate_leakage_w": 0.04, "leakage_power_w": 0.2,
            "total_power_w": 1.2,
        },
        {
            "name": "noc", "kind": "interconnect", "area_mm2": 2.0,
            "dynamic_power_w": 0.5, "subthreshold_leakage_w": 0.08,
            "gate_leakage_w": 0.02, "leakage_power_w": 0.1,
            "total_power_w": 0.6,
        },
    ))
    return modules


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


class McPATGeometryTests(unittest.TestCase):
    def test_preserves_mcpat_area_and_embedded_aspect_ratio(self):
        # Break caught: replacing aggregate McPAT cache area with raw array area.
        result = apply_mcpat_cache_geometry(
            [cache_module(area=6.0)], embedded_records()
        )
        block = result[0]
        self.assertAlmostEqual(
            block["preferred_width_mm"] * block["preferred_height_mm"], 6.0
        )
        self.assertAlmostEqual(
            block["preferred_width_mm"] / block["preferred_height_mm"], 2.0
        )
        self.assertEqual(block["area_mm2"], 6.0)
        self.assertEqual(block["area_source"], "McPAT aggregate Area")
        self.assertEqual(
            block["raw_array_dimensions_mm"], {"width": 4.0, "height": 2.0}
        )
        self.assertEqual(
            block["normalized_block_dimensions_mm"],
            {
                "width": block["preferred_width_mm"],
                "height": block["preferred_height_mm"],
            },
        )
        self.assertEqual(block["aspect_ratio"], 2.0)
        self.assertEqual(
            block["geometry_formula"],
            "width=sqrt(McPAT_area_mm2*embedded_width_mm/embedded_height_mm); "
            "height=McPAT_area_mm2/width",
        )
        self.assertEqual(
            block["embedded_cacti_record_id"], embedded_records()[-1]["record_id"]
        )
        self.assertNotIn("source_cacti", block)
        self.assertNotIn("cacti_characterization_id", block)
        self.assertNotIn("mcpat_reported_area_mm2", block)

    def test_matches_l1_records_by_core_and_l2_to_the_shared_record(self):
        # Break caught: selecting one level-wide L1 shape for every core.
        records = embedded_records({
            ("l1i", 0): (3.0, 3.0),
            ("l1i", 1): (8.0, 2.0),
            ("l1d", 3): (9.0, 1.0),
            ("l2", None): (10.0, 2.0),
        })
        modules = [
            cache_module("l1i", 0), cache_module("l1i", 1),
            cache_module("l1d", 3), cache_module("l2"),
        ]
        result = apply_mcpat_cache_geometry(modules, records)
        by_name = {module["name"]: module for module in result}
        self.assertEqual(by_name["core0_l1i"]["aspect_ratio"], 1.0)
        self.assertEqual(by_name["core1_l1i"]["aspect_ratio"], 4.0)
        self.assertEqual(by_name["core3_l1d"]["aspect_ratio"], 9.0)
        self.assertEqual(by_name["shared_l2"]["aspect_ratio"], 5.0)

    def test_rejects_missing_duplicate_zero_and_divergent_records(self):
        valid = embedded_records()
        cases = []
        cases.append([record for record in valid if record["cache"] != "l2"])
        cases.append(valid + [dict(valid[-1])])
        zero = [dict(record) for record in valid]
        zero[-1]["width_mm"] = 0.0
        cases.append(zero)
        divergent = [dict(record) for record in valid]
        divergent[-1]["width_mm"] = 8.0
        cases.append(divergent)
        empty_identity = [dict(record) for record in valid]
        empty_identity[-1]["record_id"] = "0" * 64
        cases.append(empty_identity)
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                apply_mcpat_cache_geometry([cache_module()], records)

    def test_record_identity_rejects_nonphysical_or_noncanonical_values(self):
        # Break caught: a self-consistent forged ID must not legitimize an
        # invalid embedded timing, dimension, model, or cache/core identity.
        source = embedded_records()[-1]
        mutations = (
            ("access_time_s", 0.0), ("cycle_time_s", float("inf")),
            ("width_mm", 0.0), ("height_mm", -1.0),
            ("mcpat_version", "2.0"), ("model", "standalone-cacti"),
            ("core", 0),
        )
        for field, value in mutations:
            record = {**source, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                mcpat_embedded_cache_record_identity(record)

    def test_contract_has_one_movable_l2_and_fixed_granular_cores(self):
        # Break caught: exposing a core block or NoC as an optimizer variable.
        modules = granular_34_modules()
        contract = validate_mobility_contract(modules)
        self.assertEqual(contract["movable_names"], ["shared_l2"])
        self.assertEqual(len(contract["fixed_names"]), 33)
        self.assertEqual(contract["expected_core_count"], 4)
        self.assertEqual(contract["observed_core_indices"], [0, 1, 2, 3])
        self.assertTrue(next(m for m in modules if m["name"] == "shared_l2")["movable"])
        self.assertTrue(all(
            not module["movable"]
            for module in modules if module["name"] != "shared_l2"
        ))

    def test_contract_rejects_non_four_core_and_aggregate_fallback_models(self):
        missing_core = [
            module for module in granular_34_modules() if module.get("core") != 3
        ]
        aggregate = granular_34_modules()
        aggregate[0] = {
            **aggregate[0], "name": "core0_logic", "kind": "core_logic",
        }
        for modules in (missing_core, aggregate):
            with self.subTest(modules=modules), self.assertRaises(ValueError):
                validate_mobility_contract(modules)

    def test_contract_rejects_missing_functional_kind_and_multiple_l2s(self):
        missing_kind = [
            module for module in granular_34_modules()
            if module["name"] != "core2_mmu"
        ]
        multiple_l2s = granular_34_modules() + [{
            **granular_34_modules()[-2], "name": "shared_l2_duplicate",
        }]
        for modules in (missing_kind, multiple_l2s):
            with self.subTest(modules=modules), self.assertRaises(ValueError):
                validate_mobility_contract(modules)

    def write_native_model_inputs(self, root: Path, *, provenance=None,
                                  module_totals=None):
        r1 = root / "r1"
        r1.mkdir()
        metadata = {
            "num_cores": 4, "cpu_clock": "2GHz",
            "l1i_size": "16kB", "l1d_size": "32kB", "l2_size": "512kB",
            "l1_associativity": 2, "l2_associativity": 8,
            "cache_line_bytes": 64,
        }
        (r1 / "r1_metadata.json").write_text(json.dumps(metadata))
        (r1 / "stats.txt").write_text("".join(
            f"system.cpu{core}.commitStats0.numInsts 100\n"
            f"system.cpu{core}.numCycles 100\n"
            for core in range(4)
        ))
        modules = granular_34_modules()
        modules[-2].update({
            "source_cacti": "/legacy/cacti.json",
            "cacti_characterization_id": "a" * 64,
            "mcpat_reported_area_mm2": modules[-2]["area_mm2"],
        })
        calculated_totals = {
            field: sum(module[field] for module in modules)
            for field in (
                "area_mm2", "dynamic_power_w", "leakage_power_w", "total_power_w",
            )
        }
        payload = {
            "modules": modules,
            "module_totals": calculated_totals if module_totals is None else module_totals,
            "checks": {
                "core_count": 4,
                "core_logic_granularity": "McPAT top-level functional blocks",
            },
            "power_provenance": {"postprocessing": "none"},
            "cache_contract": {"schema_version": 1, "contract_id": "d" * 64},
            "embedded_cacti_p": {
                "schema_version": 1,
                "authority": "McPAT 1.3 embedded CACTI-P",
                "records": embedded_records(),
            },
            "provenance": provenance or {
                "xml_sha256": "1" * 64, "mapping_sha256": "2" * 64,
                "output_sha256": "3" * 64, "binary_sha256": "4" * 64,
                "patch_sha256": "5" * 64,
            },
            "source_cacti": "/legacy/cacti.json",
            "cacti_characterization_id": "a" * 64,
        }
        mcpat_path = root / "mcpat.json"
        mcpat_path.write_text(json.dumps(payload))
        return r1, mcpat_path, payload

    def test_build_model_serializes_native_authority_schema_and_conservation(self):
        # Break caught: reintroducing standalone CACTI area or losing strict
        # record/hash/mobility authority in modules.json.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1, mcpat_path, source = self.write_native_model_inputs(root)
            output = root / "modules.json"
            model = build_model(r1, mcpat_path, output)
        self.assertEqual(model["cache_authority"], "McPAT 1.3 embedded CACTI-P")
        self.assertEqual(model["embedded_cacti_p"], source["embedded_cacti_p"])
        self.assertEqual(model["mcpat_provenance"], source["provenance"])
        self.assertEqual(model["module_schema"]["core_count"], 4)
        self.assertEqual(model["module_schema"]["module_count"], 34)
        self.assertEqual(
            model["module_schema"]["core_logic_granularity"],
            "McPAT top-level functional blocks",
        )
        self.assertEqual(model["mobility_contract"]["movable_names"], ["shared_l2"])
        self.assertEqual(len(model["mobility_contract"]["fixed_names"]), 33)
        self.assertEqual(model["totals"], source["module_totals"])
        self.assertNotIn("source_cacti", model)
        self.assertNotIn("cacti_characterization_id", model)
        forbidden = {
            "source_cacti", "cacti_characterization_id",
            "mcpat_reported_area_mm2",
        }
        self.assertTrue(all(
            forbidden.isdisjoint(module) for module in model["modules"]
        ))

    def test_build_model_rejects_missing_hash_and_nonconserving_totals(self):
        complete = {
            "xml_sha256": "1" * 64, "mapping_sha256": "2" * 64,
            "output_sha256": "3" * 64, "binary_sha256": "4" * 64,
            "patch_sha256": "5" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_hash = dict(complete)
            missing_hash.pop("output_sha256")
            r1, mcpat_path, _ = self.write_native_model_inputs(
                root, provenance=missing_hash,
            )
            with self.assertRaisesRegex(ValueError, "hash|provenance"):
                build_model(r1, mcpat_path, root / "modules.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1, mcpat_path, _ = self.write_native_model_inputs(
                root,
                module_totals={
                    "area_mm2": 999.0, "dynamic_power_w": 17.5,
                    "leakage_power_w": 3.5, "total_power_w": 21.0,
                },
            )
            with self.assertRaisesRegex(ValueError, "conserv"):
                build_model(r1, mcpat_path, root / "modules.json")

    def test_build_model_rejects_missing_or_contradictory_strict_checks(self):
        for checks in (
            {},
            {"core_count": 3,
             "core_logic_granularity": "McPAT top-level functional blocks"},
            {"core_count": 4,
             "core_logic_granularity": "legacy aggregate core_logic fallback"},
        ):
            with self.subTest(checks=checks), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                r1, mcpat_path, payload = self.write_native_model_inputs(root)
                payload["checks"] = checks
                mcpat_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "strict|granular|core"):
                    build_model(r1, mcpat_path, root / "modules.json")


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
