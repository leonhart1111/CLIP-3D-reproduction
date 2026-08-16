from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import csv
from pathlib import Path

from workflow.cache_contract import (
    build_cache_contract,
    cache_access_cycles,
    characterization_identity,
    mcpat_embedded_cache_record_identity,
    stable_identity,
    validate_characterization,
)
from workflow.cacti.characterize_cache import make_config, normalized_text
from workflow.mcpat.gem5_to_mcpat import convert, named_child, component_by_id
from workflow.common import PROJECT_ROOT
from workflow.run_lifting_pipeline import validate_config
from workflow.r2.build_latency_vector import build_vector
from workflow.common import write_json
from scripts.characterize_local_table_ii import write_reports


class CacheCycleTests(unittest.TestCase):
    def test_cache_cycles_use_ceiling_not_nearest(self):
        self.assertEqual(cache_access_cycles(1.01, 2.0), 3)
        self.assertEqual(cache_access_cycles(3.14, 2.0), 7)

    def test_exact_cycle_boundary_does_not_gain_a_cycle(self):
        self.assertEqual(cache_access_cycles(1.5, 2.0), 3)
        self.assertEqual(cache_access_cycles(1.5000000000000002, 2.0), 3)

    def test_cacti_evidence_normalizes_only_known_subnormal_garbage(self):
        text = (
            "\tTotal leakage power in cells (mW): 6.94253e-307  \n"
            "\tTotal leakage power in row logic(mW): 6.91243e-307\n"
            "\tTotal leakage power in column logic(mW): 6.90665e-307\n"
            "\tTotal leakage power in H-tree (mW): 1.25e-12\n"
        )
        self.assertEqual(normalized_text(text), (
            "\tTotal leakage power in cells (mW): 0\n"
            "\tTotal leakage power in row logic(mW): 0\n"
            "\tTotal leakage power in column logic(mW): 0\n"
            "\tTotal leakage power in H-tree (mW): 1.25e-12\n"
        ))


class CacheContractTests(unittest.TestCase):
    def metadata(self):
        return {
            "l1i_size": "16kB",
            "l1d_size": "32kB",
            "l2_size": "512kB",
            "l1_associativity": 2,
            "l2_associativity": 8,
            "cache_line_bytes": 64,
            "l1_cache_banks": 1,
            "l2_cache_banks": 1,
            "l1_cache_output_width_bits": 512,
            "l2_cache_output_width_bits": 512,
            "num_cores": 4,
        }

    def test_contract_has_distinct_gem5_derived_cache_records(self):
        contract = build_cache_contract(
            self.metadata(), technology_nm=45, temperature_k=320,
            device_type=0, interconnect_projection_type=1,
        )
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(
            contract["authority"],
            "gem5 R1 metadata for McPAT cache organization",
        )
        records = {record["level"]: record for record in contract["records"]}
        self.assertEqual(set(records), {"l1i", "l1d", "l2"})
        self.assertEqual(records["l1i"]["size_bytes"], 16 * 1024)
        self.assertEqual(records["l1d"]["size_bytes"], 32 * 1024)
        self.assertEqual(records["l2"]["associativity"], 8)
        self.assertEqual(records["l2"]["bank_count"], 1)
        self.assertEqual(records["l2"]["output_width_bits"], 512)
        self.assertEqual(records["l2"]["temperature_k"], 320)
        self.assertEqual(records["l2"]["device_type"], "itrs-hp")
        self.assertEqual(records["l2"]["interconnect_projection"], "conservative")

    def test_completed_legacy_r1_receives_documented_defaults(self):
        metadata = self.metadata()
        for key in (
            "l1_cache_banks", "l2_cache_banks",
            "l1_cache_output_width_bits", "l2_cache_output_width_bits",
        ):
            metadata.pop(key)
        contract = build_cache_contract(
            metadata, technology_nm=45, temperature_k=320,
            device_type=0, interconnect_projection_type=1,
        )
        self.assertEqual(contract["legacy_defaults_applied"], {
            "l1_cache_banks": 1,
            "l2_cache_banks": 1,
            "l1_cache_output_width_bits": 512,
            "l2_cache_output_width_bits": 512,
        })

    def test_generated_config_matches_contract(self):
        base = Path(
            "/home/zyjiang/Agenticflow/CLIP/tools/src/cacti/cache.cfg"
        ).read_text(encoding="utf-8")
        record = build_cache_contract(
            self.metadata(), technology_nm=45, temperature_k=320,
            device_type=0, interconnect_projection_type=1,
        )["records"][2]
        text = make_config(base, record)
        self.assertTrue(all(line == line.rstrip() for line in text.splitlines()))
        required = (
            "-size (bytes) 524288", "-block size (bytes) 64",
            "-associativity 8", "-UCA bank count 1",
            "-output/input bus width 512", "-technology (u) 0.045",
            "-operating temperature (K) 320",
            '-access mode (normal, sequential, fast) - "normal"',
            '-Add ECC - "true"', '-Interconnect projection - "conservative"',
            '-Cache level (L2/L3) - "L2"',
        )
        for directive in required:
            with self.subTest(directive=directive):
                self.assertIn(directive, text)


class McPATContractTests(unittest.TestCase):
    def test_mcpat_xml_encodes_the_same_structural_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r1 = root / "r1"
            r1.mkdir()
            metadata = CacheContractTests().metadata()
            metadata.update({"num_cores": 4, "cpu_clock": "2GHz"})
            (r1 / "r1_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            lines = []
            for core in range(4):
                lines.extend((
                    f"system.cpu{core}.numCycles 1000\n",
                    f"system.cpu{core}.commitStats0.numInsts 100\n",
                    f"system.cpu{core}.commitStats0.numOps 100\n",
                ))
            (r1 / "stats.txt").write_text("".join(lines), encoding="utf-8")
            xml_path = root / "input.xml"
            report_path = root / "mapping_report.json"

            template = Path(
                "/home/zyjiang/Agenticflow/CLIP/tools/src/mcpat/"
                "ProcessorDescriptionFiles/ARM_A9_2GHz.xml"
            )
            report = convert(
                r1, xml_path, template=template, report_path=report_path,
            )

            tree = ET.parse(xml_path)
            system = component_by_id(tree.getroot(), "system")
            expected = {
                "system.core0.icache": ("icache_config", "16384,64,2,1,10,10,512,0"),
                "system.core0.dcache": ("dcache_config", "32768,64,2,1,10,10,512,1"),
                "system.L20": ("L2_config", "524288,64,8,1,10,10,512,1"),
            }
            for identifier, (name, value) in expected.items():
                with self.subTest(identifier=identifier):
                    component = component_by_id(system, identifier)
                    self.assertEqual(named_child(component, "param", name).get("value"), value)
            self.assertEqual(report["cache_contract"]["records"][2]["bank_count"], 1)
            self.assertEqual(report["cache_contract"]["records"][2]["output_width_bits"], 512)
            self.assertEqual(
                report["optimization_constraints"][
                    "cache_latency_throughput_cycles"
                ]["classification"],
                "not a measurement",
            )
            self.assertNotIn("cacti_characterization_id", report)


class UnscaledConfigurationTests(unittest.TestCase):
    def test_official_configs_have_no_global_area_calibration(self):
        forbidden = {
            "area_reference_mm2", "area_reference_raw_mm2",
            "area_reference_basis",
        }
        for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.json")):
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(forbidden.isdisjoint(config.get("physical", {})))
                self.assertNotIn("150 mm^2 reference calibration", path.read_text())

    def test_pipeline_rejects_new_area_scaling_controls(self):
        config = {
            "schema_version": 1,
            "physical": {
                "r_convec_k_per_w": 5.0,
                "area_reference_mm2": 150.0,
            },
            "layout_optimizer": {"r_convec_k_per_w": 5.0},
            "mcpat": {}, "cacti": {}, "delay": {},
        }
        with self.assertRaisesRegex(ValueError, "area scaling"):
            validate_config(config, "fixed-bin")


class CharacterizationIdentityTests(unittest.TestCase):
    def contract(self):
        return build_cache_contract(
            CacheContractTests().metadata(), technology_nm=45,
            temperature_k=320, device_type=0,
            interconnect_projection_type=1,
        )

    def characterization(self):
        records = []
        for index, contract in enumerate(self.contract()["records"]):
            record = {
                **contract,
                "access_time_ns": 0.5 + index,
                "cycle_time_ns": 0.75 + index,
                "access_cycles": 1 + index * 2,
                "cycle_cycles": 2 + index * 2,
                "area_mm2": 0.2 + index,
                "width_mm": 0.5 + index,
                "height_mm": (0.2 + index) / (0.5 + index),
                "config_sha256": "a" * 64,
                "raw_output_sha256": "b" * 64,
            }
            record["cacti_record_id"] = stable_identity({
                key: value for key, value in record.items()
                if key not in ("config", "raw_output", "cacti_record_id")
            })
            records.append(record)
        result = {
            "schema_version": 2,
            "frequency_ghz": 2.0,
            "rounding": "ceiling, minimum one cycle; 1e-12 tolerance at exact integer boundaries",
            "records": records,
            "provenance": {"cacti_executable_sha256": "c" * 64},
        }
        result["characterization_id"] = characterization_identity(result)
        return result

    def native_model(self):
        metadata = {
            **CacheContractTests().metadata(), "cpu_clock": "2GHz",
        }
        timing = {
            "l1i": (0.5e-9, 0.75e-9),
            "l1d": (1.5e-9, 1.75e-9),
            "l2": (2.5e-9, 2.75e-9),
        }
        records = []
        for cache, core in (
            *(("l1i", core) for core in range(4)),
            *(("l1d", core) for core in range(4)),
            ("l2", None),
        ):
            access, cycle = timing[cache]
            record = {
                "cache": cache, "core": core,
                "access_time_s": access, "cycle_time_s": cycle,
                "height_mm": 0.5, "width_mm": 1.0,
                "mcpat_version": "1.3", "model": "embedded-cacti-p",
            }
            record["record_id"] = mcpat_embedded_cache_record_identity(record)
            records.append(record)
        return {
            "schema_version": 3,
            "architecture": metadata,
            "cache_contract": self.contract(),
            "cache_authority": "McPAT 1.3 embedded CACTI-P",
            "embedded_cacti_p": {
                "schema_version": 1,
                "authority": "McPAT 1.3 embedded CACTI-P",
                "records": records,
            },
            "mcpat_provenance": {
                "schema_version": 1,
                "authority": "CLIP strict patched McPAT 1.3 runner",
                "hashes": {
                    "xml_sha256": "1" * 64,
                    "mapping_sha256": "2" * 64,
                    "output_sha256": "3" * 64,
                    "binary_sha256": "4" * 64,
                    "patch_sha256": "5" * 64,
                },
            },
            "communication_profile": {"status": "unavailable"},
            "modules": [],
        }

    def test_valid_characterization_returns_distinct_level_records(self):
        selected = validate_characterization(
            self.characterization(), self.contract()
        )
        self.assertEqual(set(selected), {"l1i", "l1d", "l2"})

    def test_every_contract_field_mismatch_is_rejected(self):
        mutations = {
            "size_bytes": 12345,
            "associativity": 99,
            "bank_count": 2,
            "output_width_bits": 64,
            "technology_nm": 22,
            "temperature_k": 360,
            "ecc": False,
        }
        for field, value in mutations.items():
            characterization = self.characterization()
            characterization["records"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                validate_characterization(characterization, self.contract())

    def test_non_ceiling_and_duplicate_records_are_rejected(self):
        characterization = self.characterization()
        characterization["rounding"] = "nearest"
        with self.assertRaisesRegex(ValueError, "ceiling"):
            validate_characterization(characterization, self.contract())
        characterization = self.characterization()
        characterization["records"].append(dict(characterization["records"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_characterization(characterization, self.contract())

    def test_tampered_artifact_identity_is_rejected(self):
        characterization = self.characterization()
        characterization["characterization_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "characterization_id"):
            validate_characterization(characterization, self.contract())

    def test_characterization_identity_is_independent_of_evidence_paths(self):
        first = self.characterization()
        second = json.loads(json.dumps(first))
        for index, record in enumerate(second["records"]):
            record["config"] = f"/another/output/cache-{index}.cfg"
            record["raw_output"] = f"/another/output/cache-{index}.stdout.txt"
        second["provenance"].update({
            "cacti_executable": "/another/checkout/cacti",
            "base_config": "/another/checkout/cache.cfg",
        })
        self.assertEqual(
            characterization_identity(first), characterization_identity(second)
        )

    def test_r2_records_native_record_and_runner_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.native_model()
            modules = root / "modules.json"
            write_json(modules, model)

            vector = build_vector(
                modules, root / "latency.json",
                tsv_hops=1, wire_cycles=0,
            )

            provenance = vector["mcpat_cacti_p_provenance"]
            self.assertEqual(
                provenance["records"]["l2"]["record_ids"],
                [model["embedded_cacti_p"]["records"][-1]["record_id"]],
            )
            self.assertEqual(provenance["mcpat_output_sha256"], "3" * 64)
            self.assertEqual(provenance["mcpat_binary_sha256"], "4" * 64)
            self.assertNotIn("cacti_provenance", vector)

    def test_r2_rejects_native_record_identity_different_from_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.native_model()
            model["embedded_cacti_p"]["records"][-1]["access_time_s"] = 3e-9
            modules = root / "modules.json"
            write_json(modules, model)
            with self.assertRaisesRegex(ValueError, "record_id"):
                build_vector(
                    modules, root / "latency.json",
                    tsv_hops=1, wire_cycles=0,
                )


class LocalTableReportTests(unittest.TestCase):
    def test_report_writes_ordered_nine_row_provenance_table(self):
        records = []
        sizes = (
            (("l1d", 16), ("l1d", 32), ("l1d", 64), ("l1d", 128)),
            (("l2", 128), ("l2", 256), ("l2", 512),
             ("l2", 1024), ("l2", 2048)),
        )
        for level_sizes in sizes:
            for level, size_kib in level_sizes:
                records.append({
                    "level": level, "size": f"{size_kib}kB",
                    "size_bytes": size_kib * 1024,
                    "access_time_ns": 1.01,
                    "cycle_time_ns": 1.50,
                    "access_cycles_unrounded": 2.02,
                    "access_cycles": 3,
                    "cycle_cycles_unrounded": 3.0,
                    "cycle_cycles": 3,
                    "area_mm2": 0.5,
                    "width_mm": 1.0, "height_mm": 0.5,
                    "associativity": 2 if level == "l1d" else 8,
                    "bank_count": 1, "output_width_bits": 512,
                    "technology_nm": 45, "temperature_k": 320,
                    "config_sha256": "a" * 64,
                    "raw_output_sha256": "b" * 64,
                    "cacti_record_id": "c" * 64,
                })
        characterization = {
            "schema_version": 2, "frequency_ghz": 2.0,
            "rounding": "ceiling, minimum one cycle",
            "characterization_id": "d" * 64,
            "records": records,
            "provenance": {
                "cacti_git_revision": "e" * 40,
                "cacti_executable_sha256": "f" * 64,
                "base_config_sha256": "0" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, csv_path = write_reports(characterization, root)
            payload = json.loads(json_path.read_text())
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(payload["records"]), 9)
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["level"], "l1d")
        self.assertEqual(rows[-1]["size_bytes"], str(2048 * 1024))
        self.assertEqual(rows[0]["access_cycles"], "3")
        self.assertEqual(rows[0]["cacti_executable_sha256"], "f" * 64)


if __name__ == "__main__":
    unittest.main()
