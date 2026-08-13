from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from workflow.cache_contract import (
    build_cache_contract,
    cache_access_cycles,
)
from workflow.cacti.characterize_cache import make_config
from workflow.mcpat.gem5_to_mcpat import convert, named_child, component_by_id


class CacheCycleTests(unittest.TestCase):
    def test_cache_cycles_use_ceiling_not_nearest(self):
        self.assertEqual(cache_access_cycles(1.01, 2.0), 3)
        self.assertEqual(cache_access_cycles(3.14, 2.0), 7)

    def test_exact_cycle_boundary_does_not_gain_a_cycle(self):
        self.assertEqual(cache_access_cycles(1.5, 2.0), 3)
        self.assertEqual(cache_access_cycles(1.5000000000000002, 2.0), 3)


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
                r1, xml_path, template=template, report_path=report_path
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


if __name__ == "__main__":
    unittest.main()
