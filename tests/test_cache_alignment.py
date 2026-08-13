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


if __name__ == "__main__":
    unittest.main()
