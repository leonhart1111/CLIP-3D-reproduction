import json
import tempfile
import unittest
from pathlib import Path

from workflow.accelerator.testbench import (
    build_plan,
    load_and_validate,
    materialize_selection,
    select_workloads,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/accelerators/gemmini_logicfolding_mvp.json"


class GemminiTestbenchTests(unittest.TestCase):
    def test_manifest_is_valid_and_has_explicit_transformer_status(self):
        manifest = load_and_validate(MANIFEST)
        workloads = {item["id"]: item for item in manifest["workloads"]}
        self.assertEqual(manifest["accelerator"]["mesh"], {"rows": 16, "cols": 16})
        self.assertEqual(workloads["transformer_mlp_tiny"]["status"], "planned")
        self.assertEqual(
            workloads["transformer_mlp_tiny"]["source"]["kind"],
            "custom-baremetal-gemm-chain",
        )

    def test_suite_selection_preserves_manifest_order(self):
        manifest = load_and_validate(MANIFEST)
        selected = select_workloads(manifest, "sensitivity_mvp")
        self.assertEqual(
            [item["id"] for item in selected],
            ["gemm_64", "gemm_256", "conv2d_8x8"],
        )

    def test_build_plan_does_not_claim_transformer_binary(self):
        manifest = load_and_validate(MANIFEST)
        plan = build_plan(manifest, "smoke", Path("/tmp/clip-external"))
        self.assertEqual([item["id"] for item in plan["workloads"]], ["gemm_64"])
        self.assertIn("revision-specific", plan["commands"][-1]["command"])

    def test_materialized_selection_contains_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selection.json"
            selection = materialize_selection(
                MANIFEST, "sensitivity_mvp", output, Path(directory) / "external"
            )
            self.assertTrue(output.is_file())
            self.assertEqual(selection["manifest_sha256"],
                             json.loads(output.read_text())["manifest_sha256"])
            self.assertEqual(selection["provenance"]["transformer_status"],
                             "planned_custom_gemm_chain")

    def test_selected_workload_requires_build_description(self):
        manifest = load_and_validate(MANIFEST)
        broken = json.loads(json.dumps(manifest))
        broken["workloads"][0].pop("build")
        with self.assertRaisesRegex(ValueError, "needs a build description"):
            validate_manifest(broken)


if __name__ == "__main__":
    unittest.main()
