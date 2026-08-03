from pathlib import Path
import unittest

from workflow.common import read_json
from workflow.run_lifting_pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json"
EXPLORATORY = (
    ROOT
    / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
)
REPORT = (
    ROOT
    / "results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json"
)


class LambdaWireExploratoryConfigTests(unittest.TestCase):
    def test_exact_measured_rejected_value_is_isolated_and_valid(self):
        self.assertTrue(EXPLORATORY.is_file(), f"missing exploratory config: {EXPLORATORY}")
        source = read_json(SOURCE)
        candidate = read_json(EXPLORATORY)
        report = read_json(REPORT)

        self.assertEqual(source["layout_optimizer"]["lambda_wire"], 0.0)
        self.assertEqual(
            candidate["layout_optimizer"]["lambda_wire"], report["lambda_wire"]
        )
        self.assertEqual(
            candidate["layout_optimizer"]["lambda_wire"], 0.0020119160767721133
        )
        self.assertEqual(
            candidate["layout_optimizer"]["wire_objective"], "continuous"
        )
        self.assertFalse(report["recommendation"]["accepted_for_this_workload"])
        self.assertFalse(
            report["recommendation"]["cross_workload_transfer_validated"]
        )
        self.assertEqual(
            candidate["experiment_classification"],
            {
                "mode": "operational-exploratory",
                "non_formal": True,
                "paper_equivalent": False,
                "shared_parameter_accepted": False,
            },
        )
        provenance = candidate["layout_optimizer"]["parameter_provenance"][
            "lambda_wire"
        ]
        self.assertEqual(provenance["source"], str(REPORT.relative_to(ROOT)))
        self.assertEqual(provenance["field"], "lambda_wire")
        self.assertEqual(provenance["value"], report["lambda_wire"])
        self.assertFalse(provenance["accepted_for_formal_or_shared_use"])
        self.assertEqual(provenance["purpose"], "optimizer feasibility only")

        source_without_identity = dict(source)
        candidate_without_identity = dict(candidate)
        source_without_identity.pop("name")
        candidate_without_identity.pop("name")
        candidate_without_identity.pop("experiment_classification")
        source_optimizer = dict(source_without_identity["layout_optimizer"])
        candidate_optimizer = dict(candidate_without_identity["layout_optimizer"])
        source_optimizer.pop("lambda_wire")
        candidate_optimizer.pop("lambda_wire")
        source_optimizer.pop("parameter_provenance")
        candidate_optimizer.pop("parameter_provenance")
        source_without_identity["layout_optimizer"] = source_optimizer
        candidate_without_identity["layout_optimizer"] = candidate_optimizer
        source_without_identity.pop("provenance")
        candidate_without_identity.pop("provenance")
        self.assertEqual(candidate_without_identity, source_without_identity)

        validate_config(candidate, "clip3d")


if __name__ == "__main__":
    unittest.main()
