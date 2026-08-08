from copy import deepcopy
from pathlib import Path
import unittest

from workflow.common import read_json
from workflow.run_lifting_pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "configs/experiments/"
    "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
    "traffic_weighted_exploratory.json"
)
CANDIDATE = ROOT / (
    "configs/experiments/"
    "clip3d_constrained_5p0_raw_power_p1_lambda0020119_"
    "traffic_weighted_discrete_partition_exploratory.json"
)
ADDED_PROVENANCE = {
    (
        "Integer-cycle partition search uses the same nearest-rounding "
        "function as gem5 R2."
    ),
    (
        "The fixed-bin layout is an explicit optimizer candidate; this "
        "corrects continuous-to-integer boundary regressions but does not "
        "validate the shared lambda_wire IPC model."
    ),
}


class DiscretePartitionConfigTests(unittest.TestCase):
    def test_candidate_is_an_isolated_nonformal_discrete_copy(self):
        source = read_json(SOURCE)
        candidate = read_json(CANDIDATE)

        self.assertEqual(
            source["layout_optimizer"]["wire_objective"], "continuous"
        )
        self.assertEqual(
            candidate["layout_optimizer"]["wire_objective"],
            "discrete-partition",
        )
        self.assertEqual(
            candidate["layout_optimizer"]["partition_grid_steps"], 41
        )
        self.assertIs(
            candidate["layout_optimizer"]["include_fixed_baseline"], True
        )
        self.assertEqual(
            candidate["experiment_classification"],
            {
                "mode": (
                    "operational-exploratory-traffic-weighted-"
                    "discrete-partition"
                ),
                "non_formal": True,
                "paper_equivalent": False,
                "shared_parameter_accepted": False,
            },
        )
        validate_config(candidate, "clip3d")

        source_normalized = deepcopy(source)
        candidate_normalized = deepcopy(candidate)
        source_normalized.pop("name")
        candidate_normalized.pop("name")
        source_normalized.pop("experiment_classification")
        candidate_normalized.pop("experiment_classification")
        source_optimizer = source_normalized["layout_optimizer"]
        candidate_optimizer = candidate_normalized["layout_optimizer"]
        source_optimizer.pop("wire_objective")
        candidate_optimizer.pop("wire_objective")
        candidate_optimizer.pop("partition_grid_steps")
        candidate_optimizer.pop("include_fixed_baseline")
        candidate_assumptions = candidate_normalized["provenance"][
            "reproduction_assumptions"
        ]
        self.assertTrue(ADDED_PROVENANCE.issubset(set(candidate_assumptions)))
        candidate_normalized["provenance"]["reproduction_assumptions"] = [
            statement for statement in candidate_assumptions
            if statement not in ADDED_PROVENANCE
        ]
        self.assertEqual(candidate_normalized, source_normalized)


if __name__ == "__main__":
    unittest.main()
