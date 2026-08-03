# Exploratory Lambda-Wire MATMUL Single-Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable non-formal `lambda_wire = 0.0020119160767721133` configuration and run one fresh MATMUL 32kB/512kB CLIP-3D point through HotSpot and gem5 R2 without rerunning R1.

**Architecture:** Keep the accepted operational configuration and all existing results immutable.  A new JSON profile changes only experiment identity, wire coefficient, and classification/provenance; the existing lifting pipeline consumes it.  A final Markdown/JSON evidence report compares the fresh point with the matched fixed-bin and zero-lambda pilots and separates continuous wire effects from integer R2 latency effects.

**Tech Stack:** JSON experiment profiles, Python 3 standard-library `unittest`, existing CLIP-3D Python workflow, CACTI, McPAT, HotSpot, gem5 R2, SHA-256 provenance checks.

## Global Constraints

- Reuse `runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB`; never rerun or modify it.
- Do not modify `configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json`.
- Use exactly `lambda_wire = 0.0020119160767721133` and retain `wire_objective = "continuous"`.
- Label the new profile operational, exploratory, non-formal, rejected for shared/formal promotion, and `paper_equivalent = false`.
- Do not overwrite `runs/operational_raw_power_p1/pilot_direct_20260731/{fixed-bin,clip3d}`.
- Write the fresh point below `runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d`.
- Execute a fresh gem5 R2 even when its integer latency vector matches an existing vector.
- Preserve the FFT report's rejection and do not claim that this one-point result validates a transferable parameter.

---

## File structure

- Create `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json`: isolated experiment input containing the exact measured wire coefficient and explicit limitations.
- Create `tests/test_lambda_wire_exploratory_config.py`: focused regression test proving source-config immutability, exact parameter/provenance, and normal pipeline validity.
- Create `results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_comparison.json`: machine-readable three-way comparison after the run succeeds.
- Create `results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_report.md`: human-readable interpretation of optimizer, HotSpot, latency, IPC, and BIPS evidence.
- Do not modify workflow implementation files; the experiment exercises their current behavior.

### Task 1: Add the isolated exploratory configuration

**Files:**
- Create: `tests/test_lambda_wire_exploratory_config.py`
- Create: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json`
- Reference: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json`
- Reference: `results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json`

**Interfaces:**
- Consumes: `workflow.common.read_json(path: Path) -> dict` and `workflow.run_lifting_pipeline.validate_config(config: dict, layout_method: str) -> None`.
- Produces: a schema-version-1 pipeline profile accepted for `layout_method="clip3d"`, with exact experimental classification and provenance.

- [ ] **Step 1: Write the failing configuration regression test**

```python
from pathlib import Path
import unittest

from workflow.common import read_json
from workflow.run_lifting_pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json"
EXPLORATORY = ROOT / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json"
REPORT = ROOT / "results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json"


class LambdaWireExploratoryConfigTests(unittest.TestCase):
    def test_exact_measured_rejected_value_is_isolated_and_valid(self):
        source = read_json(SOURCE)
        candidate = read_json(EXPLORATORY)
        report = read_json(REPORT)

        self.assertEqual(source["layout_optimizer"]["lambda_wire"], 0.0)
        self.assertEqual(candidate["layout_optimizer"]["lambda_wire"], report["lambda_wire"])
        self.assertEqual(candidate["layout_optimizer"]["lambda_wire"], 0.0020119160767721133)
        self.assertEqual(candidate["layout_optimizer"]["wire_objective"], "continuous")
        self.assertFalse(report["recommendation"]["accepted_for_this_workload"])
        self.assertFalse(report["recommendation"]["cross_workload_transfer_validated"])
        self.assertEqual(candidate["experiment_classification"], {
            "mode": "operational-exploratory",
            "non_formal": True,
            "paper_equivalent": False,
            "shared_parameter_accepted": False,
        })
        provenance = candidate["layout_optimizer"]["parameter_provenance"]["lambda_wire"]
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
```

- [ ] **Step 2: Run the test and verify the missing config fails**

Run:

```bash
python -m unittest tests.test_lambda_wire_exploratory_config -v
```

Expected: `ERROR` with `FileNotFoundError` for the exploratory JSON.

- [ ] **Step 3: Create the minimal exploratory profile**

Copy every physical, frequency, McPAT, CACTI, non-wire optimizer, comparison,
and delay field verbatim from the operational profile.  Apply only these
semantic edits:

```json
{
  "name": "constrained_5p0_raw_power_p1_lambda0020119_exploratory",
  "experiment_classification": {
    "mode": "operational-exploratory",
    "non_formal": true,
    "paper_equivalent": false,
    "shared_parameter_accepted": false
  },
  "layout_optimizer": {
    "lambda_wire": 0.0020119160767721133,
    "wire_objective": "continuous",
    "parameter_provenance": {
      "lambda_wire": {
        "source": "results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json",
        "field": "lambda_wire",
        "value": 0.0020119160767721133,
        "accepted_for_formal_or_shared_use": false,
        "purpose": "optimizer feasibility only",
        "rejection": "R^2 below 0.95, one monotonicity violation, and no cross-workload transfer validation"
      }
    }
  }
}
```

Retain the source alpha, beta, and cross-tier provenance entries.  Replace only
the source reproduction statement about zero lambda with statements describing
the exact FFT-local source, rejection, and feasibility-only use.

- [ ] **Step 4: Run the focused and existing validation tests**

Run:

```bash
python -m unittest tests.test_lambda_wire_exploratory_config -v
python -m unittest tests.test_workflow.WorkflowTests.test_invalid_paper_mode_controls_are_rejected -v
```

Expected: both commands pass.

- [ ] **Step 5: Verify the source profile did not change**

Run:

```bash
git diff --exit-code -- configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json
git diff --check -- configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json tests/test_lambda_wire_exploratory_config.py
```

Expected: both commands exit zero with no output.

- [ ] **Step 6: Commit the config and focused test**

```bash
git add configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json tests/test_lambda_wire_exploratory_config.py
git commit -m "exp: add exploratory lambda-wire profile"
```

### Task 2: Preflight and run the full MATMUL point

**Files:**
- Read-only: `runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/r1_metadata.json`
- Read-only: `runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/stats.txt`
- Read-only controls: `runs/operational_raw_power_p1/pilot_direct_20260731/{fixed-bin,clip3d}`
- Generate: `runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d/**`
- Generate: `/tmp/clip3d_lambda0020119_matmul_r1_before.sha256`
- Generate: `/tmp/clip3d_lambda0020119_matmul_r1_after.sha256`

**Interfaces:**
- Consumes: the canonical R1 and exploratory schema-version-1 config.
- Produces: `pipeline_summary.json`, `optimizer_report.json`, HotSpot artifacts, `r2_latency.json`, and a fresh successful `gem5_r2/r2_result.json`.

- [ ] **Step 1: Verify the canonical R1 and both controls are complete and matched**

Run:

```bash
python - <<'PY'
from pathlib import Path
from workflow.common import read_json

root = Path("/home/zyjiang/Agenticflow/CLIP")
r1 = root / "runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB"
fixed = root / "runs/operational_raw_power_p1/pilot_direct_20260731/fixed-bin"
zero = root / "runs/operational_raw_power_p1/pilot_direct_20260731/clip3d"
source_config = read_json(
    root / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json"
)
for path in (r1 / "r1_metadata.json", r1 / "stats.txt",
             fixed / "pipeline_summary.json", fixed / "gem5_r2/r2_result.json",
             fixed / "gem5_r2/status.json", fixed / "run_config.json",
             zero / "pipeline_summary.json", zero / "optimizer_report.json",
             zero / "gem5_r2/r2_result.json", zero / "gem5_r2/status.json",
             zero / "run_config.json"):
    assert path.is_file(), path
metadata = read_json(r1 / "r1_metadata.json")
assert (metadata["workload"], metadata["l1d_size"], metadata["l2_size"]) == (
    "matmul", "32kB", "512kB")
for point in (fixed, zero):
    summary = read_json(point / "pipeline_summary.json")
    assert (summary["workload"], summary["l1d_size"], summary["l2_size"]) == (
        "matmul", "32kB", "512kB")
    assert Path(summary["r1"]).resolve() == r1.resolve()
    assert summary["cooling"] == {"r_convec_k_per_w": 5.0, "ambient_c": 25.0}
    assert read_json(point / "gem5_r2/status.json")["state"] == "success"
    assert read_json(point / "run_config.json")["config"] == source_config
assert read_json(zero / "optimizer_report.json")["parameters"]["lambda_wire"] == 0.0
print("preflight matched: canonical R1, fixed-bin, zero-lambda clip3d")
PY
```

Expected: the final printed line and exit zero.

- [ ] **Step 2: Hash the read-only R1 scientific inputs before execution**

Run:

```bash
sha256sum \
  runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/r1_metadata.json \
  runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/stats.txt \
  > /tmp/clip3d_lambda0020119_matmul_r1_before.sha256
```

Expected: exit zero and a two-line manifest.

- [ ] **Step 3: Ensure the new output path is unused**

Run:

```bash
test ! -e runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d
```

Expected: exit zero.  If it exists, stop and inspect it; do not delete or overwrite it automatically.

- [ ] **Step 4: Execute the fresh end-to-end point**

Run:

```bash
source scripts/env.sh
python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json \
  --layout-method clip3d \
  --run-r2
```

Expected after approximately 3.2 hours: `Pipeline complete: method=clip3d, ...`.
The command must not use `--rerun-r2` because the output directory is new.

- [ ] **Step 5: Verify completion and exact parameter consumption**

Run:

```bash
python - <<'PY'
from pathlib import Path
from workflow.common import read_json

point = Path("runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d")
summary = read_json(point / "pipeline_summary.json")
run_config = read_json(point / "run_config.json")["config"]
optimizer = read_json(point / "optimizer_report.json")
status = read_json(point / "gem5_r2/status.json")
assert run_config["layout_optimizer"]["lambda_wire"] == 0.0020119160767721133
assert optimizer["parameters"]["lambda_wire"] == 0.0020119160767721133
assert optimizer["parameters"]["wire_objective"] == "continuous"
assert status["state"] == "success"
assert summary["ipc2"] is not None and summary["bips2"] is not None
print(summary["tmax_c"], summary["sustainable_frequency_ghz"],
      summary["ipc2"], summary["bips2"])
PY
```

Expected: exit zero and four numeric result values.

- [ ] **Step 6: Prove R1 remained unchanged**

Run:

```bash
sha256sum \
  runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/r1_metadata.json \
  runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/stats.txt \
  > /tmp/clip3d_lambda0020119_matmul_r1_after.sha256
diff -u \
  /tmp/clip3d_lambda0020119_matmul_r1_before.sha256 \
  /tmp/clip3d_lambda0020119_matmul_r1_after.sha256
```

Expected: `diff` exits zero with no output.

### Task 3: Produce and verify the three-way evidence report

**Files:**
- Create: `results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_comparison.json`
- Create: `results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_report.md`
- Read: all three points' `pipeline_summary.json`, `r2_latency.json`, `hotspot/layout.json`, and available `optimizer_report.json`.

**Interfaces:**
- Consumes: completed fixed-bin, zero-lambda CLIP-3D, and candidate CLIP-3D point artifacts.
- Produces: a machine-readable comparison and a concise scientific interpretation with explicit non-formal limits.

- [ ] **Step 1: Extract the exact comparison values without rounding**

Run:

```bash
python - <<'PY'
from pathlib import Path
from workflow.common import read_json

root = Path("runs/operational_raw_power_p1")
points = {
    "fixed_bin": root / "pilot_direct_20260731/fixed-bin",
    "lambda_zero": root / "pilot_direct_20260731/clip3d",
    "lambda_0020119": root / "lambda0020119_matmul_32kB_512kB_20260803/clip3d",
}
for label, path in points.items():
    summary = read_json(path / "pipeline_summary.json")
    latency = read_json(path / "r2_latency.json")
    row = {
        "label": label,
        "tmax_c": summary["tmax_c"],
        "f_sus_ghz": summary["sustainable_frequency_ghz"],
        "ipc2": summary["ipc2"],
        "bips2": summary["bips2"],
        "wire_unrounded": summary["layout_delays"]["wire_cycles_unrounded"],
        "wire_rounded": summary["layout_delays"]["wire_cycles"],
        "critical_cycles": summary["r2_critical_path_cycles"],
        "gem5_overrides": latency["gem5_overrides"],
    }
    if (path / "optimizer_report.json").is_file():
        optimizer = read_json(path / "optimizer_report.json")
        row["lambda_wire"] = optimizer["parameters"]["lambda_wire"]
        row["selected"] = optimizer["selected"]
    print(row)
PY
```

Expected: three dictionaries, including candidate lambda
`0.0020119160767721133` and non-null R2 values.

- [ ] **Step 2: Write the machine-readable comparison**

Create the JSON with:

- `schema_version: 1`;
- `classification` equal to the exploratory profile classification;
- absolute paths and SHA-256 hashes of each source summary, latency, layout, and optimizer report;
- the exact fields printed in Step 1;
- candidate-minus-zero and candidate-minus-fixed absolute and percentage deltas;
- booleans for selected-layout change, continuous-wire change, rounded-wire change, complete-latency-vector change, and positive/negative BIPS2 change;
- limitations copied from the FFT lambda report and this design.

- [ ] **Step 3: Write the human-readable report**

The report must state:

1. why lambda zero disabled only the wire term in candidate scoring;
2. whether `0.0020119160767721133` changed the selected L2 placement and candidate ranking;
3. whether continuous changes survived integer cycle rounding;
4. the actual HotSpot, frequency, IPC2, and BIPS2 deltas;
5. that a one-point favorable or unfavorable result cannot validate cross-workload transfer;
6. whether the experiment demonstrated optimizer feasibility and what next matched-R2 experiments are still required.

- [ ] **Step 4: Verify artifact consistency and report classification**

Run:

```bash
python -m json.tool results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_comparison.json >/dev/null
rg -n "operational-exploratory|non-formal|paper-equivalent|0.0020119160767721133|R.2|monotonic|cross-workload" \
  results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_report.md
git diff --check -- \
  results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_comparison.json \
  results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_report.md
```

Expected: JSON validation succeeds, every required limitation is found, and
`git diff --check` exits zero.

- [ ] **Step 5: Run final scoped verification**

Run:

```bash
python -m unittest tests.test_lambda_wire_exploratory_config -v
python -m unittest tests.test_workflow -v
```

Expected: all tests pass.  Any pre-existing unrelated failure must be reported
with its exact test name and is not to be hidden or changed as part of this
experiment.

- [ ] **Step 6: Commit only the experiment definition and tracked evidence**

```bash
git add \
  configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json \
  tests/test_lambda_wire_exploratory_config.py \
  results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_comparison.json \
  results/parameter_studies/lambda_wire_exploratory_20260803/matmul_32kB_512kB_report.md
git commit -m "exp: record exploratory lambda-wire MATMUL point"
```

If the repository ignores `runs/` or generated result evidence, leave those
artifacts uncommitted and report their absolute paths rather than using
`git add -f`.
