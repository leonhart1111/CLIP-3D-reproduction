# Discrete Wire-Cycle Partition Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional deterministic L2 partition search whose optimizer wire penalty uses the exact integer cycle later injected into gem5 R2, while retaining fixed-bin as a candidate and preserving every existing mode.

**Architecture:** Keep the existing continuous L-BFGS-B path unchanged. Add a separate grid-based path for `wire_objective="discrete-partition"`: evaluate legal L2 positions with the closed-form thermal proxy, map continuous traffic-weighted delay through the shared rounding function, keep the best candidate per integer-cycle partition, and select globally across those representatives plus fixed-bin. The normal pipeline still performs only one final HotSpot solve and does not call gem5 during layout search.

**Tech Stack:** Python 3 standard library, existing SciPy-independent floorplan helpers, JSON experiment configuration, `unittest`, existing McPAT/CACTI/HotSpot pipeline for the final smoke test.

## Global Constraints

- Do not modify or rerun R1.
- Do not modify or delete `runs/operational_balanced50_traffic_weighted`; it is retained only as historical exploratory output and is no longer an acceptance dependency.
- Do not edit the existing traffic-weighted continuous configuration.
- Preserve `continuous` and `r2-quantized` behavior exactly.
- Preserve the configured `nearest` rounding rule for this correction.
- Continue using the current non-formal `lambda_wire=0.0020119160767721133`; IPC-sensitivity recalibration is outside this plan.
- Do not invoke HotSpot or gem5 inside optimizer candidate evaluation.
- The new experiment configuration must remain `non_formal=true` and `paper_equivalent=false`.
- Use `apply_patch` for source changes and stage only files named by the current task.

---

## File map

- Modify `workflow/floorplan/optimize_layout.py`: discrete score helper, deterministic grid, partition grouping, fixed-bin candidate, report fields.
- Modify `workflow/run_lifting_pipeline.py`: validate/pass the new mode and grid settings; expose the mode as a CLI override.
- Create `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json`: isolated corrected experiment entry.
- Modify `tests/test_workflow.py`: optimizer-unit, boundary, validation, determinism, and R2-cycle identity tests.
- Create `tests/test_discrete_partition_config.py`: provenance and isolation checks for the new configuration.
- Modify `docs/clip3d_pipeline_zh.md`: state when to use continuous versus discrete-partition mode and mark the abandoned 50-point run as historical exploratory evidence.

---

### Task 1: Implement the integer-cycle partition search in the floorplanner

**Files:**
- Modify: `workflow/floorplan/optimize_layout.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: `round_wire_cycles(value: float, policy: str) -> int`, `baseline_layout(model, utilization) -> dict`, `aggregate_wire_cycles(...) -> float`, and the existing `proxy_temperature`/`closed_form_frequency` functions.
- Produces: `discrete_wire_score(ipc1: float, frequency_ghz: float, lambda_wire: float, continuous_wire_cycles: float, rounding: str) -> tuple[float, int]` and two new optional `optimize` arguments: `partition_grid_steps: int = 41`, `include_fixed_baseline: bool = True`.

- [ ] **Step 1: Add a failing boundary-score test**

Add the import and test to `tests/test_workflow.py`:

```python
from workflow.floorplan.optimize_layout import (
    discrete_wire_score,
    optimize,
    proxy_temperature,
)

def test_discrete_wire_score_blocks_harmful_rounding_boundary(self):
    ipc1 = 4.31314420772608
    lambda_wire = 0.0020119160767721133
    fixed_score, fixed_cycle = discrete_wire_score(
        ipc1, 0.7894564656933903, lambda_wire, 1.3580618602253112,
        "nearest",
    )
    proposed_score, proposed_cycle = discrete_wire_score(
        ipc1, 0.7906404937722684, lambda_wire, 1.5577477284674477,
        "nearest",
    )
    self.assertEqual((fixed_cycle, proposed_cycle), (1, 2))
    self.assertLess(fixed_score, proposed_score)
```

- [ ] **Step 2: Run the boundary test and verify it fails**

Run:

```bash
python -m unittest tests.test_workflow.GridTests.test_discrete_wire_score_blocks_harmful_rounding_boundary -v
```

Expected: `ImportError` because `discrete_wire_score` does not exist.

- [ ] **Step 3: Implement the minimal integer-score helper**

Add near the other small helpers in `workflow/floorplan/optimize_layout.py`:

```python
def discrete_wire_score(ipc1: float, frequency_ghz: float,
                        lambda_wire: float, continuous_wire_cycles: float,
                        rounding: str) -> tuple[float, int]:
    rounded = round_wire_cycles(continuous_wire_cycles, rounding)
    score = -ipc1 * frequency_ghz + lambda_wire * ipc1 * rounded
    if not math.isfinite(score):
        raise ValueError("discrete partition score must be finite")
    return score, rounded
```

- [ ] **Step 4: Run the boundary test and verify it passes**

Run the command from Step 2.

Expected: one test passes and confirms that the known Cholesky `1.358 -> 1.558` move is rejected after integer mapping.

- [ ] **Step 5: Add failing option-validation and grid tests**

Add tests that call `optimize` with a temporary module model:

```python
def test_discrete_partition_rejects_invalid_grid_and_disabled_baseline(self):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_json(root / "modules.json", self.model())
        for steps in (2, 4):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "odd integer"):
                    optimize(
                        root / "modules.json", root / "layout.json",
                        root / "report.json", wire_objective="discrete-partition",
                        partition_grid_steps=steps,
                    )
        with self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
            optimize(
                root / "modules.json", root / "layout.json",
                root / "report.json", wire_objective="discrete-partition",
                partition_grid_steps=3, include_fixed_baseline=False,
            )
```

Add a deterministic integration test using `partition_grid_steps=5` and
`wire_aggregation="mean"`. Assert:

```python
self.assertEqual(report["parameters"]["wire_objective"], "discrete-partition")
self.assertEqual(report["discrete_search"]["grid_steps"], 5)
self.assertTrue(report["discrete_search"]["fixed_baseline_included"])
self.assertIn(report["selected"]["origin"], ("fixed-bin", "partition-grid"))
self.assertIsInstance(report["selected"]["r2_wire_cycles"], int)
self.assertEqual(
    report["selected"]["wire_objective_cycles"],
    report["selected"]["r2_wire_cycles"],
)
self.assertGreaterEqual(len(report["discrete_search"]["partitions"]), 1)
```

Run the same search twice into separate temporary paths and assert equality of
`selected` and `discrete_search.partitions`.

- [ ] **Step 6: Run the new tests and verify they fail**

Run:

```bash
python -m unittest \
  tests.test_workflow.GridTests.test_discrete_partition_rejects_invalid_grid_and_disabled_baseline \
  tests.test_workflow.GridTests.test_discrete_partition_is_deterministic_and_reports_integer_partitions \
  -v
```

Expected: failures because the new mode and arguments are not implemented.

- [ ] **Step 7: Add a separate discrete candidate evaluator**

Extend `optimize` without changing the positional order of existing arguments:

```python
def optimize(...,
             wire_aggregation: str = "mean",
             partition_grid_steps: int = 41,
             include_fixed_baseline: bool = True) -> dict:
```

Allow:

```python
if wire_objective not in ("continuous", "r2-quantized", "discrete-partition"):
    raise ValueError(
        "wire_objective must be continuous, r2-quantized, or discrete-partition"
    )
```

In discrete mode validate with exact type checks so booleans are rejected:

```python
if (not isinstance(partition_grid_steps, int)
        or isinstance(partition_grid_steps, bool)
        or partition_grid_steps < 3
        or partition_grid_steps % 2 == 0):
    raise ValueError("partition_grid_steps must be an odd integer >= 3")
if include_fixed_baseline is not True:
    raise ValueError("discrete-partition requires the fixed-bin baseline")
```

Add a nested or module-private evaluator that creates the L2 module, rejects
overlap, computes proxy temperature/frequency, derives continuous selected wire
cycles, calls `discrete_wire_score`, and returns a candidate with these stable
fields:

```python
{
    "origin": origin,
    "tier": tier,
    "x_mm": x,
    "y_mm": y,
    "loss": score,
    "collision_mm2": collision,
    "proxy_tmax_c": proxy,
    "proxy_frequency_ghz": frequency,
    "mean_wire_cycles": mean_wire,
    "wire_objective_cycles": rounded_wire,
    "r2_wire_cycles": rounded_wire,
    "continuous_selected_wire_cycles": selected_wire,
}
```

When aggregation is traffic-weighted, also retain the existing
`traffic_weighted_wire_cycles` continuous diagnostic.

- [ ] **Step 8: Generate the grid and choose partition representatives**

For each allowed tier, generate boundary-inclusive coordinates:

```python
xs = [upper[0] * index / (partition_grid_steps - 1)
      for index in range(partition_grid_steps)]
ys = [upper[1] * index / (partition_grid_steps - 1)
      for index in range(partition_grid_steps)]
```

Evaluate fixed-bin first, then grid candidates in deterministic
`tier -> y -> x` order. Exclude colliding grid candidates. Define one ordering
function:

```python
def key(candidate):
    return (
        candidate["loss"],
        candidate["continuous_selected_wire_cycles"],
        candidate["tier"], candidate["y_mm"], candidate["x_mm"],
    )
```

Keep `min(candidate, key=key)` per integer `r2_wire_cycles`, then choose `best`
from all partition representatives plus fixed-bin. Since fixed-bin is inserted
first, an exactly identical grid point keeps the fixed provenance.

Store only fixed-bin and the partition representatives in top-level
`candidates`; do not serialize all 1,681 grid evaluations. Add:

```python
"discrete_search": {
    "grid_steps": partition_grid_steps,
    "total_grid_candidates": total,
    "legal_grid_candidates": legal,
    "rejected_grid_candidates": rejected,
    "fixed_baseline_included": True,
    "fixed_baseline": fixed_candidate,
    "partitions": partition_records,
}
```

Use the selected candidate to emit the layout, then recompute through
`derive_layout_delays`. Raise `RuntimeError` if the recomputed selected rounded
cycle differs from `best["r2_wire_cycles"]`.

- [ ] **Step 9: Run the focused floorplanner tests**

Run:

```bash
python -m unittest \
  tests.test_workflow.GridTests.test_discrete_wire_score_blocks_harmful_rounding_boundary \
  tests.test_workflow.GridTests.test_discrete_partition_rejects_invalid_grid_and_disabled_baseline \
  tests.test_workflow.GridTests.test_discrete_partition_is_deterministic_and_reports_integer_partitions \
  tests.test_workflow.GridTests.test_optimizer_and_r2_use_the_same_traffic_weighted_aggregate \
  tests.test_workflow.GridTests.test_optimizer_reports_physical_observability \
  -v
```

Expected: all pass. The last two existing tests demonstrate that continuous
mode and R2 aggregation remain compatible.

- [ ] **Step 10: Commit the floorplanner task**

```bash
git add workflow/floorplan/optimize_layout.py tests/test_workflow.py
git commit -m "feat: search integer wire-cycle partitions"
```

---

### Task 2: Connect the new mode to pipeline validation and configuration

**Files:**
- Modify: `workflow/run_lifting_pipeline.py`
- Create: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json`
- Create: `tests/test_discrete_partition_config.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: Task 1's `optimize(..., partition_grid_steps=41, include_fixed_baseline=True)`.
- Produces: a validated `discrete-partition` experiment config and CLI override that feed the same rounding/aggregation values to optimizer and R2.

- [ ] **Step 1: Add failing pipeline-validation tests**

Extend `test_invalid_paper_mode_controls_are_rejected` or add focused methods:

```python
def test_discrete_partition_config_requires_valid_grid_and_baseline(self):
    config = {
        "schema_version": 1,
        "physical": {"r_convec_k_per_w": 5.0},
        "layout_optimizer": {
            "r_convec_k_per_w": 5.0,
            "wire_objective": "discrete-partition",
            "partition_grid_steps": 41,
            "include_fixed_baseline": True,
        },
    }
    validate_config(config, "clip3d")
    config["layout_optimizer"]["partition_grid_steps"] = 40
    with self.assertRaisesRegex(ValueError, "odd integer"):
        validate_config(config, "clip3d")
    config["layout_optimizer"]["partition_grid_steps"] = 41
    config["layout_optimizer"]["include_fixed_baseline"] = False
    with self.assertRaisesRegex(ValueError, "fixed-bin baseline"):
        validate_config(config, "clip3d")
```

- [ ] **Step 2: Run the validation test and verify it fails**

```bash
python -m unittest tests.test_workflow.FormalGuardTests.test_discrete_partition_config_requires_valid_grid_and_baseline -v
```

Expected: `validate_config` rejects `discrete-partition` as an invalid mode.

- [ ] **Step 3: Extend validation, pipeline forwarding, and CLI choice**

In `validate_config`, allow the new value and apply the exact Task 1 option
checks only for that mode. In the `optimize` call append:

```python
int(optimizer.get("partition_grid_steps", 41)),
optimizer.get("include_fixed_baseline", True),
```

Extend the CLI override choices:

```python
choices=("continuous", "r2-quantized", "discrete-partition")
```

Do not add separate grid CLI flags; the committed configuration is the
reproducibility authority.

- [ ] **Step 4: Run pipeline-validation tests**

```bash
python -m unittest \
  tests.test_workflow.FormalGuardTests.test_discrete_partition_config_requires_valid_grid_and_baseline \
  tests.test_workflow.FormalGuardTests.test_invalid_paper_mode_controls_are_rejected \
  tests.test_workflow.FormalGuardTests.test_mismatched_optimizer_and_hotspot_cooling_is_rejected \
  -v
```

Expected: all pass.

- [ ] **Step 5: Add the isolated exploratory configuration**

Copy the current traffic-weighted exploratory JSON to the new filename. Change
only these fields:

```json
"name": "constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory"
```

```json
"experiment_classification": {
  "mode": "operational-exploratory-traffic-weighted-discrete-partition",
  "non_formal": true,
  "paper_equivalent": false,
  "shared_parameter_accepted": false
}
```

```json
"layout_optimizer": {
  "wire_objective": "discrete-partition",
  "partition_grid_steps": 41,
  "include_fixed_baseline": true
}
```

Append these exact provenance statements:

```json
"Integer-cycle partition search uses the same nearest-rounding function as gem5 R2."
```

```json
"The fixed-bin layout is an explicit optimizer candidate; this corrects continuous-to-integer boundary regressions but does not validate the shared lambda_wire IPC model."
```

All original physical, thermal, power, traffic, and lambda values remain byte-for-byte equal.

- [ ] **Step 6: Add a configuration-isolation test**

Create `tests/test_discrete_partition_config.py`. Load source and candidate;
assert the exact new identity/classification/mode/grid/baseline fields,
`validate_config(candidate, "clip3d")`, and equality after removing only:

- `name`;
- `experiment_classification`;
- `layout_optimizer.wire_objective`;
- `layout_optimizer.partition_grid_steps`;
- `layout_optimizer.include_fixed_baseline`;
- the two appended provenance strings.

Also assert source remains `wire_objective == "continuous"` so the running
experiment definition was not silently changed.

- [ ] **Step 7: Run the configuration tests**

```bash
python -m unittest \
  tests.test_discrete_partition_config \
  tests.test_lambda_wire_exploratory_config \
  -v
```

Expected: all pass and the old exploratory configuration remains unchanged.

- [ ] **Step 8: Commit pipeline and configuration integration**

```bash
git add \
  workflow/run_lifting_pipeline.py \
  configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  tests/test_workflow.py \
  tests/test_discrete_partition_config.py
git commit -m "feat: expose discrete partition layout mode"
```

---

### Task 3: Verify the known real point and document the corrected entry

**Files:**
- Modify: `docs/clip3d_pipeline_zh.md`
- Runtime output only: `runs/discrete_partition_validation/cholesky_128kB_1024kB_20260808/`

**Interfaces:**
- Consumes: the Task 2 configuration and the completed canonical Cholesky R1 point.
- Produces: a lifting-only real HotSpot smoke result proving the selected vector no longer makes the known harmful `1 -> 2` transition, plus the documented command for later experiments.

- [ ] **Step 1: Run the complete fast automated test set**

```bash
python -m unittest \
  tests.test_workflow \
  tests.test_discrete_partition_config \
  tests.test_lambda_wire_exploratory_config \
  -v
```

Expected: all tests pass; tool-dependent skips are allowed only when already
declared by `unittest.skipUnless`.

- [ ] **Step 2: Run the known Cholesky lifting-only smoke**

Run from `/home/zyjiang/Agenticflow/CLIP`:

```bash
source scripts/env.sh
python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/paper/cholesky/l1d_128kB/l2_1024kB \
  --output-dir runs/discrete_partition_validation/cholesky_128kB_1024kB_20260808 \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  --layout-method clip3d \
  --transient false
```

Expected: pipeline completes without `--run-r2`; it runs the normal one final
HotSpot solve and writes `optimizer_report.json`, `r2_latency.json`, and
`pipeline_summary.json`.

- [ ] **Step 3: Verify integer-cycle identity and boundary regression**

```bash
jq '{origin: .selected.origin,
     continuous: .selected.continuous_selected_wire_cycles,
     integer: .selected.r2_wire_cycles,
     fixed_integer: .discrete_search.fixed_baseline.r2_wire_cycles,
     partitions: [.discrete_search.partitions[].r2_wire_cycles]}' \
  runs/discrete_partition_validation/cholesky_128kB_1024kB_20260808/optimizer_report.json
```

```bash
jq '{layout_wire: .components_cycles.layout_wire,
     critical: .critical_l1d_to_l2_cycles}' \
  runs/discrete_partition_validation/cholesky_128kB_1024kB_20260808/r2_latency.json
```

Expected:

- selected `integer` equals `r2_latency.components_cycles.layout_wire`;
- when fixed-bin has one wire cycle, the selected layout does not choose the
  previously observed thermally marginal two-cycle candidate;
- no gem5 R2 directory is created by this smoke.

- [ ] **Step 4: Document experiment status and commands**

Add a short section to `docs/clip3d_pipeline_zh.md` stating:

- `continuous` is retained for historical/paper-style comparison;
- `discrete-partition` is the corrected exploratory entry for new tests;
- the old Balanced-50 run is historical exploratory evidence and must not be
  combined with corrected-mode results;
- the correction prevents continuous-to-integer boundary mistakes but does
  not prove positive BIPS until the IPC model is separately validated;
- include the exact lifting-only command from Step 2.

- [ ] **Step 5: Run repository syntax and diff checks**

```bash
python -m compileall -q workflow tests
git diff --check
git status --short
```

Expected: compilation and diff checks succeed; status shows only the intended
documentation change plus any pre-existing user-owned files that were already
dirty before implementation.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/clip3d_pipeline_zh.md
git commit -m "docs: add discrete partition experiment entry"
```

- [ ] **Step 7: Record handoff facts without launching a new sweep**

Report:

- selected origin and integer wire cycle for the real Cholesky smoke;
- fixed versus selected proxy temperature/frequency/score;
- final HotSpot temperature and generated R2 critical path;
- automated test counts and skips;
- explicit statement that no R1 was rerun and no replacement 50-point sweep
  was launched.

Do not start R2 or a new multi-point sweep until the smoke evidence is reviewed.

---

## Completion definition

This correction is complete when all three task commits exist, automated tests
pass, and the real Cholesky lifting-only smoke demonstrates that optimizer and
R2 use the same integer wire cycle without selecting the known harmful rounding
boundary move. Positive BIPS2 across workloads is intentionally not part of
this completion definition.
