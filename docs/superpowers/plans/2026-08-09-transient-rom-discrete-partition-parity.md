# Transient ROM Discrete-Partition Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-formal transient-ROM experiment that uses the same nonzero λ, traffic-weighted nearest-rounded integer wire-cycle partitions, fixed baseline, and R2 mapping as the current steady 5-point experiment, then reports a real-HotSpot and real-gem5 fixed/CLIP paired comparison.

**Architecture:** Extract the steady optimizer's deterministic integer-cycle grid/partition mechanism into a shared floorplan module driven by a candidate evaluator callback. Preserve the existing continuous ROM optimizer for the λ=0 isolation profile, while the new configuration selects the shared discrete path and validates both fixed-bin and selected layouts with real transient HotSpot and branch-specific R2 vectors/results.

**Tech Stack:** Python 3.12, `unittest`, SciPy/NumPy ROM model, McPAT, CACTI, HotSpot, gem5, JSON/CSV evidence artifacts.

## Global Constraints

- Work only in `/home/zyjiang/Agenticflow/CLIP/.worktrees/transient-rom` on `feature/transient-rom`; do not modify the running `main` worktree.
- Do not modify or rerun canonical R1. Reuse the matching 2 ms periodic R1 when performing the real experiment.
- Keep the default steady entry point and `configs/experiments/clip3d_transient_rom_exploratory.json` behavior unchanged.
- The new profile is `non_formal=true`, `paper_equivalent=false`, and `shared_parameter_accepted=false`; formal promotion remains forbidden.
- Use `lambda_wire=0.0020119160767721133`, `wire_aggregation=traffic-weighted`, `wire_rounding=nearest`, `wire_objective=discrete-partition`, `partition_grid_steps=41`, `include_fixed_baseline=true`, and `allowed_l2_tiers=[1]` exactly.
- The ROM optimizer must make zero HotSpot calls. Calibration stays fixed at 8 training plus 2 holdout HotSpot jobs; final fixed/CLIP validation calls are counted separately.
- Final `BIPS2_trans` uses real gem5 `IPC2` multiplied by real-HotSpot sustainable frequency, never ROM-predicted frequency.
- Edit source files only with `apply_patch`; run formatting/test commands normally.
- Follow red-green-refactor for every behavior change and commit after each independently testable task.

---

## File Structure

### New files

- `workflow/floorplan/discrete_partition.py` — shared candidate geometry, deterministic grid enumeration, integer-cycle partitioning, fixed-baseline inclusion, and tie-breaking.
- `workflow/transient/rom/paired_validation.py` — one-branch final HotSpot/R2 validation and publication of paired fixed/CLIP metrics.
- `configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json` — exact nonzero-λ transient profile.
- `manifests/parameter_provenance/lambda_wire_fft_rejected.json` — tracked immutable copy of the measured-but-rejected λ report.
- `tests/test_transient_rom_discrete_parity.py` — focused configuration, shared search, cycle identity, and paired-publication tests.

### Modified files

- `workflow/floorplan/optimize_layout.py` — delegate discrete-partition enumeration/selection to the shared module without changing continuous and r2-quantized paths.
- `workflow/transient/rom/optimize_layout.py` — accept `discrete-partition`, evaluate the shared 41×41 grid with ROM frequency, and write partition evidence.
- `workflow/transient/rom/run_pipeline.py` — dispatch paired final validation only for the discrete ROM profile and preserve legacy single-layout behavior otherwise.
- `workflow/run_lifting_pipeline.py` — validate the new ROM profile's discrete controls through the existing configuration gate.
- `tests/test_workflow.py` — retain steady discrete regressions and assert common-entry validation/forwarding for the reviewed ROM discrete controls.
- `tests/test_transient_rom.py` — update existing fixtures only where the paired discrete branch adds required fields; continuous-profile tests retain their assertions.
- `tests/test_lambda_wire_exploratory_config.py` — read the tracked λ provenance artifact instead of ignored local results.
- `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json` — point provenance at the tracked artifact without changing the numeric profile.
- `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json` — point provenance at the tracked artifact without changing the current 5-point numeric profile.
- `docs/transient_rom_usage_zh.md` — document the new profile, paired evidence, and direct command.

---

### Task 1: Track λ evidence and add the parity configuration

**Files:**
- Create: `manifests/parameter_provenance/lambda_wire_fft_rejected.json`
- Create: `configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json`
- Create: `tests/test_transient_rom_discrete_parity.py`
- Modify: `tests/test_lambda_wire_exploratory_config.py:7-18`
- Modify: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json`
- Modify: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json`

**Interfaces:**
- Consumes: the local source report at `results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json` and the reviewed steady/ROM configurations.
- Produces: a tracked JSON report with `lambda_wire`, `recommendation.accepted_for_this_workload`, and `recommendation.cross_workload_transfer_validated`; a new configuration accepted by `validate_config(config, "clip3d")`.

- [ ] **Step 1: Write failing portability and parity tests**

Add constants and tests to `tests/test_transient_rom_discrete_parity.py`:

```python
from pathlib import Path
import unittest

from workflow.common import read_json
from workflow.run_lifting_pipeline import validate_config


ROOT = Path(__file__).resolve().parents[1]
STEADY = ROOT / "configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json"
ROM = ROOT / "configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json"
PROVENANCE = ROOT / "manifests/parameter_provenance/lambda_wire_fft_rejected.json"


class TransientROMParityConfigTests(unittest.TestCase):
    def test_tracked_lambda_evidence_preserves_rejection(self):
        report = read_json(PROVENANCE)
        self.assertEqual(report["lambda_wire"], 0.0020119160767721133)
        self.assertFalse(report["recommendation"]["accepted_for_this_workload"])
        self.assertFalse(report["recommendation"]["cross_workload_transfer_validated"])

    def test_rom_profile_matches_all_nonthermal_discrete_controls(self):
        steady, rom = read_json(STEADY), read_json(ROM)
        paths = (
            ("frequency", "f0_ghz"), ("frequency", "fmin_ghz"),
            ("frequency", "tsafe_c"), ("frequency", "ambient_c"),
            ("physical", "grid_size"), ("physical", "utilization"),
            ("physical", "r_convec_k_per_w"),
            ("layout_optimizer", "r_convec_k_per_w"),
            ("layout_optimizer", "alpha"), ("layout_optimizer", "beta"),
            ("layout_optimizer", "cross_tier_weight"),
            ("layout_optimizer", "lambda_wire"),
            ("layout_optimizer", "allowed_l2_tiers"),
            ("layout_optimizer", "wire_objective"),
            ("layout_optimizer", "partition_grid_steps"),
            ("layout_optimizer", "include_fixed_baseline"),
            ("delay", "wire_rounding"), ("delay", "wire_aggregation"),
        )
        for section, field in paths:
            self.assertEqual(rom[section][field], steady[section][field])
        self.assertTrue(rom["transient_rom"]["enabled"])
        self.assertFalse(rom["formal_validation"]["strict_p1"])
        self.assertFalse(rom["formal_validation"]["accepted"])
        validate_config(rom, "clip3d")
```

Change `REPORT` in `tests/test_lambda_wire_exploratory_config.py` to `PROVENANCE` and assert all nonzero-λ configuration provenance `source` fields equal its repository-relative path.

- [ ] **Step 2: Run tests and verify the intended failures**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity.TransientROMParityConfigTests \
  tests.test_lambda_wire_exploratory_config -v
```

Expected: FAIL because the tracked evidence and new ROM configuration do not exist, and the old test still depends on ignored `results/`.

- [ ] **Step 3: Add the tracked evidence and exact configuration**

Create the tracked evidence as an exact JSON copy of the existing report, preserving all rejection fields. Create the new ROM profile by taking the steady discrete profile's nonthermal sections and the existing ROM profile's complete `transient_rom` section. Set classification to:

```json
{
  "mode": "operational-exploratory-transient-rom-traffic-weighted-discrete-partition",
  "non_formal": true,
  "paper_equivalent": false,
  "shared_parameter_accepted": false
}
```

Set formal validation to:

```json
{
  "strict_p1": false,
  "accepted": false,
  "promotion": "forbidden: transient ROM profile-guided research extension"
}
```

Update the three nonzero-λ configuration provenance references to the tracked artifact while retaining the exact numeric λ and rejection text.

- [ ] **Step 4: Run focused configuration tests**

Run the Step 2 command.

Expected: PASS; `validate_config` accepts the non-formal profile, and no test reads ignored `results/`.

- [ ] **Step 5: Commit**

```bash
git add manifests/parameter_provenance/lambda_wire_fft_rejected.json \
  configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json \
  configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  tests/test_transient_rom_discrete_parity.py \
  tests/test_lambda_wire_exploratory_config.py
git commit -m "config: add transient ROM discrete parity profile"
```

---

### Task 2: Extract the shared integer-cycle partition engine

**Files:**
- Create: `workflow/floorplan/discrete_partition.py`
- Modify: `tests/test_transient_rom_discrete_parity.py`

**Interfaces:**
- Consumes: a base layout with exactly one named L2 and an evaluator callback `Callable[[dict, str, str], dict | None]`.
- Produces:
  - `place_l2(base_layout: dict, l2_name: str, tier: int, x_mm: float, y_mm: float) -> dict`
  - `validate_partition_controls(grid_steps: int, include_fixed_baseline: bool) -> None`
  - `search_discrete_partitions(base_layout: dict, l2_name: str, allowed_tiers: Sequence[int], grid_steps: int, include_fixed_baseline: bool, evaluate: Callable[[dict, str, str], dict | None]) -> dict`
- Candidate callback output requires `objective_loss`, `continuous_selected_wire_cycles`, and integer `r2_wire_cycles`; domain-specific fields pass through unchanged.

- [ ] **Step 1: Write failing shared-engine tests**

Add `DiscretePartitionEngineTests` with a two-module base layout. The evaluator derives coordinates from the returned layout and emits:

```python
{
    "objective_loss": abs(x_mm - 1.0) + cycle,
    "continuous_selected_wire_cycles": float(cycle) + x_mm / 100.0,
    "r2_wire_cycles": cycle,
    "x_mm": x_mm,
    "y_mm": y_mm,
    "tier": tier,
}
```

Assert:

```python
result = search_discrete_partitions(
    base, "L2", [1], 3, True, evaluator,
)
self.assertEqual(result["total_grid_candidates"], 9)
self.assertTrue(result["fixed_baseline_included"])
self.assertEqual(result["fixed_baseline"]["origin"], "fixed-bin")
self.assertEqual(
    [item["r2_wire_cycles"] for item in result["partitions"]],
    sorted({item["r2_wire_cycles"] for item in result["legal_grid_candidates_detail"]}),
)
self.assertEqual(result, search_discrete_partitions(base, "L2", [1], 3, True, evaluator))
```

Also assert invalid even/too-small grids and `include_fixed_baseline=False` raise `ValueError`, and an overlapping grid point is counted under `geometry_rejections` rather than sent to the evaluator.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity.DiscretePartitionEngineTests -v
```

Expected: ERROR with `ModuleNotFoundError: workflow.floorplan.discrete_partition`.

- [ ] **Step 3: Implement the minimal shared engine**

Implement these rules in `workflow/floorplan/discrete_partition.py`:

```python
def candidate_key(candidate: dict) -> tuple[float, float, int, float, float]:
    return (
        float(candidate["objective_loss"]),
        float(candidate["continuous_selected_wire_cycles"]),
        int(candidate["tier"]),
        float(candidate["y_mm"]),
        float(candidate["x_mm"]),
    )
```

`place_l2` must copy the layout/modules, replace exactly one L2, and call `check_geometry`. `search_discrete_partitions` must evaluate the explicit fixed position once, enumerate `grid_steps² × len(tiers)` points in y-major/x-minor order, group selectable candidates by integer `r2_wire_cycles`, retain each partition minimum by `candidate_key`, and select the minimum of fixed plus partitions. Preserve a detailed list of legal grid candidates and geometry/evaluator rejections for auditability.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command.

Expected: PASS with deterministic fixed and partition records.

- [ ] **Step 5: Commit**

```bash
git add workflow/floorplan/discrete_partition.py \
  tests/test_transient_rom_discrete_parity.py
git commit -m "feat: share integer wire-cycle partition search"
```

---

### Task 3: Refactor the steady discrete optimizer onto the shared engine

**Files:**
- Modify: `workflow/floorplan/optimize_layout.py:88-270`
- Modify: `tests/test_workflow.py:871-970`
- Modify: `tests/test_transient_rom_discrete_parity.py`

**Interfaces:**
- Consumes: `search_discrete_partitions` from Task 2.
- Produces: the existing steady `optimization_report.json` schema plus `shared_partition_engine=true`; existing `selected.loss`, `discrete_search.partitions`, fixed baseline, and R2 cycle checks remain available.

- [ ] **Step 1: Add a failing steady-delegation regression**

Patch `workflow.floorplan.optimize_layout.search_discrete_partitions` with `wraps` around the real helper in the existing deterministic optimizer test, then assert one call with `grid_steps=5` and `include_fixed_baseline=True`. Add a parity test that the shared helper's `selected.r2_wire_cycles` equals the optimizer report's selected integer cycle for a small synthetic model.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_workflow.GridTests.test_discrete_partition_is_deterministic_and_reports_integer_partitions \
  tests.test_transient_rom_discrete_parity -v
```

Expected: FAIL because `optimize_layout.py` does not import or call the shared engine.

- [ ] **Step 3: Delegate the steady discrete branch**

Retain the current steady candidate thermal and wire calculations, but normalize each callback result with:

```python
candidate.update({
    "objective_loss": candidate["loss"],
    "continuous_selected_wire_cycles": selected_wire,
    "r2_wire_cycles": rounded_wire,
})
```

Replace the local fixed/grid/partition loop with `search_discrete_partitions`. Map its result back into the existing `discrete_search` and `candidates` report fields so downstream steady tests and current experiment readers remain compatible. Set `shared_partition_engine=True` in the report parameters or search evidence. Keep the post-write assertion that standard `derive_layout_delays` produces the selected integer cycle.

- [ ] **Step 4: Run steady discrete and configuration regressions**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_workflow.GridTests \
  tests.test_workflow.FormalGuardTests \
  tests.test_discrete_partition_config \
  tests.test_transient_rom_discrete_parity -v
```

Expected: PASS; no existing steady report field or selected integer cycle changes.

- [ ] **Step 5: Commit**

```bash
git add workflow/floorplan/optimize_layout.py tests/test_workflow.py \
  tests/test_transient_rom_discrete_parity.py
git commit -m "refactor: share steady discrete partition engine"
```

---

### Task 4: Add discrete-partition search to the ROM optimizer

**Files:**
- Modify: `workflow/transient/rom/optimize_layout.py:179-455`
- Modify: `tests/test_transient_rom.py:1780-1980`
- Modify: `tests/test_transient_rom_discrete_parity.py`

**Interfaces:**
- Consumes: `search_discrete_partitions`, accepted ROM package evidence, `find_rom_sustainable_frequency`, `derive_layout_delays`, and canonical communication weights.
- Produces: for discrete mode, `optimization/partition_search.json`, a selected candidate containing `objective_loss`, `r2_wire_cycles`, `wire_objective_cycles`, `traffic_weighted_wire_cycles`, and `f_sus_trans_rom_ghz`; continuous/r2-quantized mode retains the existing lattice-plus-refinement report.

- [ ] **Step 1: Write failing ROM discrete tests**

Extend the existing ROM optimizer fixture with a discrete configuration using `partition_grid_steps=3`, `include_fixed_baseline=True`, traffic weights, and nonzero λ. Stub `find_rom_sustainable_frequency` to return a coordinate-dependent finite frequency. Assert:

```python
report = optimize_transient_layout(
    self.modules, self.accepted_package, self.output,
    self.config_path, self.power_windows, hotspot=self.hotspot,
)
self.assertEqual(report["parameters"]["wire_objective"], "discrete-partition")
self.assertEqual(report["parameters"]["lambda_wire"], 0.0020119160767721133)
self.assertEqual(report["parameters"]["wire_aggregation"], "traffic-weighted")
self.assertEqual(report["hotspot_calls_inside_optimizer"], 0)
self.assertTrue(report["search"]["fixed_baseline_included"])
self.assertEqual(report["selected"]["wire_objective_cycles"], report["selected"]["r2_wire_cycles"])
self.assertIsInstance(report["selected"]["r2_wire_cycles"], int)
self.assertTrue((output / "partition_search.json").is_file())
```

Add rejection tests for grid `40`, grid `2`, and `include_fixed_baseline=False` before any ROM frequency call.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom.ROMOptimizerTests \
  tests.test_transient_rom_discrete_parity -v
```

Expected: FAIL with `layout_optimizer.wire_objective is invalid` for `discrete-partition`.

- [ ] **Step 3: Implement the discrete ROM branch**

Accept the third wire objective and read exact controls:

```python
partition_grid_steps = optimizer.get("partition_grid_steps", 41)
include_fixed_baseline = optimizer.get("include_fixed_baseline", True)
```

For each shared-engine layout callback:

1. reject positions outside the accepted ROM interpolation domain;
2. call `find_rom_sustainable_frequency` once;
3. compute continuous traffic-weighted cycles from per-core delays;
4. compute integer cycles only through `round_wire_cycles`;
5. return `objective_loss = -ipc1 * f_rom + lambda_wire * ipc1 * integer_cycles`;
6. retain ROM frequency/temperature evidence and standard layout delays.

Discrete mode skips the old 25×25 plus five-refinement algorithm entirely. Continuous and r2-quantized modes continue to require `search_grid_points_per_axis=25` and `refinement_starts=5`; discrete mode instead requires the reviewed odd partition grid. Write the shared search evidence to both the main optimization report and `partition_search.json`.

- [ ] **Step 4: Verify optimizer tests**

Run the Step 2 command.

Expected: PASS; continuous-profile regression outputs remain unchanged and discrete mode reports zero optimizer HotSpot calls.

- [ ] **Step 5: Run static optimizer audit**

Run:

```bash
rg -n "subprocess\.run|run_hotspot|search_layout_frequency" \
  workflow/transient/rom/optimize_layout.py \
  workflow/floorplan/discrete_partition.py
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add workflow/transient/rom/optimize_layout.py \
  tests/test_transient_rom.py tests/test_transient_rom_discrete_parity.py
git commit -m "feat: optimize transient ROM in integer cycle partitions"
```

---

### Task 5: Enforce optimizer-to-R2 integer-cycle identity

**Files:**
- Create: `workflow/transient/rom/paired_validation.py`
- Modify: `tests/test_transient_rom_discrete_parity.py`

**Interfaces:**
- Consumes: a selected optimizer candidate, a built R2 vector, and `wire_aggregation`.
- Produces:
  - `require_selected_cycle_identity(selected: dict, vector: dict, wire_aggregation: str) -> int`
  - `branch_metrics(ipc2: float | None, f_hotspot_ghz: float | None) -> dict`
  - `publish_paired_comparison(fixed: dict, clip3d: dict, output_dir: Path, r2_requested: bool) -> dict | None`

- [ ] **Step 1: Write failing identity and publication tests**

Assert `require_selected_cycle_identity` returns `7` when:

```python
selected = {"r2_wire_cycles": 7}
vector = {
    "components_cycles": {"layout_wire": 7},
    "wire_cycle_aggregation_for_r2": "traffic-weighted",
    "layout_delays": {"traffic_weighted_wire_cycles": 7},
}
```

Assert it raises before R2 on a `7` versus `6` mismatch or wrong aggregation. Assert `branch_metrics(1.5, 1.8)` returns measured `bips2_trans=2.7`, while either `None` input leaves it `None`.

Assert `publish_paired_comparison` writes neither JSON nor CSV when R2 was not requested or one branch lacks measured BIPS. With two complete branches, assert:

```python
expected = (clip_bips - fixed_bips) / fixed_bips * 100.0
self.assertEqual(report["bips2_trans_improvement_percent"], expected)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity.PairedValidationUnitTests -v
```

Expected: ERROR because `workflow.transient.rom.paired_validation` does not exist.

- [ ] **Step 3: Implement fail-closed helpers**

`require_selected_cycle_identity` must read all three integer-cycle representations and require exact equality. `branch_metrics` must label values as `validated_f_sus_trans_hotspot_ghz`, `measured_ipc2`, and `measured_bips2_trans`, never generic `bips2`.

`publish_paired_comparison` must:

- validate branch labels `fixed-bin` and `clip3d`;
- reject non-finite/non-positive fixed BIPS;
- publish atomically via `write_json` and `csv.DictWriter` only after both branches are complete;
- include absolute difference and percentage improvement;
- include `non_formal=true`, `paper_equivalent=false`, λ, aggregation, rounding, and each branch artifact identity.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command.

Expected: PASS and no partial comparison files in negative cases.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/paired_validation.py \
  tests/test_transient_rom_discrete_parity.py
git commit -m "feat: gate transient ROM paired performance evidence"
```

---

### Task 6: Run paired fixed/CLIP real-HotSpot validation and R2

**Files:**
- Modify: `workflow/transient/rom/run_pipeline.py:520-927`
- Modify: `tests/test_transient_rom.py:2760-3655`
- Modify: `tests/test_transient_rom_discrete_parity.py`

**Interfaces:**
- Consumes: discrete optimizer report, steady preflight fixed layout, shared power windows, accepted ROM package, canonical frequency grid, CACTI characterization, and Task 5 helpers.
- Produces for discrete mode:
  - `final_validation/fixed_bin/branch_summary.json`
  - `final_validation/clip3d/branch_summary.json`
  - per-branch `r2_latency.json` and optional `gem5_r2/r2_result.json`
  - complete-only `paired_comparison.json` and `paired_comparison.csv`
  - top-level `transient_rom_summary.json` containing `branches`, predicted/validated/measured namespaces, and no ambiguous `bips2` key.

- [ ] **Step 1: Write failing paired-pipeline tests**

Using the existing `ROMPipelineTests` fixture, set optimization parameters to `wire_objective="discrete-partition"` and provide a fixed layout under `steady_preflight/hotspot/layout.json`. Patch final search with two valid side effects and assert calls target:

```text
final_validation/fixed_bin
final_validation/clip3d
```

Patch `build_vector` to return branch-specific layout-wire cycles and assert it receives the fixed and proposed layout paths separately. With `execute_r2=False`, assert no R2 calls and no paired comparison files. With `execute_r2=True`, return fixed `ipc2=1.4` and CLIP `ipc2=1.5`; assert two R2 calls, two measured branch BIPS values, and the exact paired percentage.

Add a mismatch test where the CLIP vector cycle differs from `selected.r2_wire_cycles`; assert both R2 and comparison publication are skipped and an auditable failure summary is written.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity.TransientROMPairedPipelineTests -v
```

Expected: FAIL because the current pipeline validates only the selected layout and writes one R2 vector.

- [ ] **Step 3: Factor one-branch final validation inside the pipeline**

Add the following private helper in `run_pipeline.py` and implement the branch sequence described immediately below it:

```python
def _run_final_branch(
    *, branch: str, layout_path: Path, selected: dict | None,
    modules_path: Path, cacti_path: Path, power_windows_path: Path,
    output_dir: Path, config_path: Path, config: dict, settings: ROMSettings,
    frequency_grid: list[float], hotspot: Path, source_r1_dir: Path,
    execute_r2: bool, rerun_r2: bool,
) -> dict:
    """Validate one layout with real HotSpot, bind its R2 vector, and optionally run gem5."""
```

The helper must run and validate real HotSpot, count its evaluations, build the branch vector, enforce selected-cycle identity for CLIP, run R2 only after all gates pass, calculate namespaced metrics, and write `branch_summary.json`. The fixed branch still records its vector cycle and fixed optimizer candidate identity even though it does not enforce the selected-CLIP assertion.

- [ ] **Step 4: Dispatch only discrete mode to paired validation**

For `wire_objective == "discrete-partition"`:

1. resolve and verify the fixed layout recorded by the steady preflight;
2. verify it matches the optimizer's fixed-baseline layout/coordinates;
3. run fixed branch and CLIP branch into separate directories;
4. publish paired comparison only after both real R2 results exist;
5. write a summary with `branches.fixed_bin` and `branches.clip3d`.

For `continuous` and `r2-quantized`, keep the current single `final_hotspot_validation`, `r2_latency.json`, and summary behavior so all λ=0 ROM tests remain valid.

- [ ] **Step 5: Verify paired and legacy pipeline tests**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity \
  tests.test_transient_rom.ROMPipelineTests \
  tests.test_transient_rom.ROMOptimizerTests -v
```

Expected: PASS; paired discrete mode and legacy single-layout mode coexist.

- [ ] **Step 6: Commit**

```bash
git add workflow/transient/rom/run_pipeline.py \
  tests/test_transient_rom.py tests/test_transient_rom_discrete_parity.py
git commit -m "feat: validate transient ROM fixed and CLIP branches"
```

---

### Task 7: Document the runnable profile and verify the complete repository

**Files:**
- Modify: `docs/transient_rom_usage_zh.md`
- Modify: `tests/test_transient_rom_discrete_parity.py`
- Modify: `tests/test_workflow.py` — add a common CLI/config gate regression for the reviewed discrete-ROM controls.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a direct no-R1-rerun command, artifact interpretation, measured-versus-predicted warnings, and fresh full-suite verification evidence.

- [ ] **Step 1: Write the failing documentation contract test**

Assert the usage document contains all of:

```text
clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
--transient-rom-r1-dir
--transient-rom-calibrate
paired_comparison.json
lambda_wire=0.0020119160767721133
wire_objective=discrete-partition
predicted
validated
measured
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity.TransientROMParityDocumentationTests -v
```

Expected: FAIL because the new command and paired artifact are undocumented.

- [ ] **Step 3: Add the direct command and evidence explanation**

Document this shape, using a new unique output directory:

```bash
cd /home/zyjiang/Agenticflow/CLIP/.worktrees/transient-rom
source scripts/env.sh

R1=/home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
PERIODIC_R1=/home/zyjiang/Agenticflow/CLIP/runs/transient_validation/matmul_32kB_512kB_lambda0020119_2ms_precision6_20260806_030743/transient/shared_r1
CFG=configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
OUT=runs/transient_rom/matmul_32kB_512kB_lambda0020119_discrete_$(date +%Y%m%d_%H%M%S)

python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir "$PERIODIC_R1" \
  --transient-rom-calibrate \
  --run-r2
```

State explicitly that the first scientific comparison uses the paired real-HotSpot and real-R2 report, while ROM-predicted values explain only the selection.

- [ ] **Step 4: Run focused suites**

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom_discrete_parity \
  tests.test_transient_rom \
  tests.test_transient \
  tests.test_workflow \
  tests.test_discrete_partition_config \
  tests.test_lambda_wire_exploratory_config -v
```

Expected: PASS with no errors.

- [ ] **Step 5: Run the full discovery suite**

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest discover -s tests -v
```

Expected: PASS; the previous ignored-results `FileNotFoundError` is gone. Record total tests and skips from this fresh run.

- [ ] **Step 6: Run static and repository checks**

```bash
git diff --check
rg -n "subprocess\.run|run_hotspot|search_layout_frequency" \
  workflow/transient/rom/optimize_layout.py \
  workflow/floorplan/discrete_partition.py
git status --short --branch
```

Expected: no whitespace errors; no optimizer HotSpot/subprocess matches; only intended documentation/test changes remain before commit.

- [ ] **Step 7: Commit**

```bash
git add docs/transient_rom_usage_zh.md \
  tests/test_transient_rom_discrete_parity.py tests/test_workflow.py
git commit -m "docs: explain transient ROM paired discrete experiment"
```

- [ ] **Step 8: Request code review and finish the branch**

Use `superpowers:requesting-code-review`, address findings with fresh failing tests where behavior changes, rerun Task 7 Steps 4–6, then use `superpowers:finishing-a-development-branch`. Do not merge into `main` while the 5-point experiment is still launching work from that tree.
