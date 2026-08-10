# Transient ROM Balanced-2 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, resumable two-point entry point that runs transient-ROM thermal validation before optional R2 and compares the result with the completed steady MATMUL/STENCIL baseline.

**Architecture:** A tracked selection names only the two architecture keys.  A thin experiment module derives and validates canonical/periodic R1 paths, delegates each point to the existing public lifting CLI, and aggregates only validated artifacts.  Scientific algorithms remain in the existing transient-ROM pipeline.

**Tech Stack:** Python 3.12, `unittest`, JSON/CSV, SHA-256 provenance, existing gem5/HotSpot transient-ROM CLI.

## Global Constraints

- Work only in the existing isolated `fix/transient-rom-steady-init` worktree.
- Do not modify canonical R1, HotSpot source, McPAT mapping, ROM equations, layout search, R2 implementation, or steady baseline artifacts.
- Select exactly MATMUL 64 kB/512 kB and STENCIL 64 kB/512 kB in that order.
- Require 2 ms cumulative periodic R1 bound to the exact canonical source R1.
- Use `clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json` unchanged.
- Keep `pss_tolerance_c = 0.01`, 12-call package calibration, zero optimizer HotSpot calls, and paired fixed-bin/CLIP-3D final validation.
- Never populate measured transient IPC/BIPS fields from ROM predictions.
- Do not start any R2 unless both thermal points validate.
- Never overwrite failed/non-reusable output; require a new output root.
- Keep all results `non_formal=true` and `paper_equivalent=false`.

---

### Task 1: Selection and Input Preflight

**Files:**
- Create: `configs/experiments/transient_rom_balanced2_selection.json`
- Create: `workflow/experiments/transient_rom_balanced2.py`
- Create: `tests/test_transient_rom_balanced2.py`

**Interfaces:**
- Produces: `load_selection(path: Path) -> list[dict]`.
- Produces: `preflight_inputs(selection_path: Path, canonical_r1_root: Path, periodic_r1_root: Path, steady_baseline_csv: Path, config_path: Path) -> dict`.
- Each returned point contains `key`, canonical/periodic paths, validated steady row, and immutable input hashes.

- [ ] **Step 1: Write failing selection/preflight tests**

Create fixtures for two successful canonical R1 directories, two successful 2 ms periodic R1 directories, and a five-row-like steady CSV.  Assert exact point order and derived paths.  Add one test per rejected condition: changed point order, third point, baseline state not `success`, periodic sample other than 2 ms, periodic source mismatch, workload/cache metadata mismatch, and config without exploratory classification.

```python
inputs = preflight_inputs(selection, canonical, periodic, baseline, config)
self.assertEqual([p["key"] for p in inputs["points"]], [
    "matmul/l1d_64kB/l2_512kB",
    "stencil/l1d_64kB/l2_512kB",
])
self.assertEqual(inputs["sample_interval_ms"], 2.0)
self.assertFalse(inputs["paper_equivalent"])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2PreflightTests -v
```

Expected: import failure because the module and functions do not exist.

- [ ] **Step 3: Add the exact tracked selection**

Write schema version 1 with `non_formal=true`, `paper_equivalent=false`,
`sample_interval_ms=2.0`, and exactly:

```json
[
  {"workload": "matmul", "l1d_size": "64kB", "l2_size": "512kB"},
  {"workload": "stencil", "l1d_size": "64kB", "l2_size": "512kB"}
]
```

- [ ] **Step 4: Implement strict preflight**

Derive canonical paths as
`ROOT/workload/l1d_SIZE/l2_SIZE` and periodic paths as
`ROOT/workload_SIZE_SIZE`.  Require canonical `status.json`,
`r1_metadata.json`, and `stats.txt`; call existing
`run_transient_r1.completed(periodic_path, 2.0, canonical_path)`; require the
steady CSV to contain one and only one matching `state=success` row; validate
the scientific config through the existing CLIP-3D config validator; record
SHA-256 for selection, config, baseline, and each R1 metadata/stats/status
input.  Reject booleans as numeric sampling values and reject path overlap
between read-only inputs.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2PreflightTests -v
```

Expected: all preflight tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/transient_rom_balanced2_selection.json \
  workflow/experiments/transient_rom_balanced2.py \
  tests/test_transient_rom_balanced2.py
git commit -m "feat: validate transient ROM Balanced-2 inputs"
```

### Task 2: Thermal-First Phase Orchestration

**Files:**
- Modify: `workflow/experiments/transient_rom_balanced2.py`
- Modify: `tests/test_transient_rom_balanced2.py`

**Interfaces:**
- Produces: `validate_thermal_checkpoint(point_output: Path, point: dict) -> dict`.
- Produces: `validate_r2_checkpoint(point_output: Path, point: dict) -> dict`.
- Produces: `run_validation_set(..., execute_r2: bool = False, invoke: Callable | None = None) -> dict`.
- Thermal outputs live under `OUTPUT/thermal/<workload>_64kB_512kB` and R2 outputs under `OUTPUT/r2/<workload>_64kB_512kB`.

- [ ] **Step 1: Write failing thermal orchestration tests**

Use an injected invocation function that records argv and materializes valid
synthetic pipeline summaries.  Assert both thermal commands contain
`--transient-rom-calibrate`, omit `--run-r2`, use the matching periodic R1,
and run in selection order.  Assert a valid existing thermal checkpoint is
reused only after complete branch validation.  Assert an existing malformed
checkpoint is rejected rather than overwritten.

```python
result = run_validation_set(..., execute_r2=False, invoke=fake_invoke)
self.assertEqual(result["state"], "thermal_validated")
self.assertEqual(len(calls), 2)
self.assertTrue(all("--transient-rom-calibrate" in c for c in calls))
self.assertTrue(all("--run-r2" not in c for c in calls))
```

- [ ] **Step 2: Run the thermal tests and verify RED**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2RunnerTests.test_thermal_phase_validates_both_points_before_r2 -v
```

Expected: missing orchestration functions.

- [ ] **Step 3: Implement thermal checkpoint validation and invocation**

Require `transient_rom/transient_rom_summary.json` to identify transient-ROM,
exploratory classification, the selected source architecture, an accepted
12-call package, and both `branches.fixed_bin` and `branches.clip3d` with finite
positive `validated_f_sus_trans_hotspot_ghz`, no failure, and null measured
BIPS when R2 was not requested.  Invoke the public CLI with `sys.executable -m
workflow.run_lifting_pipeline`, write stdout/stderr logs, and atomically update
point/root status JSON before and after each call.

- [ ] **Step 4: Write failing R2 gate tests**

Assert `execute_r2=True` refuses to invoke anything if either thermal
checkpoint is missing or invalid.  With both thermal checkpoints valid, assert
two fresh commands use the corresponding package, include `--run-r2`, omit
`--transient-rom-calibrate`, and publish no root success until both R2
checkpoints validate.

- [ ] **Step 5: Run the R2 tests and verify RED**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2RunnerTests.test_r2_is_gated_by_both_thermal_points -v
```

Expected: failure because the R2 phase is not implemented.

- [ ] **Step 6: Implement R2 phase and strict checkpoint validation**

Use each thermal package at
`thermal/POINT/transient_rom/rom_package` in a fresh R2 root.  Require
`final_validation/paired_comparison.json`, validate its classification,
architecture identity, finite positive fixed/CLIP IPC2 and BIPS2 values, and
recompute the percentage improvement.  Bind the package and paired report
hashes in status.  Preserve a successful peer if the other point fails.

- [ ] **Step 7: Run runner tests and verify GREEN**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2RunnerTests -v
```

Expected: all runner tests pass.

- [ ] **Step 8: Commit**

```bash
git add workflow/experiments/transient_rom_balanced2.py \
  tests/test_transient_rom_balanced2.py
git commit -m "feat: run thermal-first transient ROM Balanced-2"
```

### Task 3: Deterministic Steady-versus-Transient Report and CLI

**Files:**
- Modify: `workflow/experiments/transient_rom_balanced2.py`
- Modify: `tests/test_transient_rom_balanced2.py`
- Modify: `docs/transient_rom_usage_zh.md`

**Interfaces:**
- Produces: `summarize_validation_set(inputs: dict, output_root: Path, require_r2: bool) -> dict`.
- Produces CLI flags: `--selection`, `--canonical-r1-root`,
  `--periodic-r1-root`, `--steady-baseline-csv`, `--config`, `--output-root`,
  `--run-r2`, and `--summarize-only`.
- Writes `summary.json`, `summary.csv`, root `status.json`, and per-point status files.

- [ ] **Step 1: Write failing report tests**

Create two validated thermal/R2 checkpoints and two baseline rows.  Assert exact
CSV header order, six-decimal temperature presentation without truncating JSON,
recomputed steady/transient percentage changes, win/tie/loss counts, mean
changes, and SHA-256 identities.  Add tests that thermal-only reporting leaves
all measured transient fields empty and that a ROM prediction cannot satisfy
`require_r2=True`.

- [ ] **Step 2: Run report tests and verify RED**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2.Balanced2ReportTests -v
```

Expected: missing summarizer/CLI behavior.

- [ ] **Step 3: Implement report publication**

Recompute:

```python
steady_gain = (steady_clip_bips2 / steady_fixed_bips2 - 1.0) * 100.0
transient_gain = (transient_clip_bips2 / transient_fixed_bips2 - 1.0) * 100.0
gain_shift = transient_gain - steady_gain
```

Publish JSON and CSV only after validating every required point.  Use existing
atomic `write_json`; write CSV through a unique temporary sibling followed by
`Path.replace`.  Classify exact zero as tie and retain signed losses.

- [ ] **Step 4: Implement CLI and documentation**

The CLI runs preflight, then either thermal/R2 orchestration or read-only
summarization.  Document the two periodic-R1 commands, the thermal command, the
later `--run-r2` command, output tree, phase states, expected runtime, and why
the report is exploratory rather than paper-equivalent.

- [ ] **Step 5: Run focused and broad tests**

Run:

```bash
source scripts/env.sh
python -m unittest tests.test_transient_rom_balanced2 -v
python -m unittest tests.test_transient tests.test_transient_rom \
  tests.test_workflow tests.test_balanced_experiment -v
python -m compileall -q workflow/experiments workflow/transient \
  tests/test_transient_rom_balanced2.py
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 6: Commit**

```bash
git add workflow/experiments/transient_rom_balanced2.py \
  tests/test_transient_rom_balanced2.py docs/transient_rom_usage_zh.md
git commit -m "feat: report steady and transient Balanced-2 results"
```

### Task 4: Real Thermal Smoke and Balanced-2 Handoff

**Files:**
- Runtime only: `runs/transient_rom/matmul_steady_init_smoke_20260811/`
- Runtime only: `runs/transient_rom_balanced2/validation_20260811/`

**Interfaces:**
- Consumes the existing MATMUL 32 kB/512 kB periodic R1 for the immediate smoke.
- Consumes the new two 64 kB/512 kB periodic R1 directories when complete.
- Produces validated runtime status and commands for user-owned long execution.

- [ ] **Step 1: Run the existing MATMUL thermal smoke without R2**

```bash
source scripts/env.sh
python -m workflow.run_lifting_pipeline \
  --r1-dir /home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB \
  --output-dir /home/zyjiang/Agenticflow/CLIP/runs/transient_rom/matmul_steady_init_smoke_20260811 \
  --config configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir /home/zyjiang/Agenticflow/CLIP/runs/transient_validation/matmul_32kB_512kB_lambda0020119_2ms_precision6_20260806_030743/transient/shared_r1 \
  --transient-rom-calibrate
```

Acceptance: both final branches pass full-grid PSS at `0.01 C`, report finite
validated frequency, and root state is validated.  Do not run R2 for this
32 kB smoke.

- [ ] **Step 2: Preflight the user-run Balanced-2 periodic R1 outputs**

Run `--summarize-only` only after both periodic `status.json` files say
`success`; expected failure before thermal artifacts is an explicit
“thermal checkpoint missing”, not an R1 identity failure.

- [ ] **Step 3: Provide the direct thermal and R2 commands**

Use a fresh `runs/transient_rom_balanced2/validation_20260811` output root.
First run without `--run-r2`; inspect `summary.json`.  Only after state
`thermal_validated`, rerun the same entry with `--run-r2` to create the fresh
R2 subroots and final measured comparison.

- [ ] **Step 4: Record handoff evidence**

Report test totals, smoke status, exact output paths, calibration/final HotSpot
call counts, validated frequencies, and—when the user-owned R2 phase finishes—
steady/transient BIPS improvements for both points.  Do not claim the long
Balanced-2 experiment is complete before its artifacts validate.
