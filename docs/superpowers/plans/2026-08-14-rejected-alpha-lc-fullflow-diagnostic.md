# Rejected Alpha/Lc Full-Flow Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one reproducible, explicitly non-formal MATMUL 32 kB/512 kB fixed-bin versus CLIP-3D full-flow comparison consume the rejected unscaled alpha/Lc evidence exactly.

**Architecture:** Extend the existing thermal-proxy interface with an optional die-normalized characteristic length and forward the parameter plus the identification campaign's HotSpot materialization controls through the normal lifting pipeline.  Add a tracked rejected-evidence manifest and one exploratory configuration; preserve all historical defaults and R1 data.

**Tech Stack:** Python 3.12 standard library, existing unittest suite, JSON experiment configurations, CACTI, McPAT, HotSpot, gem5.

## Global Constraints

- Reuse canonical gem5 R1 read-only; never modify or rerun it.
- Label the fit and the resulting comparison as rejected/non-formal.
- Use alpha `3.348558894986864`, `Lc/side=0.3525311644629249`, and cross-tier weight `0.8378797681280803` at full precision.
- Use beta `0` only as the strict-P1 unidentified constant, not as an identified value.
- Retain lambda `0.0020119160767721133` only with its rejected exploratory provenance.
- Use grid 64, module-granularity input, compact trace, precision 9, `Rconv=1.042`, and resistance scales 1.
- Run fresh fixed-bin and CLIP-3D R2 outputs; never reuse pre-alignment R2.

---

### Task 1: Thread the Characteristic Length Through the Optimizer

**Files:**
- Modify: `tests/test_workflow.py`
- Modify: `workflow/floorplan/optimize_layout.py`
- Modify: `workflow/run_lifting_pipeline.py`

**Interfaces:**
- Consumes: `layout_optimizer.lc_die_side_ratio: float | null`
- Produces: `optimize(..., lc_die_side_ratio: float | None = None) -> dict`
- Records: `parameters.lc_die_side_ratio` and `parameters.lc_mm`

- [ ] **Step 1: Write failing optimizer and forwarding tests**

Add tests that call `optimize()` with two explicit positive ratios and assert the recorded ratio/millimetres and selected proxy values differ.  Extend `test_pipeline_forwards_discrete_partition_options_to_optimizer` with `lc_die_side_ratio=0.3525311644629249` and assert the report records it.  Add validation tests for zero, negative, NaN, and infinity.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest \
  tests.test_workflow.WorkflowTests.test_pipeline_forwards_discrete_partition_options_to_optimizer \
  tests.test_workflow.WorkflowTests.test_optimizer_uses_explicit_lc_ratio -v
```

Expected: failure because `optimize()` does not accept or report the ratio.

- [ ] **Step 3: Implement the minimal length plumbing**

Append `lc_die_side_ratio: float | None = None` to `optimize()`.  Preserve the historical effective ratio `0.5` when omitted.  Validate with `math.isfinite` and positivity, compute `lc_mm = side * ratio`, pass it to every `proxy_temperature()` call, forward the config value from `optimize_clip3d_layout()`, and record both values in the report.

- [ ] **Step 4: Run focused and alpha/Lc tests**

```bash
python -m unittest tests.test_workflow tests.test_alpha_lc_identification -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/floorplan/optimize_layout.py workflow/run_lifting_pipeline.py tests/test_workflow.py
git commit -m "feat: thread fitted thermal length through floorplanner"
```

### Task 2: Preserve the Parameter-Identification HotSpot Contract

**Files:**
- Modify: `tests/test_workflow.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Modify: `workflow/floorplan/generate_hotspot_inputs.py`

**Interfaces:**
- Consumes: `physical.input_granularity`, `physical.compact_trace`, and `physical.ptrace_precision`
- Produces: identical materialization options for fixed-bin, CLIP-3D, comparison, and fallback paths

- [ ] **Step 1: Write a failing materialization-forwarding test**

Patch `workflow.run_lifting_pipeline.materialize`, exercise fixed-bin and paper-single CLIP paths with a physical configuration containing `input_granularity="module"`, `compact_trace=true`, and `ptrace_precision=9`, then assert each call receives those exact keyword arguments.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest tests.test_workflow.WorkflowTests.test_pipeline_forwards_hotspot_materialization_contract -v
```

Expected: failure because the pipeline currently relies on grid-cell/default trace arguments.

- [ ] **Step 3: Validate and forward the controls**

Validate `input_granularity` against `grid-cell/module`, require a real boolean for `compact_trace`, and require a positive integer precision.  Forward the three fields to all four `materialize()` call sites without changing defaults for old configurations.

- [ ] **Step 4: Make grid provenance dynamic**

Replace the hard-coded `"32x32 per tier"` manifest string with `f"{grid_size}x{grid_size} per tier"`.  Assert a grid-64 manifest records `64x64 per tier` and module granularity.

- [ ] **Step 5: Verify**

```bash
python -m unittest tests.test_workflow tests.test_alpha_lc_identification -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add workflow/run_lifting_pipeline.py workflow/floorplan/generate_hotspot_inputs.py tests/test_workflow.py
git commit -m "fix: preserve thermal identification materialization contract"
```

### Task 3: Add Rejected-Evidence Provenance and Diagnostic Configuration

**Files:**
- Create: `manifests/parameter_provenance/unscaled_alpha_lc_rejected_20260814.json`
- Create: `configs/experiments/clip3d_unscaled_alpha_lc_rejected_diagnostic.json`
- Modify: `tests/test_workflow.py`

**Interfaces:**
- Manifest records fit-report SHA-256 `918f93b775d5131c98b5c8300dae468b8bc2a87e29f66cd109c53bcd25795c89`, exact values, physical identity, and `accepted_for_formal_or_shared_use=false`.
- Config drives both layout methods and contains no formal-acceptance claim.

- [ ] **Step 1: Write a failing configuration contract test**

Load the two proposed files and assert exact values, rejected status, grid/module controls, unscaled stack, tier `[1]`, paper-single policy, area quadrature order 2, discrete-partition search, and rejected lambda provenance.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest tests.test_workflow.WorkflowTests.test_rejected_alpha_lc_diagnostic_config_is_exact -v
```

Expected: failure because the files do not exist.

- [ ] **Step 3: Add the manifest and configuration**

Derive the config from the current exploratory raw-power configuration but replace scaled thermal settings with the exact unscaled identification contract.  Do not mark `formal_validation.accepted` true.

- [ ] **Step 4: Verify focused and complete workflow tests**

```bash
python -m unittest tests.test_workflow tests.test_alpha_lc_identification -v
python -m workflow.run_lifting_pipeline --help
```

Expected: all tests pass and CLI help exits zero.

- [ ] **Step 5: Commit**

```bash
git add manifests/parameter_provenance/unscaled_alpha_lc_rejected_20260814.json \
  configs/experiments/clip3d_unscaled_alpha_lc_rejected_diagnostic.json \
  tests/test_workflow.py
git commit -m "test: configure rejected alpha Lc diagnostic"
```

### Task 4: Produce the User-Run Command Handoff

**Files:**
- No production files

**Interfaces:**
- Consumes canonical R1 and the new configuration
- Produces fresh `fixed-bin` and `clip3d` output trees plus a terminal comparison table

- [ ] **Step 1: Verify canonical inputs and executables without running R2**

```bash
test -s /home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/r1_metadata.json
test -s /home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB/stats.txt
scripts/check_tools.sh
```

- [ ] **Step 2: Hand off sequential fixed-bin and CLIP-3D commands**

Use one timestamped root, the same config, `--run-r2`, and separate fresh output directories.  Do not start the commands inside the agent session.

- [ ] **Step 3: Hand off a result-summary command**

Read both `pipeline_summary.json` files and print Tmax, sustainable frequency, critical cycles, IPC2, BIPS2, and CLIP-versus-fixed percentage changes.  Fail clearly if either output is incomplete.
