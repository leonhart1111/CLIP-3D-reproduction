# Alpha and Thermal-Length Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an unscaled, R1-preserving HotSpot campaign that identifies strict-P1 thermal-proxy `alpha` and `lc_mm` with grouped holdout validation.

**Architecture:** Extend the existing area-quadrature proxy with an explicit characteristic length, then add a focused `workflow/thermal/identify_alpha_lc.py` module that owns deterministic placement sampling, grid-convergence analysis, unit-response cross-tier estimation, relative-temperature fitting, grouped validation, bootstrap reporting, and an orchestration CLI.  Existing McPAT/CACTI model construction and HotSpot materialization remain the sole producers of physical geometry and reference temperatures.

**Tech Stack:** Python 3.12 standard library, NumPy/SciPy when available through `scripts/env.sh`, existing CACTI/McPAT/HotSpot executables, `unittest` regression suite, JSON/CSV manifests.

## Global Constraints

- Do not modify, delete, or rerun gem5 R1.
- Do not run gem5 R2 or fit `lambda_wire` in this plan.
- Do not use paper temperature, paper area, or desired BIPS as labels.
- Reject global area scaling and require `local_resistance_scale == 1.0`.
- Use raw McPAT dynamic and leakage power without postprocessing multipliers.
- Fit strict-P1 relative temperature; do not assert or fit `beta`.
- Cache and reuse HotSpot cases only when their content identity matches.
- Preserve all prior results and write a new timestamped result root.

---

### Task 1: Explicit Thermal Characteristic Length

**Files:**
- Modify: `workflow/floorplan/optimize_layout.py`
- Modify: `workflow/thermal/calibrate_proxy.py`
- Test: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- Produces: `spatial_coupling(modules, side, cross_tier_weight, spatial_model="area-quadrature", quadrature_order=2, lc_mm=None) -> float`
- Changes: `proxy_temperature(..., lc_mm: float | None = None) -> float`; `None` preserves the historical `side / 2` behavior for old configurations.
- Consumes: existing module dictionaries and rectangle quadrature.

- [ ] **Step 1: Write failing kernel tests**

Test that explicit larger `lc_mm` increases coupling at nonzero distance, that self-coupling remains unchanged, that area quadrature is used, and that nonpositive/nonfinite `lc_mm` is rejected.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.AlphaLcKernelTests -v`

Expected: failure because `spatial_coupling` and the `lc_mm` argument do not exist.

- [ ] **Step 3: Extract and implement the spatial feature**

Move the spatial sum currently embedded in `proxy_temperature` into the stated function.  Default `lc_mm` to `side / 2` only for compatibility; use the explicit value in every new identification path.

- [ ] **Step 4: Run focused and workflow regression tests**

Run: `python -m unittest tests.test_alpha_lc_identification.AlphaLcKernelTests tests.test_workflow -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/floorplan/optimize_layout.py workflow/thermal/calibrate_proxy.py tests/test_alpha_lc_identification.py
git commit -m "feat: expose thermal proxy characteristic length"
```

### Task 2: Deterministic Legal Placement Design

**Files:**
- Create: `workflow/thermal/identify_alpha_lc.py`
- Modify: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- Produces: `placement_design(model_path: Path, utilization: float, requested: int = 13) -> list[dict]`
- Each returned record contains `label`, `layout`, `x_mm`, `y_mm`, and normalized `fx`, `fy`.
- Consumes: `baseline_layout`, `collision_area`, and `check_geometry`.

- [ ] **Step 1: Write failing placement tests**

Use a synthetic model with four cores and one L2.  Assert deterministic order, 13 distinct legal placements where geometry permits, inclusion of center/corner/edge/core-near labels, and a clear error when fewer than 11 distinct legal placements exist.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.PlacementDesignTests -v`

Expected: import failure because `identify_alpha_lc.py` does not exist.

- [ ] **Step 3: Implement nearest-legal deterministic sampling**

Generate target coordinates, search an ordered normalized fallback lattice, reject overlaps and duplicates at a `1e-9 mm` tolerance, validate each layout, and never alter module area or power.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_alpha_lc_identification.PlacementDesignTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/thermal/identify_alpha_lc.py tests/test_alpha_lc_identification.py
git commit -m "feat: add deterministic thermal placement design"
```

### Task 3: Grid-Convergence and Content-Addressed HotSpot Cases

**Files:**
- Modify: `workflow/thermal/identify_alpha_lc.py`
- Modify: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- Produces: `case_identity(model, layout, physical, stimulus) -> str`
- Produces: `evaluate_grid_convergence(records: list[dict], grids=(32, 64, 128), tolerance_c=0.10) -> dict`
- Produces: `run_hotspot_case(...) -> dict`, reusing existing cases only when the full identity matches.

- [ ] **Step 1: Write failing convergence and identity tests**

Assert that grid 64 is selected when both 64 and 128 differ by at most 0.10 C and ordering matches; grid 128 is selected when 64 fails; missing layouts, changed thermal stack, changed model content, or ordering inversions invalidate reuse/acceptance.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.GridConvergenceTests -v`

Expected: missing API failures.

- [ ] **Step 3: Implement identity, reuse, and convergence reporting**

Hash canonical JSON plus source artifact hashes.  Preserve each grid result.  Record per-layout absolute differences and rank order.  Do not delete mismatched prior cases; write the new identity into a separate case directory.

- [ ] **Step 4: Verify GREEN and HotSpot integration smoke**

Run: `python -m unittest tests.test_alpha_lc_identification.GridConvergenceTests -v`

Then run one synthetic/unit-power HotSpot smoke through the CLI and confirm `thermal_result.json`, `hotspot_manifest.json`, and `case_manifest.json` exist.

- [ ] **Step 5: Commit**

```bash
git add workflow/thermal/identify_alpha_lc.py tests/test_alpha_lc_identification.py
git commit -m "feat: add reproducible HotSpot convergence cases"
```

### Task 4: Unit-Power Cross-Tier Evidence

**Files:**
- Modify: `workflow/thermal/identify_alpha_lc.py`
- Modify: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- Produces: `unit_power_model(model: dict, source_name: str, source_power_w=1.0) -> dict`
- Produces: `estimate_cross_tier_weight(matched_responses: list[dict], bootstrap_samples=1000, seed=20260813) -> dict`
- Report fields: estimate, 95% interval, per-case ratios, stability metrics, and acceptance.

- [ ] **Step 1: Write failing unit-response tests**

Assert exactly one selected source has 1 W total power, all others have zero, original model remains unchanged, matched response ratios use temperature rise above ambient, whole-case bootstrap is deterministic, and unstable/invalid denominators are rejected.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.UnitResponseTests -v`

Expected: missing API failures.

- [ ] **Step 3: Implement stimulus and robust estimator**

Use the median ratio for the estimate and whole-case bootstrap percentiles for uncertainty.  Mark scalar coupling inadequate when the interval is nonfinite or relative interval width exceeds the predeclared 50% stability ceiling.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_alpha_lc_identification.UnitResponseTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/thermal/identify_alpha_lc.py tests/test_alpha_lc_identification.py
git commit -m "feat: identify cross-tier unit thermal response"
```

### Task 5: Robust Alpha/Lc Fit and Grouped Validation

**Files:**
- Modify: `workflow/thermal/identify_alpha_lc.py`
- Modify: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- Produces: `fit_alpha_lc(samples: list[dict], cross_tier_weight: float, lc_bounds_ratio=(0.02, 4.0), huber_delta_c=0.5) -> dict`
- Produces: `cross_validate_alpha_lc(samples, ...) -> dict`
- Produces: `bootstrap_alpha_lc(samples, ..., samples_count=1000, seed=20260813) -> dict`
- Samples carry whole-work-point group keys, workload, size, reference/candidate layouts, HotSpot delta T, and die side.

- [ ] **Step 1: Write failing synthetic-recovery tests**

Construct multiple grouped layouts from known `alpha` and `lc_mm`, add one bounded outlier, and assert recovery within 5%, nonnegative alpha, finite confidence intervals, rank-two diagnostics, leave-one-workload-out reports, and rejection when all spatial features are degenerate.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.AlphaLcFitTests -v`

Expected: missing fit API failures.

- [ ] **Step 3: Implement robust nested fit**

Search `lc_mm / die_side` in log space, optimize nonnegative alpha with Huber loss, refine with bounded scalar minimization, calculate Jacobian singular values/condition, and prohibit boundary promotion.

- [ ] **Step 4: Implement grouped metrics**

Compute Delta-T MAE/RMSE, per-group Spearman without requiring SciPy stats, selected-layout HotSpot regret, leave-one-workload-out and size-holdout summaries, and acceptance against the frozen spec thresholds.

- [ ] **Step 5: Verify GREEN**

Run: `python -m unittest tests.test_alpha_lc_identification.AlphaLcFitTests -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add workflow/thermal/identify_alpha_lc.py tests/test_alpha_lc_identification.py
git commit -m "feat: fit alpha and thermal length with grouped validation"
```

### Task 6: Campaign CLI and Physical Contract Gates

**Files:**
- Modify: `workflow/thermal/identify_alpha_lc.py`
- Create: `configs/experiments/alpha_lc_unscaled_identification.json`
- Modify: `tests/test_alpha_lc_identification.py`

**Interfaces:**
- CLI subcommands: `grid-convergence`, `unit-response`, `real-power`, `fit`, and `run`.
- `run` consumes an explicit list/root of completed R1 work points and creates a timestamped or user-specified output directory.
- Configuration records five workloads, nine size anchors, 13 placements, physical stack, raw-power contract, bootstrap seed/count, and frozen acceptance thresholds.

- [ ] **Step 1: Write failing contract/CLI tests**

Assert rejection of scaled area metadata, `local_resistance_scale != 1`, non-raw power provenance, missing R1 inputs, output inside an R1 directory, and any attempted R1/R2 launch option.  Assert `--plan-only` produces the exact expected work-point/case manifest without running tools.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_alpha_lc_identification.CampaignCliTests -v`

Expected: missing CLI/config failures.

- [ ] **Step 3: Implement CLI and immutable manifest**

Reuse existing model-building and HotSpot APIs, expose bounded parallel jobs, write progress after every completed case, and make interrupted campaigns safely resumable by case identity.

- [ ] **Step 4: Verify GREEN and plan-only inventory**

Run: `python -m unittest tests.test_alpha_lc_identification.CampaignCliTests -v`

Run the real `--plan-only` command against the existing paper R1 root; require 45 selected work points and 585 intended real-power placements before launching HotSpot.

- [ ] **Step 5: Commit**

```bash
git add workflow/thermal/identify_alpha_lc.py configs/experiments/alpha_lc_unscaled_identification.json tests/test_alpha_lc_identification.py
git commit -m "feat: orchestrate unscaled alpha Lc campaign"
```

### Task 7: Execute Experiments and Publish Evidence

**Files:**
- Create under ignored result root: `results/parameter_studies/unscaled_alpha_lc_<timestamp>/...`
- Create: `docs/alpha_lc_identification_zh.md`
- Modify only after acceptance: a new formal config derived from `configs/experiments/alpha_lc_unscaled_identification.json`

**Interfaces:**
- Consumes: completed R1, corrected unscaled model builder, campaign CLI.
- Produces: grid report, unit-response report, samples CSV, fit/cross-validation/bootstrap/acceptance JSON, and Chinese methods/results document.

- [ ] **Step 1: Run grid convergence**

Launch the six-layout 32/64/128 campaign.  Stop if no grid satisfies the frozen threshold.

- [ ] **Step 2: Run at least 72 unit-response cases**

Estimate and freeze `wcross`.  Stop formal promotion if the scalar coupling stability gate fails; retain diagnostic results.

- [ ] **Step 3: Build corrected unscaled models from existing R1**

For 45 work points, run only CACTI/McPAT conversion.  Verify no R1 file modification timestamps or hashes change.

- [ ] **Step 4: Run the 585-case real-power HotSpot campaign**

Use bounded parallelism, resume by identity, and monitor case failures without deleting successful samples.

- [ ] **Step 5: Fit and validate alpha/Lc**

Run grouped fitting, 1000 whole-work-point bootstrap resamples, leave-one-workload-out validation, and locked acceptance evaluation.

- [ ] **Step 6: Run complete verification**

Run: `python -m unittest tests.test_alpha_lc_identification tests.test_cache_alignment tests.test_workflow -v`

Then run the full applicable suite and `git diff --check`.

- [ ] **Step 7: Publish without concealing failure**

Document measured values, confidence intervals, every acceptance check, runtime, failures, and the exact evidence identity.  Create a formal configuration only if every promotion gate passes; otherwise preserve the result as rejected diagnostic evidence.

- [ ] **Step 8: Commit source documentation**

```bash
git add docs/alpha_lc_identification_zh.md
git commit -m "docs: report unscaled alpha Lc identification"
```
