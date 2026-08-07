# Optional Transient-Thermal ROM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gated, non-formal `transient-rom` mode that learns a POD continuous-time thermal ROM from ten HotSpot calibration/holdout runs, while preserving the default steady CLIP-3D flow.

**Architecture:** Eight deterministic L2-anchor HotSpot PRBS traces identify one low-dimensional continuous state-space model; two independent real-workload HotSpot cases gate its use. The optimizer evaluates periodic steady state and frequency with matrix recurrences only, then real HotSpot and R2 verify its selected layout.

**Tech Stack:** Python 3, `unittest`, NumPy, SciPy (`linalg.expm`, `linalg.logm`, `spatial.Delaunay`), gem5, McPAT, CACTI, HotSpot.

## Global Constraints

- `--thermal-mode steady` is the default and retains existing Eq. (13), calculations, artifacts and tests unchanged.
- ROM scope is fixed die/stack/cooling/fixed non-L2 modules; only `shared_l2` moves inside existing `allowed_l2_tiers`.
- Dynamic power scales by `s=f/f0`, leakage is fixed, and window duration scales by `1/s`.
- Calibration is exactly eight training HotSpot calls and two holdout HotSpot calls. Final validation calls are reported separately.
- `workflow/transient/rom/optimize_layout.py` makes zero HotSpot calls.
- Holdout PSS, grid RMSE, peak error and safety classification must all pass before optimization.
- Every ROM artifact declares `non_formal: true`, `paper_equivalent: false`, `thermal_mode: transient-rom`.
- `bips2_trans` exists only after final real-HotSpot PSS validation and R2; it never overwrites steady `bips2`.
- Do not modify `workflow/thermal/sustainable_frequency.py`, `workflow/floorplan/optimize_layout.py`, source R1 directories, existing R2 results or live experiments.
- Use `apply_patch` for edits and commit only files introduced/changed by each task.

---

## File Structure

| Path | Responsibility |
|---|---|
| `workflow/transient/generate_hotspot_trace.py` | Frequency-scaled and repeated real trace materialization. |
| `workflow/transient/run_hotspot_transient.py` | Full-grid temperature parsing and PSS evidence. |
| `workflow/transient/verify_sustainable_frequency.py` | Layout-neutral real HotSpot periodic frequency search. |
| `workflow/transient/rom/contracts.py` | Settings, hashes, identity and acceptance contracts. |
| `workflow/transient/rom/calibration_design.py` | Legal anchors/holdouts, PRBS input and interpolation geometry. |
| `workflow/transient/rom/materialize_calibration.py` | PRBS power-window and HotSpot case files. |
| `workflow/transient/rom/pod_state_space.py` | POD, state fitting, continuous conversion and persistence. |
| `workflow/transient/rom/layout_rom.py` | L2 response interpolation, ROM PSS and frequency evaluation. |
| `workflow/transient/rom/calibrate_rom.py` | Ten-case calibration, holdout gate and package publication. |
| `workflow/transient/rom/optimize_layout.py` | ROM-only L2 search with existing wire objective conventions. |
| `workflow/transient/rom/run_pipeline.py` | Windows, ROM, final HotSpot/R2 and summary orchestration. |
| `workflow/run_lifting_pipeline.py` | Explicit mode dispatch; steady path stays untouched. |
| `configs/experiments/clip3d_transient_rom_exploratory.json` | Opt-in, non-formal ROM configuration. |
| `docs/transient_rom_usage_zh.md` | Operating guide, gates and comparison rules. |
| `tests/test_transient.py` | Shared HotSpot trace/PSS/frequency regression tests. |
| `tests/test_transient_rom.py` | ROM contract, model, gate, optimizer and integration tests. |

---

### Task 1: Add reusable real-HotSpot PSS and frequency-search primitives

**Files:**
- Modify: `workflow/transient/generate_hotspot_trace.py`
- Modify: `workflow/transient/run_hotspot_transient.py`
- Create: `workflow/transient/verify_sustainable_frequency.py`
- Modify: `tests/test_transient.py`

**Interfaces:**

```python
def materialize_trace(..., frequency_scale: float = 1.0,
                      period_repeats: int = 1) -> dict: ...
def parse_ttrace_grid(path: Path) -> tuple[list[str], list[list[float]]]: ...
def summarize_period_end_convergence(rows_k: list[list[float]],
                                     windows_per_period: int) -> dict: ...
def search_layout_frequency(modules_path: Path, layout_path: Path,
                            power_windows_path: Path, output_dir: Path,
                            config_path: Path, *, frequencies_ghz: list[float],
                            period_repeats: int, pss_tolerance_c: float,
                            frequency_tolerance_ghz: float, hotspot: Path) -> dict: ...
```

- [ ] **Step 1: Write failing regression tests**

```python
def test_frequency_scaled_trace_repeats_period_and_preserves_power_split(self):
    result = materialize_trace(..., frequency_scale=0.5, period_repeats=3)
    self.assertEqual(result["frequency_scaling"]["dynamic_power_scale"], 0.5)
    self.assertEqual(result["frequency_scaling"]["leakage_power_scale"], 1.0)
    self.assertEqual(result["windows_per_period"], 2)

def test_period_end_convergence_uses_full_grid(self):
    result = summarize_period_end_convergence(
        [[300.0, 305.0], [301.0, 310.0], [300.0, 305.0], [301.25, 310.0]], 2
    )
    self.assertAlmostEqual(result["last_delta_max_c"], 0.25)
    self.assertEqual(result["last_delta_unit_index"], 0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient.TransientTraceTests -v`  
Expected: FAIL because the new keywords and helpers are absent.

- [ ] **Step 3: Implement the primitives**

Scale each dynamic trace row by `frequency_scale`, leave leakage unchanged, recompute total, set sampling interval to `nominal / frequency_scale`, and repeat the source period. Parse finite full-grid rows, compare all cells at adjacent period ends, and include the final period's initial state in its safety peak. Port the reviewed verifier branch's local safe/unsafe bracket refinement as `search_layout_frequency()`; it takes arbitrary modules/layout paths and does not require IPC2. Keep `verify_layout()` as a compatibility wrapper that preflights a steady directory and attaches IPC2.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_transient -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/generate_hotspot_trace.py workflow/transient/run_hotspot_transient.py workflow/transient/verify_sustainable_frequency.py tests/test_transient.py
git commit -m "feat: add transient frequency search primitives"
```

### Task 2: Define strict ROM settings and artifact identity

**Files:**
- Create: `workflow/transient/rom/__init__.py`
- Create: `workflow/transient/rom/contracts.py`
- Create: `tests/test_transient_rom.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ROMSettings:
    sample_interval_ms: float; calibration_windows: int; prbs_seed: int
    prbs_fraction: float; pod_energy_threshold: float; max_pod_rank: int
    ridge: float; max_condition_number: float; pss_period_repeats: int
    pss_tolerance_c: float; frequency_tolerance_ghz: float
    max_holdout_peak_error_c: float; max_holdout_grid_rmse_c: float
    search_grid_points_per_axis: int; refinement_starts: int

def parse_settings(config: dict) -> ROMSettings: ...
def rom_input_identity(...) -> dict: ...
def require_accepted_package(package_dir: Path, identity: dict) -> dict: ...
```

- [ ] **Step 1: Write failing contract tests**

```python
def test_parse_settings_requires_exact_8_plus_2_and_positive_thresholds(self):
    with self.assertRaisesRegex(ValueError, "calibration_runs must equal 8"):
        parse_settings({"transient_rom": {"calibration_runs": 7}})

def test_rejects_accepted_package_with_changed_power_identity(self):
    with self.assertRaisesRegex(ValueError, "power trace identity"):
        require_accepted_package(self.package, {"power_trace": "changed"})
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.ROMContractTests -v`  
Expected: FAIL because package imports and functions are absent.

- [ ] **Step 3: Implement settings and identity**

Use exact defaults: 8 training, 2 holdouts, 64 windows, 2 ms, seed `20260807`, PRBS fraction `0.20`, POD energy `0.999`, rank cap `16`, ridge `1e-8`, condition cap `1e10`, 20 PSS repeats, 0.01 C PSS/frequency tolerances, 1.0 C peak error, 0.75 C grid RMSE, 25 search points per axis and five refinement starts. Reject non-finite/range-invalid values and any count other than 8/2. Identity includes canonical R1 metadata hash, power trace identity, modules/layout geometry hashes, configuration/HotSpot hashes, grid/stack/cooling and allowed tiers. Reject a reusable package if any identity field differs.

- [ ] **Step 4: Run contract tests**

Run: `python -m unittest tests.test_transient_rom.ROMContractTests -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/__init__.py workflow/transient/rom/contracts.py tests/test_transient_rom.py
git commit -m "feat: define transient ROM contracts"
```

### Task 3: Generate legal calibration designs and informative power inputs

**Files:**
- Create: `workflow/transient/rom/calibration_design.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
def build_design(model: dict, allowed_l2_tiers: list[int], settings: ROMSettings) -> dict: ...
def layout_for_point(base_layout: dict, point: dict) -> dict: ...
def make_prbs_input(module_names: list[str], l2_name: str,
                    settings: ROMSettings) -> dict: ...
def interpolation_domain(design: dict, tier: int) -> dict: ...
```

- [ ] **Step 1: Write failing geometry/PRBS tests**

```python
def test_two_tier_design_has_four_legal_anchors_per_tier(self):
    design = build_design(self.model(), [0, 1], self.settings())
    self.assertEqual(len(design["training"]), 8)
    self.assertEqual({x["tier"] for x in design["holdout"]}, {0, 1})

def test_single_tier_design_has_eight_anchors_and_two_holdouts(self):
    design = build_design(self.model(), [1], self.settings())
    self.assertEqual(len(design["training"]), 8)
    self.assertEqual(len(design["holdout"]), 2)

def test_prbs_is_repeatable_positive_and_independent(self):
    a = make_prbs_input(["core0", "shared_l2"], "shared_l2", self.settings())
    b = make_prbs_input(["core0", "shared_l2"], "shared_l2", self.settings())
    self.assertEqual(a, b)
    self.assertNotEqual(a["multipliers"]["core0"], a["multipliers"]["shared_l2"])
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.CalibrationDesignTests -v`  
Expected: FAIL because design APIs are absent.

- [ ] **Step 3: Implement deterministic geometry**

Begin with `baseline_layout()`, locate exactly one L2, and verify each trial using `check_geometry()`. With `[0, 1]`, obtain four collision-free extremes per tier (lower-left, lower-right, upper-left, upper-right); move an invalid extreme toward die center by deterministic bisection. With one allowed tier, create four corners plus four edge midpoints. Reject duplicate points within `1e-9` mm. Create holdouts at interior fractions `(.37,.61)` and `(.63,.39)`. For two tiers persist rectangle corners for bilinear interpolation; for one tier persist a `scipy.spatial.Delaunay` triangulation of eight points. Create one positive sequence `1 ± prbs_fraction` for every module name with a name-indexed seeded PRNG.

- [ ] **Step 4: Run design tests**

Run: `python -m unittest tests.test_transient_rom.CalibrationDesignTests -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/calibration_design.py tests/test_transient_rom.py
git commit -m "feat: add deterministic ROM calibration design"
```

### Task 4: Materialize the eight training and two holdout HotSpot cases

**Files:**
- Create: `workflow/transient/rom/materialize_calibration.py`
- Create: `workflow/transient/rom/calibrate_rom.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
def build_prbs_power_windows(source_windows: dict, multipliers: dict[str, list[float]],
                             window_count: int) -> dict: ...
def execute_calibration_cases(..., hotspot: Path) -> dict: ...
```

- [ ] **Step 1: Write failing materialization tests**

```python
def test_calibration_emits_exactly_eight_training_and_two_holdout_cases(self):
    report = execute_calibration_cases(..., hotspot=self.hotspot)
    self.assertEqual(report["training_hotspot_calls"], 8)
    self.assertEqual(report["holdout_hotspot_calls"], 2)

def test_prbs_windows_preserve_power_triplets(self):
    result = build_prbs_power_windows(self.raw_windows(), self.multipliers(), 64)
    self.assertTrue(all(
        m["total_power_w"] == m["dynamic_power_w"] + m["leakage_power_w"]
        for w in result["windows"] for m in w["modules"]
    ))
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.ROMCalibrationCaseTests -v`  
Expected: FAIL because materialization APIs are absent.

- [ ] **Step 3: Implement cases and audit files**

Create schema-compatible 64-window PRBS power records; multiply dynamic and leakage module values by their positive PRBS multiplier, recompute total, validate every triplet, and preserve raw provenance. Materialize/execute eight ambient-start anchor traces. Materialize/execute two ambient-start real-workload holdouts at `f0` and `0.6*f0 + 0.4*fmin`, with repeated periods and full-grid PSS evidence. Record command, elapsed time, geometry/power/trace SHA-256 values and reject a pre-existing nonempty output.

- [ ] **Step 4: Run materialization tests**

Run: `python -m unittest tests.test_transient_rom.ROMCalibrationCaseTests -v`  
Expected: PASS with mocked HotSpot call count equal to ten.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/materialize_calibration.py workflow/transient/rom/calibrate_rom.py tests/test_transient_rom.py
git commit -m "feat: materialize transient ROM calibration cases"
```

### Task 5: Fit a stable POD continuous-time state-space ROM

**Files:**
- Create: `workflow/transient/rom/pod_state_space.py`
- Modify: `workflow/transient/rom/calibrate_rom.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class StateSpaceModel:
    temperature_basis: numpy.ndarray
    a_continuous: numpy.ndarray
    b_fixed: numpy.ndarray
    b_l2_anchors: dict[str, numpy.ndarray]
    module_names: tuple[str, ...]

def fit_state_space(training_cases: list[dict], settings: ROMSettings) -> tuple[StateSpaceModel, dict]: ...
def discretize(model: StateSpaceModel, duration_s: float,
               b_l2: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]: ...
def save_model(path: Path, model: StateSpaceModel, metadata: dict) -> None: ...
def load_model(path: Path) -> tuple[StateSpaceModel, dict]: ...
```

- [ ] **Step 1: Write failing numerical tests**

```python
def test_augmented_discretization_matches_first_order_rc(self):
    model = self.first_order_model(a=-2.0, b=3.0)
    a_d, b_d = discretize(model, 0.5, model.b_l2_anchors["a0"])
    self.assertAlmostEqual(a_d[0, 0], math.exp(-1.0), places=12)
    self.assertAlmostEqual(b_d[0, 0], 1.5 * (1.0 - math.exp(-1.0)), places=12)

def test_fit_rejects_unstable_continuous_pole(self):
    with self.assertRaisesRegex(ValueError, "unstable continuous-time state matrix"):
        fit_state_space(self.unstable_training_cases(), self.settings())
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.StateSpaceTests -v`  
Expected: FAIL because the model module is absent.

- [ ] **Step 3: Implement POD and state identification**

Stack relative-to-ambient full-grid snapshots in deterministic case/time order. Use SVD and choose the smallest rank reaching `pod_energy_threshold`, failing if rank exceeds `max_pod_rank`. Fit shared `A_d`, all fixed-module input columns, and one L2 column per anchor by ridge least squares. Save singular values, residuals and Gram condition number. Compute the continuous augmented matrix with `scipy.linalg.logm([[A_d,B_d],[0,I]]) / dt`; reject material imaginary parts and any real eigenvalue above `1e-10`. Obtain every discrete `A_d,B_d` with `scipy.linalg.expm` of the continuous augmented matrix. Never invert `A_c`. Persist arrays in `pod_model.npz` and scalar audit data in `fit_report.json`.

- [ ] **Step 4: Run numerical tests**

Run: `python -m unittest tests.test_transient_rom.StateSpaceTests tests.test_transient -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/pod_state_space.py workflow/transient/rom/calibrate_rom.py tests/test_transient_rom.py
git commit -m "feat: fit transient POD state-space ROM"
```

### Task 6: Evaluate ROM PSS/frequency and enforce the two holdout gates

**Files:**
- Create: `workflow/transient/rom/layout_rom.py`
- Modify: `workflow/transient/rom/calibrate_rom.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
def interpolate_l2_input(model: StateSpaceModel, design: dict,
                         tier: int, x_mm: float, y_mm: float) -> numpy.ndarray: ...
def evaluate_layout_rom(model: StateSpaceModel, design: dict, power_windows: dict,
                        layout: dict, frequency_ghz: float, settings: ROMSettings,
                        config: dict) -> dict: ...
def find_rom_sustainable_frequency(...) -> dict: ...
def validate_holdouts(...) -> dict: ...
```

- [ ] **Step 1: Write failing interpolation/gate tests**

```python
def test_anchor_interpolation_is_exact_and_extrapolation_fails(self):
    self.assertTrue(numpy.array_equal(interpolate_l2_input(..., 0, 0.0, 0.0), self.corner_b()))
    with self.assertRaisesRegex(ValueError, "outside ROM interpolation domain"):
        interpolate_l2_input(..., 0, -0.01, 0.5)

def test_holdout_gate_rejects_peak_grid_or_safety_mismatch(self):
    result = validate_holdouts(self.failed_holdouts(), self.settings())
    self.assertFalse(result["accepted"])
    self.assertIn("peak_temperature_error", result["failure_reasons"])
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.ROMEvaluationTests -v`  
Expected: FAIL because evaluator APIs are absent.

- [ ] **Step 3: Implement evaluation and gates**

Build every real-workload input vector in fixed module-name order as `leakage_power_w + s*dynamic_power_w`, reject malformed modules, discretize at `duration_s/s`, and iterate from ambient until reconstructed full-grid PSS converges. Compute inclusive final-period peak. Implement bilinear interpolation for two-tier rectangles and Delaunay/barycentric interpolation for the single-tier eight-point domain; reject convex-hull exterior, degenerate simplex and negative weights below `-1e-12`. Reuse Task 1 local-bracket semantics for ROM frequency search. Compare ROM/HotSpot holdouts at identical geometry, input identity and frequency. Publish `validation_report.json`; create `rom_acceptance.json` only if PSS, RMSE <= 0.75 C, peak error <= 1.0 C and safety classes all pass.

- [ ] **Step 4: Run evaluator tests**

Run: `python -m unittest tests.test_transient_rom.ROMEvaluationTests -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/layout_rom.py workflow/transient/rom/calibrate_rom.py tests/test_transient_rom.py
git commit -m "feat: gate transient ROM with holdout validation"
```

### Task 7: Search L2 layouts with the accepted ROM only

**Files:**
- Create: `workflow/transient/rom/optimize_layout.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
def optimize_transient_layout(modules_path: Path, package_dir: Path,
                              output_dir: Path, config: dict) -> dict: ...
```

- [ ] **Step 1: Write failing optimizer tests**

```python
def test_optimizer_selects_higher_rom_bips_without_hotspot(self):
    report = optimize_transient_layout(self.modules, self.accepted_package,
                                       self.output, self.config())
    self.assertEqual(report["hotspot_calls_inside_optimizer"], 0)
    self.assertTrue((self.output / "proposed_layout.json").is_file())

def test_optimizer_rejects_unaccepted_package(self):
    with self.assertRaisesRegex(ValueError, "ROM package is not accepted"):
        optimize_transient_layout(self.modules, self.rejected_package, self.output, self.config())
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_transient_rom.ROMOptimizerTests -v`  
Expected: FAIL because the optimizer is absent.

- [ ] **Step 3: Implement deterministic grid-plus-refinement search**

For each allowed tier enumerate a 25-by-25 lower-left-coordinate lattice, retain only geometry- and interpolation-legal points, then refine the five best seeds with eight-neighbor bounded pattern search. Use current wire functions and exact score `-IPC1*f_sus_trans_rom + lambda_wire*IPC1*wire_objective_cycles`. Record every legal seed, rejection reason, selected layout, frequency/temperature evidence, wire delays and `hotspot_calls_inside_optimizer: 0`. This module must not import `run_hotspot*` or `subprocess`.

- [ ] **Step 4: Run optimizer tests and static audit**

Run:

```bash
python -m unittest tests.test_transient_rom.ROMOptimizerTests -v
rg -n "run_hotspot|subprocess\.run" workflow/transient/rom/optimize_layout.py
```

Expected: tests PASS; `rg` reports no matches.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/optimize_layout.py tests/test_transient_rom.py
git commit -m "feat: optimize layouts with transient ROM"
```

### Task 8: Integrate explicit thermal mode, final validation and R2

**Files:**
- Create: `workflow/transient/rom/run_pipeline.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Create: `configs/experiments/clip3d_transient_rom_exploratory.json`
- Modify: `tests/test_workflow.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**

```python
def run_transient_rom_pipeline(source_r1_dir: Path, steady_preflight_dir: Path,
                               output_dir: Path, config_path: Path,
                               transient_r1_dir: Path | None,
                               calibrate: bool, execute_r2: bool) -> dict: ...
```

- [ ] **Step 1: Write failing dispatch tests**

```python
def test_steady_mode_calls_legacy_pipeline_without_rom_import(self):
    with patch("workflow.run_lifting_pipeline.run_pipeline") as legacy:
        self.invoke_cli("--thermal-mode", "steady")
    legacy.assert_called_once()

def test_rom_mode_runs_fixed_preflight_then_rom(self):
    with patch("workflow.run_lifting_pipeline.run_pipeline", return_value=self.preflight()) as steady, \
         patch("workflow.transient.rom.run_pipeline.run_transient_rom_pipeline") as rom:
        self.invoke_cli("--thermal-mode", "transient-rom", "--run-r2")
    self.assertFalse(steady.call_args.kwargs["execute_r2"])
    self.assertTrue(rom.call_args.kwargs["execute_r2"])

def test_rom_mode_rejects_legacy_transient_flag(self):
    with self.assertRaisesRegex(SystemExit, "cannot combine"):
        self.invoke_cli("--thermal-mode", "transient-rom", "--transient", "true")
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m unittest tests.test_workflow -v`  
Expected: FAIL because `--thermal-mode` does not exist.

- [ ] **Step 3: Implement end-to-end mode dispatch**

`steady` invokes the existing function unchanged. `transient-rom` builds/reuses `OUTPUT/steady_preflight` with the existing fixed-bin pipeline and R2 disabled, uses its modules/CACTI provenance, then writes only to sibling `OUTPUT/transient_rom`. It obtains a compatible periodic R1 via existing validation, calls `prepare_power_windows()` once, calibrates or loads an accepted identity-matching package, searches with Task 7, derives the current R2 latency vector, optionally runs R2, then invokes Task 1 real HotSpot search on the final layout. Emit `f_sus_trans_rom_pred_ghz`, `f_sus_trans_hotspot_ghz`, `bips1_trans_rom_pred`, `bips2_trans` and no ambiguous `bips2` field. Reject `--transient true` in ROM mode. Create an exploratory raw-power config with the exact Task 2 settings and a non-formal note; do not alter formal configs.

- [ ] **Step 4: Run synthetic pipeline tests**

Run: `python -m unittest tests.test_workflow tests.test_transient_rom -v`  
Expected: PASS with all tool invocations mocked.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/run_pipeline.py workflow/run_lifting_pipeline.py configs/experiments/clip3d_transient_rom_exploratory.json tests/test_workflow.py tests/test_transient_rom.py
git commit -m "feat: add optional transient ROM pipeline"
```

### Task 9: Document, regress and audit the final behavior

**Files:**
- Create: `docs/transient_rom_usage_zh.md`
- Modify: `tests/test_transient_rom.py`

- [ ] **Step 1: Write failing summary-schema test**

```python
def test_summary_separates_predicted_and_validated_transient_results(self):
    summary = self.completed_rom_summary()
    self.assertTrue(summary["non_formal"])
    self.assertFalse(summary["paper_equivalent"])
    self.assertIn("bips1_trans_rom_pred", summary)
    self.assertIn("bips2_trans", summary)
    self.assertNotIn("bips2", summary)
```

- [ ] **Step 2: Run schema test**

Run: `python -m unittest tests.test_transient_rom -v`  
Expected: PASS after Task 8 creates the audited summary.

- [ ] **Step 3: Write Chinese usage guide**

Document prerequisite tools, exploratory command examples for calibration and reuse, 8+2 call meanings, package tree, thresholds, failure conditions, `steady bips2` vs validated `bips2_trans`, and non-formal/paper-inequivalent limits.

- [ ] **Step 4: Run complete relevant regressions**

Run: `python -m unittest tests.test_transient tests.test_transient_rom tests.test_workflow -v`  
Expected: PASS. If the historical `test_lambda_wire_exploratory_config.py` cannot find its untracked past experiment result, report it as an independent pre-existing fixture gap and do not fabricate an artifact.

- [ ] **Step 5: Commit**

```bash
git add docs/transient_rom_usage_zh.md tests/test_transient_rom.py
git commit -m "docs: explain transient ROM workflow"
```

## Final Audit Checklist

- [ ] Steady default tests pass without importing ROM.
- [ ] Training/holdout counts are exactly 8/2 and all input/output hashes are recorded.
- [ ] Both `[1]` and `[0, 1]` allowed-tier domains are tested without expanding their legal domains.
- [ ] Final-period initial state enters the PSS peak; PSS compares the entire grid.
- [ ] Failed holdouts prevent optimization; failed real final validation prevents `bips2_trans`.
- [ ] All reports distinguish ROM predictions from real-HotSpot validated values.
