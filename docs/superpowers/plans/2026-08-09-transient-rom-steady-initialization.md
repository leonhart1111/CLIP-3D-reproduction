# Transient ROM Matched Steady Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ROM holdout calibration and final real-HotSpot frequency validation start from a case-matched average-power steady state, while preserving ambient-start training traces and the strict `0.01 C` full-grid PSS gate.

**Architecture:** Add one focused steady-only HotSpot runner that consumes an already materialized transient case and binds its exact inputs and outputs. Use it before holdout and final transient calls, initialize the reduced model from the equivalent average-power continuous equilibrium, and expand fail-closed package/call-accounting contracts from 10 to 12 calibration calls.

**Tech Stack:** Python 3, `unittest`, NumPy/SciPy, HotSpot detailed-3D grid mode, JSON/SHA-256 provenance.

## Global Constraints

- Do not modify R1, McPAT extraction, HotSpot source code, the eight PRBS training anchors, steady-state paper reproduction, nonzero `lambda_wire`, or discrete wire-partition search.
- Keep training anchors ambient-start and require exactly eight training transient calls.
- Use exactly two holdout steady-initialization calls and two holdout transient calls; report 12 calibration HotSpot calls.
- Keep `pss_tolerance_c = 0.01`; do not convert PSS nonconvergence or tool/contract failures into thermal infeasibility.
- Run zero HotSpot calls inside the layout optimizer.
- Legacy 10-call ROM packages fail reuse validation and must be recalibrated.
- Work in an isolated worktree so in-flight experiments on `main` are unaffected.

---

### Task 1: Case-Matched Steady-Only HotSpot Runner

**Files:**
- Create: `workflow/transient/run_hotspot_steady.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: a case directory produced by `materialize_trace()` and a regular HotSpot binary path.
- Produces: `run_hotspot_steady(case_dir: Path, hotspot: Path = DEFAULT_HOTSPOT) -> dict`, `initialization.steady.txt`, `initialization.grid.steady.txt`, `hotspot_steady.log`, and `steady_initialization.json`.

- [ ] **Step 1: Write a failing behavior test for a steady-only invocation**

Add a test that creates real minimal case input files, substitutes only `subprocess.run`, writes both requested output files from the fake process, calls `run_hotspot_steady`, and asserts the returned command has no `-o`, uses `power_transient.ptrace`, and requests both initialization outputs. It must also assert that the JSON record contains SHA-256 identities for `hotspot.config`, `power_transient.ptrace`, `stack.lcf`, `materials.txt`, the HotSpot binary, and both outputs.

```python
result = run_hotspot_steady(case, hotspot)
self.assertNotIn("-o", result["command"])
self.assertEqual(result["return_code"], 0)
self.assertTrue((case / "initialization.steady.txt").is_file())
self.assertTrue((case / "initialization.grid.steady.txt").is_file())
self.assertEqual(
    read_json(case / "steady_initialization.json")["input_sha256"],
    result["input_sha256"],
)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_transient.HotSpotSteadyInitializationTests -v`

Expected: import failure because `workflow.transient.run_hotspot_steady` does not exist.

- [ ] **Step 3: Implement the minimal steady-only runner**

The command must be equivalent to:

```python
command = [
    str(hotspot.resolve()),
    "-c", "hotspot.config",
    "-p", "power_transient.ptrace",
    "-grid_layer_file", "stack.lcf",
    "-materials_file", "materials.txt",
    "-model_type", "grid",
    "-detailed_3D", "on",
    "-steady_file", "initialization.steady.txt",
    "-grid_steady_file", "initialization.grid.steady.txt",
]
```

Reject symlinks, missing inputs, nonzero return status, missing/empty outputs, and non-finite parsed steady temperatures. Persist command, elapsed time, return code, absolute artifact paths, and SHA-256 input/output identities.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.test_transient.HotSpotSteadyInitializationTests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/run_hotspot_steady.py tests/test_transient.py
git commit -m "feat: add matched HotSpot steady initializer"
```

### Task 2: Holdout Preconditioning and 12-Call Package Contract

**Files:**
- Modify: `workflow/transient/rom/materialize_calibration.py`
- Modify: `workflow/transient/rom/contracts.py`
- Modify: `workflow/transient/rom/calibrate_rom.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**
- Consumes: `run_hotspot_steady()` from Task 1.
- Produces: training cases with ambient initial state; holdout cases with matched steady initial state and separate initialization/transient evidence; strict `8+2+2` reusable-package validation.

- [ ] **Step 1: Change the calibration test to specify the new contract and verify RED**

Patch the external steady and transient runners separately. Assert eight ambient transient calls for training, two steady calls followed by two steady-start transient calls for holdouts, and these literal counters:

```python
self.assertEqual(report["training_hotspot_calls"], 8)
self.assertEqual(report["holdout_initialization_hotspot_calls"], 2)
self.assertEqual(report["holdout_transient_hotspot_calls"], 2)
self.assertEqual(report["calibration_hotspot_calls"], 12)
self.assertTrue(all(
    case["initial_temperature"] == "ambient"
    for case in report["training_cases"]
))
self.assertTrue(all(
    case["initial_temperature"] == "average_power_steady"
    for case in report["holdout_cases"]
))
```

Add mutations proving that a legacy `holdout_hotspot_calls=2` package without initialization evidence and a holdout initialization hash mismatch are rejected.

- [ ] **Step 2: Run the focused calibration/package tests and verify RED**

Run: `python -m unittest tests.test_transient_rom.ROMCalibrationCaseTests tests.test_transient_rom.ROMContractTests -v`

Expected: failures on the new counters, steady initialization, and artifact contract.

- [ ] **Step 3: Integrate steady preconditioning only for holdouts**

Split `_materialize_case` behavior by `kind`:

```python
if kind == "training":
    thermal = run_hotspot_transient(
        case_dir, hotspot=hotspot, initial_temperature="ambient"
    )
else:
    initialization = run_hotspot_steady(case_dir, hotspot=hotspot)
    thermal = run_hotspot_transient(
        case_dir,
        hotspot=hotspot,
        initial_temperature="steady",
        steady_source=case_dir / "initialization.steady.txt",
    )
```

Record and hash the steady initialization JSON, block steady file, and grid steady file in holdout artifacts. Preserve the existing full-grid PSS calculation and exception when it exceeds `settings.pss_tolerance_c`.

- [ ] **Step 4: Upgrade package validation and reporting**

Require manifest/case schema evidence for eight training transient, two holdout initialization, and two holdout transient calls. For training, require ambient state and no initialization artifacts. For holdouts, require matched-steady state, complete initialization hashes, and a transient command containing `-init_file`. Update CLI summaries to say `8 training + 2 initialization + 2 holdout transient = 12`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_transient_rom.ROMCalibrationCaseTests tests.test_transient_rom.ROMContractTests -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add workflow/transient/rom/materialize_calibration.py workflow/transient/rom/contracts.py workflow/transient/rom/calibrate_rom.py tests/test_transient_rom.py
git commit -m "fix: precondition transient ROM holdouts"
```

### Task 3: Reduced-Order Average-Power Equilibrium Initialization

**Files:**
- Modify: `workflow/transient/rom/layout_rom.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**
- Consumes: `StateSpaceModel`, the interpolated L2 input column, and `_period_inputs()`.
- Produces: `_average_power_steady_state(model, b_l2, period_inputs, max_condition_number) -> tuple[numpy.ndarray, dict]`; `evaluate_layout_rom()` reports initialization diagnostics.

- [ ] **Step 1: Add failing first-order equilibrium tests**

For `dx/dt = -2x + 3u`, constant `u=4`, assert the state is `6`. For unequal durations, assert the duration-weighted input is used. Add tests that singular/over-limit condition number and excessive normalized residual raise explicit errors instead of returning an ambient state.

```python
state, audit = _average_power_steady_state(
    model,
    model.b_l2_anchors["a0"],
    [(0.25, numpy.array([4.0])), (0.75, numpy.array([4.0]))],
    max_condition_number=1.0e10,
)
self.assertAlmostEqual(state[0], 6.0, places=12)
self.assertLessEqual(audit["normalized_residual"], 1.0e-10)
```

- [ ] **Step 2: Run the focused ROM tests and verify RED**

Run: `python -m unittest tests.test_transient_rom.ROMEvaluationTests -v`

Expected: import or assertion failure because average-power initialization is absent.

- [ ] **Step 3: Implement the checked equilibrium solve**

Construct `B(l)` by concatenating `model.b_fixed` and the interpolated L2 column. Compute the duration-weighted mean power, validate all dimensions and finite values, measure `numpy.linalg.cond(A)`, solve `A x_ss = -B(l) u_bar` with `numpy.linalg.solve`, and compute:

```python
residual = A @ state + B @ mean_power
normalized_residual = numpy.linalg.norm(residual) / max(
    1.0,
    numpy.linalg.norm(A) * numpy.linalg.norm(state)
    + numpy.linalg.norm(B @ mean_power),
)
```

Reject condition numbers over `settings.max_condition_number` and normalized residuals over `1e-10`. Initialize `evaluate_layout_rom()` from this state, retain all 20 periodic repeats and the existing full-grid PSS gate, and emit the audit under `thermal_initialization`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_transient_rom.ROMEvaluationTests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflow/transient/rom/layout_rom.py tests/test_transient_rom.py
git commit -m "fix: initialize ROM at average-power equilibrium"
```

### Task 4: Final Frequency Validation and Call Accounting

**Files:**
- Modify: `workflow/transient/verify_sustainable_frequency.py`
- Modify: `workflow/transient/rom/run_pipeline.py`
- Modify: `tests/test_transient.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**
- Consumes: `run_hotspot_steady()` and `run_hotspot_transient()`.
- Produces: paired steady/transient evaluation evidence for every real-HotSpot frequency and separate initialization/transient/total counters in ROM summaries.

- [ ] **Step 1: Add failing final-search behavior tests**

Patch both external HotSpot runners. Assert every materialized frequency case runs the steady initializer first, then transient with that case's `initialization.steady.txt`. Assert each evaluation records `initialization_hotspot_calls=1`, `transient_hotspot_calls=1`, and `hotspot_calls=2`. Add a failure test proving a steady initialization error is reported as `steady_initialization_tool_failure`, not PSS nonconvergence or thermal infeasibility.

- [ ] **Step 2: Add failing pipeline-accounting tests**

For one final frequency directory, assert:

```python
self.assertEqual(result["final_initialization_hotspot_calls"], 1)
self.assertEqual(result["final_transient_hotspot_calls"], 1)
self.assertEqual(result["final_validation_hotspot_calls"], 2)
```

For a reused package, assert historical calibration counters stay `8`, `2`, `2`, and `12`, while every calibration `*_this_invocation` counter is zero.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m unittest tests.test_transient.TransientTraceTests tests.test_transient_rom.ROMPipelineTests -v`

Expected: failures because final validation remains ambient-start and counts one call per frequency.

- [ ] **Step 4: Integrate paired calls and explicit failure classes**

In each frequency evaluation, run `run_hotspot_steady(case, hotspot)` then:

```python
thermal = run_hotspot_transient(
    case,
    hotspot=hotspot,
    initial_temperature="steady",
    steady_source=case / "initialization.steady.txt",
)
```

Record both evidence objects. Wrap steady-initialization, transient-tool, contract, and PSS failures with distinct categories at the boundary where they originate. Count actual completed/materialized steady and transient artifacts, not merely the number of frequency directories.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_transient.TransientTraceTests tests.test_transient_rom.ROMPipelineTests -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add workflow/transient/verify_sustainable_frequency.py workflow/transient/rom/run_pipeline.py tests/test_transient.py tests/test_transient_rom.py
git commit -m "fix: precondition final transient frequency search"
```

### Task 5: Documentation, Regression Suite, and Real MATMUL Acceptance

**Files:**
- Modify: `docs/transient_rom_usage_zh.md`
- Modify: `docs/transient_sustainable_frequency_zh.md`

**Interfaces:**
- Consumes: completed implementation and the existing MATMUL R1/steady outputs.
- Produces: accurate user documentation, passing regression evidence, and a fresh real MATMUL result directory.

- [ ] **Step 1: Update documentation**

Document the 12-call calibration budget, per-frequency two-call final validation, average-power reduced equilibrium, unchanged `0.01 C` PSS gate, legacy package invalidation, output artifacts, and failure categories. Remove statements that calibration always uses 10 calls or that holdouts/final validation start from ambient.

- [ ] **Step 2: Run static and focused regression checks**

Run:

```bash
git diff --check
python -m compileall -q workflow/transient tests/test_transient.py tests/test_transient_rom.py
python -m unittest tests.test_transient tests.test_transient_rom -v
```

Expected: exit status zero and no failing tests.

- [ ] **Step 3: Run the broader workflow regression suite**

Run:

```bash
python -m unittest tests.test_workflow tests.test_balanced_experiment -v
```

Expected: exit status zero and no failing tests.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/transient_rom_usage_zh.md docs/transient_sustainable_frequency_zh.md
git commit -m "docs: explain matched transient ROM initialization"
```

- [ ] **Step 5: Run a fresh real MATMUL acceptance experiment**

Use the existing nonzero-lambda, communication-weighted, discrete-partition configuration and periodic R1, but create a new twelve-call package. Run without R2 first so thermal correctness is established independently:

```bash
source scripts/env.sh
R1=/home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
PERIODIC_R1=/home/zyjiang/Agenticflow/CLIP/runs/transient_validation/matmul_32kB_512kB_lambda0020119_2ms_precision6_20260806_030743/transient/shared_r1
CFG=configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
THERMAL_OUT=/home/zyjiang/Agenticflow/CLIP/runs/transient_rom/matmul_steady_init_thermal_20260809

python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$THERMAL_OUT" \
  --config "$CFG" \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir "$PERIODIC_R1" \
  --transient-rom-calibrate
```

Acceptance requires both fixed-bin and CLIP-3D branches to contain finite sustainable frequencies, full-grid PSS convergence at `0.01 C`, and a summary state of `validated` rather than a mere file-presence check.

- [ ] **Step 6: If thermal acceptance passes, run/attach R2 and verify final outputs**

Reuse the accepted package in a second new output root and request R2:

```bash
PACKAGE="$THERMAL_OUT/transient_rom/rom_package"
R2_OUT=/home/zyjiang/Agenticflow/CLIP/runs/transient_rom/matmul_steady_init_r2_20260809

python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$R2_OUT" \
  --config "$CFG" \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir "$PERIODIC_R1" \
  --transient-rom-package-dir "$PACKAGE" \
  --run-r2
```

Acceptance requires finite `ipc2_trans` and `bips2_trans` for both branches and a fail-closed paired comparison. Record exact commands, wall time, output root, call counts, frequencies, temperatures, IPC2, and BIPS2_trans in the handoff.
