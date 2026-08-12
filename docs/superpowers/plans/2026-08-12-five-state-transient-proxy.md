# Five-State Transient Thermal Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five-receiver closed-form thermal-inertia proxy the default transient optimization backend while retaining the existing POD-ROM as a configurable alternative.

**Architecture:** A new pure numerical module aggregates window powers into four core sources and one L2 source, evaluates the existing spatial kernel per receiver, solves each scalar periodic steady state in closed form, and uses the existing sustainable-frequency bracket search. A backend-neutral optimizer selects either this evaluator or the preserved POD-ROM evaluator. The shared pipeline keeps power-window preparation, final real-HotSpot validation, latency generation, and R2 unchanged.

**Tech Stack:** Python 3.12, standard library `math`/`dataclasses`, existing JSON workflow helpers, `unittest`, existing deterministic discrete-partition search.

## Global Constraints

- Do not change the paper-equivalent steady path.
- `transient_rom.backend` defaults to `five-state`; `pod-rom` retains existing behavior and calibration artifacts.
- The optimizer must call HotSpot zero times for either backend.
- Fixed-bin and selected CLIP-3D layouts still require existing real-HotSpot periodic validation.
- Five-state results remain `non_formal=true` and `paper_equivalent=false`.
- Provisional time constants must be labelled `parameter_status=provisional`, never presented as measured values.
- Preserve all existing POD-ROM files and public functions.

---

### Task 1: Five-state configuration and scalar model

**Files:**
- Create: `workflow/transient/five_state.py`
- Create: `tests/test_transient_five_state.py`

**Interfaces:**
- Produces: `FiveStateSettings`, `parse_five_state_settings(config: dict) -> FiveStateSettings`
- Produces: `evaluate_five_state(layout: dict, power_windows: dict, frequency_ghz: float, config: dict) -> dict`
- Produces: `find_five_state_sustainable_frequency(layout: dict, power_windows: dict, frequencies_ghz: list[float], config: dict) -> dict`

- [ ] **Step 1: Write failing settings tests**

Test omitted configuration defaults to `backend=five-state` with provisional `tau_core_s=tau_l2_s=0.166`, while invalid backend/status/non-positive time constants fail.

- [ ] **Step 2: Run the focused settings tests and verify expected import failure**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state.FiveStateSettingsTests -v`

- [ ] **Step 3: Implement immutable settings parsing**

Validate backend membership, finite positive time constants, spatial model, quadrature order, and parameter status without modifying the POD `ROMSettings` contract.

- [ ] **Step 4: Write failing numerical recurrence tests**

Use synthetic rectangular modules and two power windows to assert frequency-scaled dynamic/leakage power, `duration/s`, exact PSS closure, constant-equilibrium fixed point, moving hotspot receiver, and L2-position sensitivity.

- [ ] **Step 5: Run numerical tests and verify missing behavior fails**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state.FiveStateNumericsTests -v`

- [ ] **Step 6: Implement source aggregation, area quadrature, equilibrium vectors, PSS, and peak evaluation**

Keep five scalar receiver states. Aggregate core-labelled modules into core sources; use the sole L2 as the fifth source; include all modules in global/bottom power. Reject name, core, geometry, duration, and power-component inconsistencies.

- [ ] **Step 7: Write and pass sustainable-frequency search tests**

Delegate grid/bracket refinement to existing `find_sustainable_frequency`; assert returned evaluations record receiver peaks, inertia factors, equations, and zero HotSpot calls.

- [ ] **Step 8: Run the whole new test module**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state -v`

- [ ] **Step 9: Commit**

```bash
git add workflow/transient/five_state.py tests/test_transient_five_state.py
git commit -m "feat: add five-state transient thermal proxy"
```

### Task 2: Backend-selectable L2 optimizer

**Files:**
- Create: `workflow/transient/optimize_layout.py`
- Modify: `workflow/transient/rom/optimize_layout.py`
- Modify: `tests/test_transient_five_state.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**
- Produces: `optimize_transient_layout(modules_path: Path, output_dir: Path, config_path: Path, power_windows_path: Path, *, backend: str | None = None, package_dir: Path | None = None, hotspot: Path | None = None) -> dict`
- Preserves: `workflow.transient.rom.optimize_layout.optimize_transient_layout(...)` as the POD compatibility wrapper.

- [ ] **Step 1: Write failing five-state discrete-partition integration test**

Construct a synthetic module model/layout/windows/config, run a 5x5 P1 search, and assert the fixed baseline is included, output is backend-labelled, provisional, deterministic, and has zero optimizer HotSpot calls.

- [ ] **Step 2: Verify the integration test fails because the backend-neutral optimizer is absent**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state.FiveStateOptimizerTests -v`

- [ ] **Step 3: Extract common search mechanics into the backend-neutral optimizer**

Use one evaluator callback: five-state calls `find_five_state_sustainable_frequency`; POD-ROM calls the existing interpolation and ROM search. Preserve integer partition selection, wire aggregation, score, report field aliases, and POD identity gates.

- [ ] **Step 4: Keep the old POD import path as a delegating compatibility wrapper**

Ensure existing callers and tests importing `workflow.transient.rom.optimize_layout` continue to receive identical POD behavior.

- [ ] **Step 5: Run five-state and existing POD optimizer tests**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state tests.test_transient_rom.ROMOptimizerTests -v`

- [ ] **Step 6: Commit**

```bash
git add workflow/transient/optimize_layout.py workflow/transient/rom/optimize_layout.py tests/test_transient_five_state.py tests/test_transient_rom.py
git commit -m "feat: select transient optimizer backend"
```

### Task 3: Pipeline and CLI backend routing

**Files:**
- Modify: `workflow/transient/rom/run_pipeline.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Modify: `configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json`
- Modify: `tests/test_transient_five_state.py`
- Modify: `tests/test_transient_rom.py`

**Interfaces:**
- `run_transient_rom_pipeline(...)` reads `transient_rom.backend` and bypasses calibration/package requirements for `five-state`.
- CLI `--transient-thermal-backend {five-state,pod-rom}` optionally overrides config; omission uses config/default.

- [ ] **Step 1: Write failing routing tests**

Patch expensive stages and assert default/explicit five-state never calls calibration or package evidence, while explicit POD-ROM retains current calibration/package rules. Assert `--transient-rom-calibrate` and `--transient-rom-package-dir` are rejected for five-state.

- [ ] **Step 2: Run routing tests and verify expected failures**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state.FiveStatePipelineRoutingTests -v`

- [ ] **Step 3: Implement backend routing and backend-neutral report fields**

For five-state, prepare windows once, run backend-neutral optimization, then reuse the current paired real-HotSpot and optional R2 branches. Record `thermal_backend`, parameter provenance, no calibration calls, and no package path. Leave POD-ROM summaries backward-compatible.

- [ ] **Step 4: Add the CLI override and validate incompatible flags**

Pass the resolved backend into the pipeline without changing `--thermal-mode steady` behavior.

- [ ] **Step 5: Update the exploratory configuration**

Add `backend=five-state` and the explicit provisional `five_state` block. Retain all POD settings so switching `backend` back to `pod-rom` remains possible.

- [ ] **Step 6: Run routing, config, and main-flow tests**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state tests.test_transient_rom tests.test_discrete_partition_config tests.test_workflow -q`

- [ ] **Step 7: Commit**

```bash
git add workflow/transient/rom/run_pipeline.py workflow/run_lifting_pipeline.py configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json tests/test_transient_five_state.py tests/test_transient_rom.py
git commit -m "feat: default transient flow to five-state proxy"
```

### Task 4: Documentation and full verification

**Files:**
- Modify: `docs/transient_sustainable_frequency_zh.md`
- Modify: `workflow/README.md`
- Modify: `docs/superpowers/specs/2026-08-12-five-state-transient-proxy-design.md`

**Interfaces:**
- Documents exact commands, backend choice, equations, provisional parameters, outputs, and validation boundary.

- [ ] **Step 1: Update the Chinese mathematical document**

Add the five-state derivation, periodic-state formula, frequency-time scaling, and explain why it replaces POD-ROM only as the default optimizer backend rather than deleting it.

- [ ] **Step 2: Update workflow usage documentation**

Document default five-state execution, explicit POD fallback, output fields, and real-HotSpot validation requirement.

- [ ] **Step 3: Run source hygiene and focused regressions**

Run: `rg -n 'paper_equivalent.*true|HotSpot calls inside optimizer' workflow/transient tests/test_transient_five_state.py docs/transient_sustainable_frequency_zh.md`

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_five_state tests.test_transient tests.test_transient_rom tests.test_transient_rom_discrete_parity -q`

- [ ] **Step 4: Run the full unittest suite**

Run: `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest discover -s tests -q`

- [ ] **Step 5: Review diff and commit documentation**

```bash
git diff --check
git status --short
git add docs/transient_sustainable_frequency_zh.md workflow/README.md docs/superpowers/specs/2026-08-12-five-state-transient-proxy-design.md
git commit -m "docs: explain five-state transient workflow"
```
