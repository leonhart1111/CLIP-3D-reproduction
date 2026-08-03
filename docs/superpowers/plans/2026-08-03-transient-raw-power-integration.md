# Transient Raw-Power Integration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore exact dynamic/leakage/total conservation for raw McPAT sub-block powers, remove unit-scale calibration plumbing from the optional transient path, and launch the isolated 10 ms MATMUL dual-layout validation.

**Architecture:** Keep the canonical R1, steady fixed-bin/CLIP-3D pilots, active formal R1, and non-zero `lambda_wire` experiment read-only. Reconstruct derived leakage and total fields from independently subtracted primitive McPAT fields, preserve raw subtraction residuals as diagnostics, and let the transient path consume `parse_mcpat_text` output directly. Run the real experiment only from the linked worktree and write only to the dedicated transient output root.

**Tech Stack:** Python 3 standard library, `unittest`, gem5 v23.1, McPAT 1.3, HotSpot detailed-3D, Git linked worktree.

## Global Constraints

- Modify only `/home/zyjiang/Agenticflow/CLIP/.worktrees/matmul-transient-validation` and the new transient output root.
- Do not modify, delete, overwrite, or rerun canonical R1 artifacts under `runs/architecture_sweep/r1/paper/`.
- Do not modify or restart `/home/zyjiang/Agenticflow/CLIP/runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803`.
- Do not modify the main checkout's tracked files or its unrelated dirty changes.
- Do not add power multipliers, fitted offsets, or tolerance relaxation; every emitted power record must satisfy `total_power_w = dynamic_power_w + leakage_power_w` within `1e-9 W`.
- Use the already approved canonical MATMUL `32kB/512kB` R1, existing fixed-bin and CLIP-3D steady pilots, operational raw-power config, and one shared 10 ms transient R1.
- Do not run R2 for transient validation.

---

### Task 1: Make raw McPAT subtraction conservative

**Files:**
- Modify: `tests/test_workflow.py`
- Modify: `workflow/mcpat/parse_mcpat.py`
- Modify: `workflow/transient/run_windowed_mcpat.py`
- Modify: `workflow/transient/run_transient_pipeline.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: raw McPAT `Runtime Dynamic`, `Subthreshold Leakage`, and `Gate Leakage` fields plus completed steady-pilot provenance.
- Produces: `subtract(base, children)` records whose derived leakage and total fields are algebraically conservative; transient window records with explicit raw-power provenance and no calibration fields.

- [ ] **Step 1: Add a failing subtraction regression test.**

Import `subtract` and construct one parent and child list where independently printed McPAT decimals make a primitive residual slightly negative. Assert that the returned dynamic, subthreshold, and gate fields are non-negative; leakage equals subthreshold plus gate exactly; total equals dynamic plus leakage exactly; and diagnostics retain the signed raw total residual plus the amount clipped for each primitive field.

```python
def test_subtraction_rebuilds_derived_power_after_rounding_clamp(self):
    base = {
        "area_mm2": 1.0,
        "dynamic_power_w": 1.0,
        "subthreshold_leakage_w": 0.2,
        "gate_leakage_w": 0.1,
        "leakage_power_w": 0.3,
        "total_power_w": 1.3,
    }
    child = {
        "area_mm2": 0.5,
        "dynamic_power_w": 1.0000023,
        "subthreshold_leakage_w": 0.15,
        "gate_leakage_w": 0.04,
        "leakage_power_w": 0.19,
        "total_power_w": 1.1900023,
    }
    remainder = subtract(base, [child])
    self.assertEqual(remainder["dynamic_power_w"], 0.0)
    self.assertEqual(
        remainder["leakage_power_w"],
        remainder["subthreshold_leakage_w"] + remainder["gate_leakage_w"],
    )
    self.assertEqual(
        remainder["total_power_w"],
        remainder["dynamic_power_w"] + remainder["leakage_power_w"],
    )
    self.assertAlmostEqual(
        remainder["subtraction_diagnostics"]["raw_residuals"]["total_power_w"],
        0.1099977,
    )
    self.assertAlmostEqual(
        remainder["subtraction_diagnostics"]["clipped_negative_magnitudes"]["dynamic_power_w"],
        0.0000023,
    )
```

- [ ] **Step 2: Run the focused test and verify RED.**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python \
  -m unittest -v \
  tests.test_workflow.ParserTests.test_subtraction_rebuilds_derived_power_after_rounding_clamp
```

Expected: failure because the current implementation independently subtracts and clips `leakage_power_w` and `total_power_w` and does not emit subtraction diagnostics.

- [ ] **Step 3: Implement the minimum conservative subtraction.**

For `area_mm2`, `dynamic_power_w`, `subthreshold_leakage_w`, and `gate_leakage_w`, compute the signed raw parent-minus-children residual and clip only negative values to zero. Rebuild:

```python
result["leakage_power_w"] = (
    result["subthreshold_leakage_w"] + result["gate_leakage_w"]
)
result["total_power_w"] = (
    result["dynamic_power_w"] + result["leakage_power_w"]
)
```

Store the unmodified signed residuals for all six original fields and negative-clamp magnitudes for the four primitive fields under `subtraction_diagnostics`. Do not scale or redistribute power.

- [ ] **Step 4: Remove calibration plumbing only from the transient branch.**

In `run_windowed_mcpat.py`, remove `resolve_power_calibration`, `apply_power_calibration`, scale fields, calibration provenance, and emitted `power_calibration` keys. Record `power_provenance` equal to raw McPAT dynamic and leakage definitions plus `postprocessing="none"`.

In `run_transient_pipeline.py`, stop resolving selected-config calibration. Reject any selected config that actually declares `mcpat.power_calibration`, continue accepting historical steady artifacts only when their calibration field is null or both scales equal exactly `1.0`, and report raw-power provenance without synthetic scale fields.

Update transient tests to assert the externally visible behavior: non-unit steady-pilot calibration is rejected before R1, a config declaring calibration is rejected before R1, and raw window records contain no `power_calibration` field.

- [ ] **Step 5: Run focused and complete tests and verify GREEN.**

Run:

```bash
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python \
  -m unittest -v tests.test_workflow.ParserTests \
  tests.test_transient.TransientDualLayoutTests
PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python \
  -m unittest discover -s tests -p 'test*.py' -v
```

Expected: all discovered tests pass; no test command reports zero discovered tests.

- [ ] **Step 6: Verify the real steady McPAT text.**

Parse the existing MATMUL steady `mcpat.out` from the fixed pilot and validate every module with the transient `validate_power_triplet` helper. Expected: zero violations at `1e-9 W`; the previously observed `core0_other`, `core1_other`, `core2_other`, and `core3_other` residuals disappear without changing any parent McPAT text.

- [ ] **Step 7: Commit the reviewed patch.**

```bash
git add tests/test_workflow.py tests/test_transient.py \
  workflow/mcpat/parse_mcpat.py \
  workflow/transient/run_windowed_mcpat.py \
  workflow/transient/run_transient_pipeline.py
git commit -m "fix: preserve raw transient power conservation"
```

### Task 2: Preflight and launch isolated 10 ms validation

**Files:**
- Create ignored link: `tools/src` pointing to `/home/zyjiang/Agenticflow/CLIP/tools/src`
- Create runtime output: `/home/zyjiang/Agenticflow/CLIP/runs/transient_validation/matmul_32kB_512kB_10ms_20260803/`

**Interfaces:**
- Consumes: the Task 1 branch, shared main-checkout tool binaries, canonical MATMUL R1, two completed steady pilots, and the operational raw-power config.
- Produces: a background dual-layout validation with `status.json`, `shared_r1/status.json`, `experiment.log`, and eventually fixed-bin/CLIP-3D comparison artifacts.

- [ ] **Step 1: Snapshot the non-zero-lambda experiment without changing it.**

Record its file inventory, sizes, mtimes, and `status.json` contents before launch. Report a stale `running` status as interrupted when no matching host process exists; do not rewrite the status file.

- [ ] **Step 2: Create the ignored tool-source link and run preflight.**

Verify `tools/src` is ignored, points to `/home/zyjiang/Agenticflow/CLIP/tools/src`, and that gem5, McPAT, CACTI, and HotSpot executables respond. Run the dual-layout validation preflight against the exact canonical R1, fixed pilot, CLIP-3D pilot, and operational config without starting R1.

- [ ] **Step 3: Launch one background job under an exclusive lock.**

Use `nohup` and `flock -n /tmp/clip-matmul-transient-10ms.lock` from the isolated worktree. Set `PYTHONPATH` to that worktree and use `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python`. Write combined output to `experiment.log`; request exactly the approved `--sample-ms 10` dual-layout command. Do not add `--rerun-transient-r1`.

- [ ] **Step 4: Verify the launch from host process and artifacts.**

Confirm one runner and one gem5 periodic-statistics child target the new transient root, `status.json` and `shared_r1/status.json` both report `running`, and gem5 consumes approximately one CPU core. Confirm the non-zero-lambda inventory remains unchanged except for filesystem access time, which is not part of the snapshot.

- [ ] **Step 5: Record the handoff.**

Report PID/PGID, exact output root and log, current stage, estimated 3.2-hour R1 duration, expected approximately 48 windows, and the commands for future read-only status checks. Do not claim the experiment has completed until final comparison artifacts exist and pass their validation checks.
