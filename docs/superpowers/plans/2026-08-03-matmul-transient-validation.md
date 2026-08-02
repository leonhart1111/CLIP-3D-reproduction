# MATMUL Transient Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audited 10 ms time-windowed power/temperature path and run one shared periodic-statistics MATMUL R1 through both fixed-bin and CLIP-3D layouts.

**Architecture:** Keep the canonical R1 and both completed steady pilots read-only. One shared transient R1 produces cumulative gem5 snapshots; one shared split/McPAT stage converts them to window powers; two layout-specific HotSpot stages consume the same powers; a final comparator writes deterministic JSON/CSV evidence.

**Tech Stack:** Python 3 standard library, gem5 v23.1, McPAT 1.3, HotSpot detailed-3D grid model, `unittest`, JSON and CSV.

## Global Constraints

- Do not modify, delete, overwrite, or rerun the canonical R1 under `runs/architecture_sweep/r1/paper/`.
- Do not overwrite the existing `runs/operational_raw_power_p1/pilot_direct_20260731/` fixed-bin or CLIP-3D outputs.
- Use `configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json`; label every new result operational, non-formal, and not paper-equivalent.
- Use one 10 ms periodic-statistics R1 for both layouts; do not run two layout-specific R1 jobs.
- Use raw local McPAT powers without dynamic or leakage calibration multipliers.
- Use each layout's existing steady `steady.txt` as its transient initial temperature.
- Do not run R2; the transient branch does not change physical latency.
- Preserve unrelated dirty-worktree changes and do not stage them in this work.

---

### Task 1: Add pure transient audit helpers

**Files:**
- Create: `workflow/transient/validation.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: window dictionaries containing `start_tick`, `end_tick`, `duration_s`, `modules`, and module `dynamic_power_w`, `leakage_power_w`, `total_power_w`.
- Produces: `validate_power_triplet(record: dict, context: str) -> None`, `validate_window_timeline(manifest: dict) -> dict`, and `summarize_power_windows(windows: list[dict]) -> dict`.

- [ ] **Step 1: Write failing tests for invalid and weighted power data.**

Add tests that assert negative, non-finite, or non-conserving power raises `ValueError`; discontinuous ticks raise `ValueError`; and two windows of 0.01 s at 10 W plus 0.005 s at 4 W produce a time-weighted mean of 8 W.

```python
def test_power_validation_and_duration_weighted_summary(self):
    with self.assertRaisesRegex(ValueError, "dynamic.*leakage.*total"):
        validate_power_triplet(
            {"dynamic_power_w": 2.0, "leakage_power_w": 1.0,
             "total_power_w": 4.0}, "bad"
        )
    windows = [
        {"start_tick": 0, "end_tick": 10, "duration_s": 0.01,
         "totals": {"dynamic_power_w": 8.0, "leakage_power_w": 2.0,
                    "total_power_w": 10.0}},
        {"start_tick": 10, "end_tick": 15, "duration_s": 0.005,
         "totals": {"dynamic_power_w": 3.0, "leakage_power_w": 1.0,
                    "total_power_w": 4.0}},
    ]
    summary = summarize_power_windows(windows)
    self.assertAlmostEqual(summary["total_power_w"]["weighted_mean"], 8.0)
```

- [ ] **Step 2: Run the tests and verify RED.**

Run: `python -m unittest -v tests.test_transient`

Expected: import failure because `workflow.transient.validation` does not exist.

- [ ] **Step 3: Implement strict validation and summaries.**

Use `math.isfinite`, reject values below `-1e-12`, and require
`math.isclose(total, dynamic + leakage, rel_tol=1e-9, abs_tol=1e-9)`.
`validate_window_timeline` must require positive durations, exact tick adjacency,
and return `window_count`, `total_duration_s`, `first_tick`, and `last_tick`.
`summarize_power_windows` must return min, max, weighted mean, peak window index,
and peak end time for all three power fields.

- [ ] **Step 4: Run the tests and verify GREEN.**

Run: `python -m unittest -v tests.test_transient`

Expected: all transient tests pass.

- [ ] **Step 5: Commit only the helper and its tests.**

```bash
git add workflow/transient/validation.py tests/test_transient.py
git commit -m "feat: audit transient power windows"
```

### Task 2: Enforce timeline and power conservation at existing boundaries

**Files:**
- Modify: `workflow/transient/stats_windows.py`
- Modify: `workflow/transient/generate_hotspot_trace.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: `validate_window_timeline`, `validate_power_triplet`, and `summarize_power_windows` from Task 1.
- Produces: timeline audit fields in `windows_manifest.json`, power statistics and maximum grid residual in `transient_trace_manifest.json`.

- [ ] **Step 1: Write failing tests for a tick gap and corrupted module totals.**

Extend the synthetic trace test so a window module with total power unequal to
dynamic plus leakage causes `materialize_trace` to raise. Add a direct manifest
test with `end_tick=10` followed by `start_tick=11` and require timeline rejection.

```python
with self.assertRaisesRegex(ValueError, "window timeline gap"):
    validate_window_timeline({"windows": [
        {"index": 0, "start_tick": 0, "end_tick": 10, "duration_s": 0.01},
        {"index": 1, "start_tick": 11, "end_tick": 20, "duration_s": 0.009},
    ]})
```

- [ ] **Step 2: Run the focused tests and verify RED.**

Run: `python -m unittest -v tests.test_transient.TransientStatisticsTests tests.test_transient.TransientTraceTests`

Expected: the new corrupted-total trace test does not raise before implementation.

- [ ] **Step 3: Wire the audit helpers into both stages.**

After `stats_windows.py` builds its manifest, call `validate_window_timeline` and
store its result under `timeline_audit`. In `materialize_trace`, validate every
module power triplet before gridding, calculate `power_summary`, and store
`maximum_grid_residual_w` as the maximum absolute residual already returned by
`grid_power`. Do not change the existing `.17g` trace serialization.

- [ ] **Step 4: Run the focused and full transient tests.**

Run: `python -m unittest -v tests.test_transient`

Expected: all tests pass with no warnings or errors.

- [ ] **Step 5: Commit the boundary checks.**

```bash
git add workflow/transient/stats_windows.py workflow/transient/generate_hotspot_trace.py tests/test_transient.py
git commit -m "fix: enforce transient timeline and power conservation"
```

### Task 3: Report trajectory peak, minimum, final temperature, and thermal lag inputs

**Files:**
- Modify: `workflow/transient/run_hotspot_transient.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: parsed temperature samples from `parse_ttrace` and the initial peak dictionary.
- Produces: `summarize_temperature_samples(samples: list[dict], initial_peak: dict) -> dict`; `transient_result.json` fields `trace_min_peak`, `trace_peak`, `final_peak`, and `overall_peak`.

- [ ] **Step 1: Write a failing summary test.**

```python
def test_temperature_summary_separates_initial_trace_and_final(self):
    samples = [
        {"index": 0, "time_s": 0.01, "peak_unit": "a",
         "tmax_c": 90.0, "tavg_c": 70.0},
        {"index": 1, "time_s": 0.02, "peak_unit": "b",
         "tmax_c": 95.0, "tavg_c": 72.0},
        {"index": 2, "time_s": 0.03, "peak_unit": "a",
         "tmax_c": 92.0, "tavg_c": 71.0},
    ]
    result = summarize_temperature_samples(
        samples, {"time_s": 0.0, "peak_unit": "initial", "tmax_c": 100.0}
    )
    self.assertEqual(result["trace_peak"]["time_s"], 0.02)
    self.assertEqual(result["final_peak"]["tmax_c"], 92.0)
    self.assertEqual(result["overall_peak"]["peak_unit"], "initial")
```

- [ ] **Step 2: Run the test and verify RED.**

Run: `python -m unittest -v tests.test_transient.TransientTraceTests.test_temperature_summary_separates_initial_trace_and_final`

Expected: import failure for `summarize_temperature_samples`.

- [ ] **Step 3: Implement and use the summary helper.**

Keep `tmax_c` as the all-time maximum for backward compatibility, but add explicit
trajectory-only fields. Record `trace_peak_minus_initial_c` and
`final_minus_initial_c`. Do not treat the initial steady temperature as a new
time-varying peak.

- [ ] **Step 4: Run all transient tests.**

Run: `python -m unittest -v tests.test_transient`

Expected: all tests pass.

- [ ] **Step 5: Commit the temperature semantics.**

```bash
git add workflow/transient/run_hotspot_transient.py tests/test_transient.py
git commit -m "fix: separate transient trajectory temperature metrics"
```

### Task 4: Share preprocessing across two layouts and produce comparison evidence

**Files:**
- Create: `workflow/transient/compare_layouts.py`
- Create: `workflow/transient/run_dual_layout_validation.py`
- Modify: `workflow/transient/run_transient_pipeline.py`
- Modify: `tests/test_transient.py`

**Interfaces:**
- Consumes: one canonical R1 directory, one periodic R1 directory, one config, two completed steady output directories, and the Task 1–3 summaries.
- Produces: `prepare_power_windows(source_r1_dir: Path, transient_r1_dir: Path, output_dir: Path, config: dict, sample_ms: float) -> dict`, `run_layout_thermal(source_r1_dir: Path, steady_output_dir: Path, output_dir: Path, config: dict, power_windows_path: Path, initial_temperature: str = "steady") -> dict`, `compare_layout_results(fixed: dict, clip3d: dict, output_dir: Path) -> dict`, plus the dual-layout CLI.

- [ ] **Step 1: Write failing tests for shared preprocessing and comparison.**

Use temporary synthetic fixed/CLIP summaries with the same `transient_r1`, sample
interval, and window count. Require `compare_layout_results` to reject mismatched
R1 paths or counts. For matching inputs, require the CLIP-minus-fixed trace peak,
steady peak, final peak, and peak-time lag fields and deterministic CSV headers.

```python
result = compare_layout_results(fixed, clip, output_dir)
self.assertAlmostEqual(result["temperature_c"]["trace_peak_clip_minus_fixed"], -1.5)
self.assertEqual(result["shared_input"]["window_count"], 48)
self.assertTrue((output_dir / "transient_comparison.csv").is_file())
```

- [ ] **Step 2: Run the new comparison tests and verify RED.**

Run: `python -m unittest -v tests.test_transient.TransientComparisonTests`

Expected: import failure because `compare_layouts.py` and the dual runner do not exist.

- [ ] **Step 3: Split the reusable stages without changing the single-layout CLI.**

In `run_transient_pipeline.py`, extract the exact public signatures
`prepare_power_windows(source_r1_dir: Path, transient_r1_dir: Path,
output_dir: Path, config: dict, sample_ms: float) -> dict` and
`run_layout_thermal(source_r1_dir: Path, steady_output_dir: Path,
output_dir: Path, config: dict, power_windows_path: Path,
initial_temperature: str = "steady") -> dict`.

The existing `run_transient_pipeline` must call these functions and retain its
public arguments and output path behavior.

- [ ] **Step 4: Implement the comparator.**

`compare_layout_results(fixed: dict, clip3d: dict, output_dir: Path) -> dict`
must validate the shared R1, sample interval, window count, actual duration, and
power trace identity. It writes `transient_comparison.json`,
`transient_comparison.csv`, and `power_temperature_timeseries.csv`. The report
must include `mode="operational transient validation"`, `non_formal=true`, model
limitations, temperature deltas, duration/padding, weighted power summaries, and
power-peak-to-temperature-peak lag.

- [ ] **Step 5: Implement the dual-layout runner and status recording.**

The CLI accepts explicit `--source-r1-dir`, `--fixed-steady-dir`,
`--clip3d-steady-dir`, `--output-root`, `--config`, and `--sample-ms`. It writes
`status.json` before the long R1 starts, reuses only a successful compatible
`shared_r1/`, prepares `shared/windows/` once, then runs fixed-bin and CLIP-3D
HotSpot branches. On success it writes `experiment_summary.json`; on failure it
updates `status.json` with exception type and message without deleting artifacts.

- [ ] **Step 6: Run focused and full tests.**

Run: `python -m unittest -v tests.test_transient`

Expected: all transient tests pass, including unchanged single-layout behavior.

Run: `python -m unittest discover -v`

Expected: the complete Python suite passes.

- [ ] **Step 7: Commit only clean/new implementation files.**

```bash
git add workflow/transient/compare_layouts.py workflow/transient/run_dual_layout_validation.py \
  workflow/transient/run_transient_pipeline.py tests/test_transient.py
git commit -m "feat: run shared-power dual-layout transient validation"
```

### Task 5: Document the real experiment and exact artifacts

**Files:**
- Modify: `docs/transient_thermal_zh.md`
- Modify: `workflow/README.md`

**Interfaces:**
- Consumes: the dual-layout CLI and output schema from Task 4.
- Produces: exact Chinese commands and interpretation rules for reproducing this experiment.

- [ ] **Step 1: Add documentation assertions to the existing source-level test.**

Add a test that requires the transient document to mention the operational config,
the canonical MATMUL 32kB/512kB R1, both steady layout paths, the dated output root,
and the terms `operational` and `non-formal`.

- [ ] **Step 2: Run the assertion and verify RED.**

Run: `python -m unittest -v tests.test_transient.TransientDocumentationTests`

Expected: failure because the dual-layout command is not yet documented.

- [ ] **Step 3: Document the exact launch and result reading commands.**

The launch command must be:

```bash
python -m workflow.transient.run_dual_layout_validation \
  --source-r1-dir runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB \
  --fixed-steady-dir runs/operational_raw_power_p1/pilot_direct_20260731/fixed-bin \
  --clip3d-steady-dir runs/operational_raw_power_p1/pilot_direct_20260731/clip3d \
  --output-root runs/transient_validation/matmul_32kB_512kB_10ms_20260803 \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json \
  --sample-ms 10
```

Explain that `shared_r1/` is a new periodic-statistics run, while the canonical R1
and completed pilots remain read-only. Document how to inspect `status.json`,
`experiment_summary.json`, and `comparison/transient_comparison.json`.

- [ ] **Step 4: Run documentation and full tests.**

Run: `python -m unittest -v tests.test_transient.TransientDocumentationTests`

Expected: pass.

Run: `python -m unittest discover -v`

Expected: complete suite passes.

- [ ] **Step 5: Commit the documentation and its test.**

```bash
git add docs/transient_thermal_zh.md workflow/README.md tests/test_transient.py
git commit -m "docs: explain dual-layout transient validation"
```

### Task 6: Launch, monitor, and validate the real MATMUL experiment

**Files:**
- Create at runtime: `runs/transient_validation/matmul_32kB_512kB_10ms_20260803/**`

**Interfaces:**
- Consumes: the Task 4 CLI, canonical R1, operational config, and two existing steady layouts.
- Produces: one successful shared transient R1, approximately 48 power windows, two HotSpot traces, and one comparison report.

- [ ] **Step 1: Record preflight hashes and tool health.**

Run: `source scripts/env.sh`

Run: `python scripts/check_tools.py`

Expected: gem5, McPAT, CACTI, and HotSpot all report usable binaries.

Run: `python -m unittest discover -v`

Expected: all tests pass immediately before launch.

- [ ] **Step 2: Launch the real experiment without tmux.**

Run the exact Task 5 command under `nohup` with stdout/stderr redirected to
`runs/transient_validation/matmul_32kB_512kB_10ms_20260803/experiment.log`.
Use one named `flock` file so a second copy cannot start accidentally.

- [ ] **Step 3: Verify that the long R1 is genuinely progressing.**

Inspect `status.json`, `shared_r1/status.json`, the process CPU percentage, and
the growth of `shared_r1/stats.txt` or logs. Expected state is `running`, one gem5
process near one full CPU core, and no second process using the same output root.

- [ ] **Step 4: Monitor to completion and run artifact validation.**

After the process exits, require `state=success`, verify approximately 48 windows,
check total duration near 0.477667 seconds, verify every power audit, both HotSpot
return codes, temperature sample counts, and all comparison consistency checks.

- [ ] **Step 5: Summarize evidence without formal claims.**

Report actual wall times, window count, power ranges, weighted mean versus steady
power, both steady and trace-only peak temperatures, peak times, thermal lag, and
CLIP-minus-fixed differences. State the fixed-temperature leakage, no-DVFS,
10 ms averaging, partial-window padding, and initial-steady-history limitations.
