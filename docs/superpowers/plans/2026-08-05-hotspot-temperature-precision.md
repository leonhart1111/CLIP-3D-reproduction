# HotSpot Temperature Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit six-decimal HotSpot temperatures throughout the steady/transient path and make every live transient limitation report the run's actual sampling interval.

**Architecture:** Keep the thermal and power models unchanged. Centralize sampling-resolution wording in one pure Python helper, format human-readable temperature outputs at their file/CLI boundaries, and carry the third-party HotSpot change as a tracked patch that is also applied to the ignored local checkout.

**Tech Stack:** Python 3.12 standard library and `unittest`, C/Make for HotSpot, unified diff/`patch`, Markdown documentation.

## Global Constraints

- HotSpot machine-readable temperature files use exactly six digits after the decimal point.
- JSON temperatures remain numeric values; only textual outputs receive fixed-width formatting.
- A run requested with `sample_ms=2.0` reports `2 ms`, never a stale `10 ms` limitation.
- The CLI default remains 10 ms for backward compatibility.
- Do not change gem5 R1, windowed McPAT, layout optimization, R2, HotSpot equations, or existing result directories.
- Historical files under `docs/superpowers/specs/` and `docs/superpowers/plans/` retain their original 10 ms experiment wording.
- Modify files in the clean `feature/matmul-transient-validation` worktree; do not absorb unrelated dirty changes from `main`.

## File Structure

- `workflow/transient/validation.py`: owns the pure sampling-limit wording helper.
- `workflow/transient/run_transient_pipeline.py`: uses the helper in single-layout summaries and prints six-decimal temperatures.
- `workflow/transient/compare_layouts.py`: uses the helper in comparison summaries.
- `workflow/transient/run_dual_layout_validation.py`: uses the helper in experiment summaries and prints six-decimal layout deltas.
- `workflow/transient/run_hotspot_transient.py`: writes six-decimal Celsius CSV values and six-decimal CLI output.
- `tests/test_transient.py`: covers the helper, 2 ms summary propagation, and CSV presentation.
- `patches/hotspot/0001-six-decimal-temperature-output.patch`: reproducible third-party source change.
- `tools/src/hotspot/{hotspot.c,temperature_grid.c,temperature_block.c}`: ignored local checkout to which the tracked patch is applied.
- `docs/DOWNLOAD_TOOLS.md`: documents idempotent patch detection, application, and rebuild.
- `docs/transient_thermal_zh.md`: describes a configurable interval and uses the current 2 ms example.
- `workflow/README.md`: removes the obsolete name “10 ms transient branch.”

---

### Task 1: Parameterize live sampling reports and format Python temperature output

**Files:**
- Modify: `tests/test_transient.py`
- Modify: `workflow/transient/validation.py`
- Modify: `workflow/transient/run_transient_pipeline.py`
- Modify: `workflow/transient/compare_layouts.py`
- Modify: `workflow/transient/run_dual_layout_validation.py`
- Modify: `workflow/transient/run_hotspot_transient.py`
- Modify: `workflow/run_lifting_pipeline.py`

**Interfaces:**
- Produces: `sampling_resolution_limitation(sample_ms: float) -> str` in `workflow.transient.validation`.
- Consumes: existing `sample_ms`, `trace["sample_interval_s"]`, and temperature sample dictionaries.
- Preserves: existing summary keys and JSON numeric types.

- [ ] **Step 1: Write failing helper and CSV tests**

Add a direct helper contract to `tests/test_transient.py`:

```python
def test_sampling_limitation_uses_actual_interval(self):
    from workflow.transient.validation import sampling_resolution_limitation

    message = sampling_resolution_limitation(2.0)
    self.assertIn("2 ms averaging", message)
    self.assertNotIn("10 ms averaging", message)
```

Extend the mocked `run_hotspot_transient` test so the generated trace contains
six-decimal input tokens and assert the CSV text contains fixed-width Celsius
values:

```python
csv_text = (root / "transient_summary.csv").read_text(encoding="utf-8")
self.assertIn("27.850000", csv_text)
self.assertIn("28.850000", csv_text)
```

Extend `TransientComparisonTests.summary(...)` with a
`sample_ms: float = 10.0` argument. Derive both window durations, temperature
sample times, `actual_gem5_duration_s`, `hotspot_trace_duration_s`, and the
power-window manifest's nominal interval from that value. Change the
comparison limitation test to call both summaries with `sample_ms=2.0`, then
assert `"2 ms averaging"` is present and `"10 ms averaging"` is absent.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m unittest \
  tests.test_transient.TransientComparisonTests.test_comparison_carries_complete_audit_classification_and_limitations \
  tests.test_transient.TransientTraceTests.test_sampling_limitation_uses_actual_interval \
  tests.test_transient.TransientTraceTests.test_thermal_result_has_standard_classification_and_acceptance_evidence -v
```

Expected: failure because `sampling_resolution_limitation` does not exist,
the comparison still says 10 ms, and CSV temperatures are not fixed to six
decimal places.

- [ ] **Step 3: Implement the pure limitation helper**

Add to `workflow/transient/validation.py`:

```python
def sampling_resolution_limitation(sample_ms: float) -> str:
    if isinstance(sample_ms, bool) or not isinstance(sample_ms, Real):
        raise ValueError("sample_ms must be a finite positive number")
    value = float(sample_ms)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sample_ms must be a finite positive number")
    return (
        f"{value:g} ms averaging cannot observe sub-window microsecond "
        "power peaks."
    )
```

Import and use it at all four live summary construction sites:

```python
sampling_resolution_limitation(sample_ms)
```

For `run_layout_thermal`, derive the argument from the materialized trace:

```python
sample_ms = float(trace["sample_interval_s"]) * 1000.0
```

- [ ] **Step 4: Format CSV and CLI temperature presentation**

In `run_hotspot_transient.py`, retain numeric in-memory samples but render the
CSV temperature fields explicitly:

```python
writer.writerow({
    "index": sample["index"],
    "time_s": sample["time_s"],
    "peak_unit": sample["peak_unit"],
    "tmax_c": f"{sample['tmax_c']:.6f}",
    "tavg_c": f"{sample['tavg_c']:.6f}",
})
```

Change transient temperature and layout-delta CLI format specifiers from
`.3f` to `.6f` in `run_hotspot_transient.py`, `run_transient_pipeline.py`,
`run_dual_layout_validation.py`, and the transient-specific output in
`workflow/run_lifting_pipeline.py`. Leave unrelated steady pipeline display
formatting unchanged.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same `python -m unittest ... -v` command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 6: Search for stale live report text**

Run:

```bash
rg -n '10 ms averaging|Tmax=.*\.3f|trace_peak_clip_minus_fixed.*\.3f' \
  workflow/transient workflow/run_lifting_pipeline.py
```

Expected: no live hard-coded limitation or three-decimal transient temperature
display remains. Definitions with a default value of `10.0` are allowed.

- [ ] **Step 7: Commit Task 1**

```bash
git add tests/test_transient.py workflow/transient/validation.py \
  workflow/transient/run_transient_pipeline.py \
  workflow/transient/compare_layouts.py \
  workflow/transient/run_dual_layout_validation.py \
  workflow/transient/run_hotspot_transient.py \
  workflow/run_lifting_pipeline.py
git commit -m "fix: preserve transient report precision"
```

### Task 2: Version and apply the six-decimal HotSpot output patch

**Files:**
- Modify: `tests/test_transient.py`
- Create: `patches/hotspot/0001-six-decimal-temperature-output.patch`
- Modify locally/ignored: `tools/src/hotspot/hotspot.c`
- Modify locally/ignored: `tools/src/hotspot/temperature_grid.c`
- Modify locally/ignored: `tools/src/hotspot/temperature_block.c`

**Interfaces:**
- Consumes: the upstream HotSpot source layout already recorded in `manifests/source_archives.sha256`.
- Produces: a tracked `-p1` unified patch and a rebuilt `tools/src/hotspot/hotspot` executable.
- Preserves: HotSpot equations, solver tolerances, and diagnostic-only formatting.

- [ ] **Step 1: Add a failing tracked-patch contract test**

Add a test that requires the patch artifact and its intended format changes:

```python
def test_hotspot_patch_tracks_six_decimal_temperature_outputs(self):
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "patches/hotspot/0001-six-decimal-temperature-output.patch"
    )
    text = patch_path.read_text(encoding="utf-8")
    self.assertNotIn('+    fprintf(fp, "%.2f', text)
    self.assertGreaterEqual(text.count("%.6f"), 11)
    self.assertIn("hotspot.c", text)
    self.assertIn("temperature_grid.c", text)
    self.assertIn("temperature_block.c", text)
```

If `tools/src/hotspot` exists, also assert the local machine-readable source
format strings have no `%.2f` after patch application; skip only that local
source assertion when third-party tools have not been downloaded.

- [ ] **Step 2: Run the patch contract and verify RED**

Run:

```bash
python -m unittest \
  tests.test_transient.HotSpotPrecisionPatchTests.test_hotspot_patch_tracks_six_decimal_temperature_outputs -v
```

Expected: failure with `FileNotFoundError` because the tracked patch does not
yet exist.

- [ ] **Step 3: Create the tracked unified patch**

Create `patches/hotspot/0001-six-decimal-temperature-output.patch` with
`a/` and `b/` paths for these exact replacements:

```text
hotspot.c:              write_vals %.2f -> %.6f (2 format strings)
temperature_grid.c:    temperature dumps %.2f -> %.6f (4 format strings)
temperature_block.c:   named temperature dumps %.2f -> %.6f (5 format strings)
```

Do not change ambient, geometry, percentage, or diagnostic formatting.

- [ ] **Step 4: Apply the patch to the ignored local checkout**

Detect an already-applied patch first:

```bash
patch -d tools/src/hotspot -p1 --dry-run -R \
  < patches/hotspot/0001-six-decimal-temperature-output.patch
```

If that command fails, apply once:

```bash
patch -d tools/src/hotspot -p1 \
  < patches/hotspot/0001-six-decimal-temperature-output.patch
```

Then verify the eleven machine-readable format strings are six decimal:

```bash
rg -n 'fprintf\(.*%\.6f' \
  tools/src/hotspot/hotspot.c \
  tools/src/hotspot/temperature_grid.c \
  tools/src/hotspot/temperature_block.c
```

- [ ] **Step 5: Run the patch contract and verify GREEN**

Run the same unittest command from Step 2.

Expected: pass with the tracked patch present and the local checkout patched.

- [ ] **Step 6: Rebuild the real HotSpot binary**

Run:

```bash
make -C tools/src/hotspot hotspot
```

Expected: `hotspot.c`, `temperature_grid.c`, and `temperature_block.c` compile
and link without errors.

- [ ] **Step 7: Run a one-row real HotSpot precision smoke test**

Create a temporary case by copying the configuration/geometry files from the
completed 2 ms fixed-bin case, and reduce its power trace to header plus one
row. Invoke the rebuilt binary with the same detailed-3D arguments recorded in
`transient_result.json`. Verify both the second row of `transient.ttrace` and
the values in `average_power.steady.txt` match a six-decimal token pattern:

```bash
awk 'NR==2 {for (i=1;i<=NF;i++) if ($i !~ /^-?[0-9]+\.[0-9]{6}$/) exit 1}' \
  transient.ttrace
awk 'NF>=2 && $NF !~ /^-?[0-9]+\.[0-9]{6}$/ {exit 1}' \
  average_power.steady.txt
```

Expected: both commands exit zero.

- [ ] **Step 8: Commit Task 2**

Only the tracked patch and test are committed; ignored vendor sources and the
built binary remain local products:

```bash
git add tests/test_transient.py \
  patches/hotspot/0001-six-decimal-temperature-output.patch
git commit -m "fix: patch HotSpot temperature output precision"
```

### Task 3: Update live documentation and run full verification

**Files:**
- Modify: `docs/DOWNLOAD_TOOLS.md`
- Modify: `docs/transient_thermal_zh.md`
- Modify: `workflow/README.md`

**Interfaces:**
- Documents: tracked patch application, unchanged 10 ms default, and current 2 ms usage.
- Preserves: historical 10 ms provenance documents under `docs/superpowers/`.

- [ ] **Step 1: Add patch/rebuild instructions to the tool guide**

Document the reverse dry-run detection, one-time `patch -p1` application, and
`make -C tools/src/hotspot hotspot` commands from Task 2. State that the patch
changes only textual temperature precision and must be applied after a fresh
HotSpot download.

- [ ] **Step 2: Generalize the transient user guide**

Apply these live-document changes:

```text
"10 ms瞬态热仿真旁路" -> "可配置时间窗瞬态热仿真旁路"
"10 ms累计统计" -> "按 --sample-ms 设置的周期累计统计"
current command examples -> --sample-ms 2 / --transient-sample-ms 2
"每10 ms增量统计" -> "按所选采样间隔生成增量统计"
```

Retain an explicit subsection explaining that 10 ms was the earlier comparison
and 2 ms is the current higher-resolution experiment; do not rewrite the
historical plan/spec files.

- [ ] **Step 3: Update the workflow index wording**

Change `workflow/README.md` from “10 ms 瞬态旁路” to “可配置采样间隔的瞬态旁路.”

- [ ] **Step 4: Run documentation and source scans**

Run:

```bash
rg -n '10 ms averaging' workflow tests
rg -n '^# 10 ms瞬态|每10 ms增量统计|专用瞬态R1（10 ms' \
  docs/transient_thermal_zh.md workflow/README.md
git diff --check
```

Expected: the first two searches return no matches; `git diff --check` returns
no whitespace errors. Historical `docs/superpowers/**` references are excluded.

- [ ] **Step 5: Run all Python regressions**

Run:

```bash
python -m unittest tests.test_transient -v
python -m unittest tests.test_workflow -v
```

Expected: all tests pass; only pre-existing environment-dependent skips are
allowed.

- [ ] **Step 6: Verify repository and local-tool state**

Run:

```bash
git status --short
git diff --check
stat -c '%n %s %y' tools/src/hotspot/hotspot
```

Expected: only Task 3 documentation files remain uncommitted, the worktree has
no whitespace errors, and the HotSpot binary timestamp reflects the rebuild.

- [ ] **Step 7: Commit Task 3**

```bash
git add docs/DOWNLOAD_TOOLS.md docs/transient_thermal_zh.md workflow/README.md
git commit -m "docs: parameterize transient sampling guidance"
```

- [ ] **Step 8: Final verification after commits**

Run:

```bash
python -m unittest tests.test_transient tests.test_workflow -v
git status --short
git log -4 --oneline
```

Expected: the complete suite passes, the feature worktree is clean, and the
design, implementation tasks, and documentation commits are visible.
