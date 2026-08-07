# Legacy Instruction-Window Scope Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept pre-schema canonical R1/module evidence with an absent instruction-window scope as `cpu0`, record that default in newly generated module models, and recover the saved R2 smoke without rerunning gem5.

**Architecture:** Add one shared normalization function that defaults only an absent key to `cpu0`; explicit values remain unchanged.  Use it when building future module models and when comparing legacy module architecture to canonical R1 during local attachment.  The existing immutable R1/R2 outputs remain inputs; the final smoke only attaches the proven local gem5 cache.

**Tech Stack:** Python 3 standard library, `unittest`, gem5 workflow JSON artifacts.

## Global Constraints

- An absent `instruction_window_scope` means exactly `cpu0`.
- An explicit scope is authoritative; an explicit incompatible scope must be rejected.
- Do not modify, rerun, move, or delete canonical R1 files.
- Do not delete or rerun a successful `gem5_r2/` result.
- The operational configuration remains `operational-exploratory-traffic-weighted` and non-formal.

---

### Task 1: Normalize legacy scope in module production and attachment validation

**Files:**
- Modify: `workflow/common.py`
- Modify: `workflow/floorplan/build_module_model.py`
- Modify: `workflow/r2/attachment_validation.py`
- Modify: `tests/test_balanced_experiment.py`
- Modify: `tests/test_workflow.py`

**Interfaces:**
- Produces: `instruction_window_scope(metadata: dict) -> object` in `workflow.common`.
- Consumes: R1 metadata and `modules.json["architecture"]` dictionaries.
- Preserves: explicit values, including invalid ones, so the existing R1 catalogue and attachment validation can reject them.

- [ ] **Step 1: Write the failing legacy attachment test and the explicit-mismatch regression**

Add these methods to `StrictR2ReuseTests` in `tests/test_balanced_experiment.py`.  Adapt the fixture identity helper to use `metadata.get("instruction_window_scope", "cpu0")`, because it represents the same historical R2 command compatibility rule as production.

```python
def test_local_attach_accepts_legacy_r1_and_module_scope_omission(self):
    """A pre-schema cpu0 run can attach its already measured local R2."""
    from workflow.r2.attach_result import attach

    self.metadata.pop("instruction_window_scope")
    write_json(self.r1 / "r1_metadata.json", self.metadata)
    modules_path = self.fixed / "modules.json"
    modules = read_json(modules_path)
    modules["architecture"].pop("instruction_window_scope")
    write_json(modules_path, modules)
    self._write_source_result()

    attached = attach(self.fixed)

    self.assertEqual(attached["ipc2"], 3.25)

def test_local_attach_rejects_explicit_scope_mismatch(self):
    """Compatibility for an absent key must not accept all-cores evidence."""
    from workflow.r2.attach_result import attach

    modules_path = self.fixed / "modules.json"
    modules = read_json(modules_path)
    modules["architecture"]["instruction_window_scope"] = "all-cores"
    write_json(modules_path, modules)

    with self.assertRaisesRegex(ValueError, "instruction_window_scope"):
        attach(self.fixed)
```

Run:

```bash
python -m unittest \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_accepts_legacy_r1_and_module_scope_omission \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_explicit_scope_mismatch -v
```

Expected: the legacy method fails with `target modules architecture instruction_window_scope differs from canonical R1`; the explicit mismatch method passes.

- [ ] **Step 2: Write the failing new-module-record test**

Extend the existing module-model parser test in `tests/test_workflow.py` to remove `instruction_window_scope` from its temporary R1 metadata before calling `build_model`, then assert both output locations are explicit:

```python
metadata.pop("instruction_window_scope")
write_json(r1_dir / "r1_metadata.json", metadata)
result = build_model(r1_dir, mcpat_json, cacti_json, output)
assert result["architecture"]["instruction_window_scope"] == "cpu0"
assert result["communication_profile"]["instruction_window_scope"] == "cpu0"
```

Run the single amended parser test.  Expected: it fails because the architecture key is absent and/or the communication profile says `not recorded`.

- [ ] **Step 3: Add the shared absent-key normalizer**

In `workflow/common.py`, add the narrow helper:

```python
def instruction_window_scope(metadata: dict) -> object:
    """Return an explicit scope, or the historical cpu0 default when absent."""
    return metadata["instruction_window_scope"] if "instruction_window_scope" in metadata else "cpu0"
```

Do not coerce, trim, or replace explicit values.  This keeps explicit invalid
values observable to existing validation.

- [ ] **Step 4: Use the normalizer in both evidence boundaries**

In `build_model`, create a copied metadata record before downstream use:

```python
metadata = dict(read_json(r1_dir / "r1_metadata.json"))
metadata["instruction_window_scope"] = instruction_window_scope(metadata)
```

Pass the normalized scope to `extract_communication_profile`; retain the
normalized copy in `result["architecture"]`.

In `validate_physical_coherence`, compare scopes through the helper:

```python
expected = instruction_window_scope(metadata)
actual = instruction_window_scope(architecture)
if actual != expected:
    reasons.append(
        "target modules architecture instruction_window_scope differs from canonical R1"
    )
```

Keep the existing direct comparisons for workload, cache sizes, core count,
and instruction counts unchanged.

- [ ] **Step 5: Verify the focused tests and the affected regression classes**

Run:

```bash
python -m unittest \
  tests.test_balanced_experiment.StrictR2ReuseTests \
  tests.test_workflow.ParserTests -v
python -m compileall -q workflow tests
git diff --check
```

Expected: all selected tests pass; compilation and whitespace checks exit zero.

- [ ] **Step 6: Commit the implementation**

```bash
git add workflow/common.py workflow/floorplan/build_module_model.py \
  workflow/r2/attachment_validation.py tests/test_balanced_experiment.py \
  tests/test_workflow.py
git commit -m "fix: support legacy cpu0 scope metadata"
```

### Task 2: Validate and recover the saved operational smoke

**Files:**
- Read: `runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/`
- Write: existing `performance.json`, `pipeline_summary.json`, and paired-status files only through `workflow.r2.run_paired_sweep`

**Interfaces:**
- Consumes: the merged Task-1 code and the existing fixed/CLIP layout-only roots.
- Produces: one certified smoke pair in `runs/operational_balanced50_traffic_weighted/paired_r2_status/`.

- [ ] **Step 1: Snapshot the existing successful gem5 result**

Before executing the paired runner, record the SHA-256 values of:

```bash
sha256sum \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/stats.txt \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/r2_result.json \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/status.json
```

- [ ] **Step 2: Run the fixed-first one-pair smoke without `--rerun`**

```bash
source scripts/env.sh
python -m workflow.r2.run_paired_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --fixed-root runs/operational_balanced50_traffic_weighted/fixed_bin \
  --clip-root runs/operational_balanced50_traffic_weighted/clip3d \
  --selection configs/experiments/balanced50_traffic_weighted.json \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json \
  --status-root runs/operational_balanced50_traffic_weighted/paired_r2_status \
  --jobs 1 --limit 1
```

Expected: the pair state is `success`; `physical_r2_runs` reflects the
certified fixed result plus a CLIP run only if the two complete override
vectors differ.  No command may contain `--rerun`.

- [ ] **Step 3: Prove that the saved fixed R2 was not rerun**

Run the same three `sha256sum` inputs from Step 1 after smoke, then inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
status = json.loads(Path(
    "runs/operational_balanced50_traffic_weighted/paired_r2_status/"
    "fft/l1d_16kB/l2_128kB/pair_status.json"
).read_text())
print(status["state"], status.get("fixed_complete"), status.get("clip_complete"))
PY
```

Expected: all pre/post SHA-256 values are identical and the pair state is
`success`.

- [ ] **Step 4: Commit only code and tests; leave experiment outputs ignored**

```bash
git status --short
git diff --check
```

Expected: no `runs/` or `results/` path is staged or committed.
