# Legacy Instruction-Window Scope Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept pre-schema canonical R1/module evidence with an absent instruction-window scope as `cpu0`, record that default in newly generated module models, and recover the saved R2 smoke without rerunning gem5.

**Architecture:** Add one shared normalization function that defaults only an absent key to `cpu0`; explicit values remain unchanged.  Use it when building future module models and when comparing legacy module architecture to canonical R1 during local attachment.  The existing immutable R1/R2 outputs remain inputs; the final fixed-only smoke only attaches the proven local gem5 cache.

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

## Post-merge operational validation (not an SDD task)

Run this only after Task 1 has passed its task and whole-branch reviews and
has been merged into `main`.  The command intentionally writes the existing
main-worktree experiment outputs, so it cannot execute from the isolated
implementation worktree.

**Files:**
- Read: `runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/`
- Write: `runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/performance.json` and `pipeline_summary.json` only through `workflow.r2.attach_result`

**Interfaces:**
- Consumes: the merged Task-1 code and the existing successful fixed-bin R2 result.
- Produces: refreshed, locally certified fixed-point attachment-derived outputs; it does not produce a paired status.

1. **Snapshot the existing successful gem5 result**

Before executing the attachment command, record the SHA-256 values of the
immutable R2 evidence in a temporary guard file:

```bash
sha256sum \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/stats.txt \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/r2_result.json \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/status.json \
  > /tmp/legacy-scope-fixed-r2-before.sha256
```

2. **Run the fixed-only local attachment smoke**

```bash
source scripts/env.sh
python -m workflow.r2.attach_result --point-dir \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB
```

Expected: the command validates the saved fixed-bin R2 and publishes only its
attachment-derived `performance.json` and `pipeline_summary.json` outputs.
It does not run gem5, touch CLIP, create a paired-status file, or claim a
complete pair.

3. **Prove that the saved fixed R2 was not rerun**

Regenerate the same hashes and require an exact match with Step 1 before
inspecting the fixed attachment fields:

```bash
sha256sum \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/stats.txt \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/r2_result.json \
  runs/operational_balanced50_traffic_weighted/fixed_bin/fft/l1d_16kB/l2_128kB/gem5_r2/status.json \
  | diff -u /tmp/legacy-scope-fixed-r2-before.sha256 -
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path(
    "runs/operational_balanced50_traffic_weighted/fixed_bin/"
    "fft/l1d_16kB/l2_128kB/pipeline_summary.json"
).read_text())
print(summary["ipc2"], summary["bips2"], summary["r2_source"])
PY
```

Expected: `diff` exits zero, so all pre/post SHA-256 values are identical.  The printed `ipc2`,
`bips2`, and `r2_source` are the fixed point's locally verifiable attachment
fields; `r2_source` names its existing `gem5_r2/r2_result.json`.

4. **Defer the full paired sweep to its authorized workflow**

This smoke is intentionally not a paired sweep.  The later full Balanced-50
run follows `50+K`: if a fixed and CLIP point have different complete latency
override vectors, CLIP must run an independent R2 and cannot be recorded as a
reuse of fixed-bin evidence.  That possible CLIP gem5 work is outside this
no-gem5 compatibility smoke.

5. **Keep experiment outputs out of Git**

```bash
git status --short
git diff --check
```

Expected: no `runs/` or `results/` path is staged or committed.
