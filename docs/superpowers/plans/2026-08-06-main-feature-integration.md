# Main Feature Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the nonzero-`lambda_wire`, traffic-weighted wire, transient thermal, and six-decimal temperature work into `main` without including the strict-P1 global-bound tool or changing completed R1 results.

**Architecture:** Start from `feature/traffic-weighted-wire`, whose ancestry already contains the exploratory nonzero-lambda configuration, and merge `feature/matmul-transient-validation` in an isolated integration worktree. Resolve the shared files semantically so the current raw-power/no-calibration path and traffic aggregation remain intact while the transient validation and precision-preserving paths are added. After full verification, preserve the dirty main checkout on a named safety branch, then merge the validated integration branch into a clean `main`.

**Tech Stack:** Git, Python 3 standard-library `unittest`, gem5/McPAT/CACTI/HotSpot workflow code, JSON experiment configurations.

## Global Constraints

- Do not merge or cherry-pick `feature/strict-p1-global-bound`.
- Do not rerun, alter, delete, or relocate existing R1 outputs.
- Preserve the raw McPAT power path; do not restore removed dynamic/leakage calibration factors.
- Preserve `lambda_wire = 0.0020119160767721133` as explicitly exploratory and non-formal.
- Preserve both arithmetic-mean wire aggregation and the opt-in communication-frequency-weighted extension.
- Preserve transient `sample_ms` reporting and six-decimal steady/transient HotSpot temperature handling.
- Do not commit ignored `results/`, `runs/`, tool binaries, or the local `results` symlink used only to expose test evidence inside the worktree.

---

### Task 1: Merge transient and precision work into the traffic-weighted base

**Files:**
- Merge: `feature/matmul-transient-validation`
- Resolve: `tests/test_workflow.py`
- Resolve: `workflow/README.md`
- Resolve: `workflow/common.py`
- Resolve: `workflow/mcpat/parse_mcpat.py`
- Resolve: `workflow/run_lifting_pipeline.py`
- Resolve: `workflow/transient/run_windowed_mcpat.py`
- Add from transient branch: `patches/hotspot/0001-six-decimal-temperature-output.patch`
- Add from transient branch: `workflow/transient/compare_layouts.py`
- Add from transient branch: `workflow/transient/run_dual_layout_validation.py`
- Add from transient branch: `workflow/transient/validation.py`

**Interfaces:**
- Consumes: traffic-weighted `communication_profile`, `wire_aggregation`, and `derive_layout_delays(..., aggregation=..., communication_weights=...)` behavior.
- Produces: one tree supporting both traffic-weighted R2 latency construction and transient dual-layout validation with six-decimal temperature artifacts.

- [ ] **Step 1: Merge without committing so every conflict is visible**

Run:

```bash
git merge --no-ff --no-commit feature/matmul-transient-validation
git status --short
git diff --name-only --diff-filter=U
```

Expected: only shared implementation/test/documentation files require conflict resolution; no strict-P1 global-bound file appears.

- [ ] **Step 2: Resolve shared code semantically**

For each conflicted file, retain the traffic branch's raw-power and communication-weight interfaces and incorporate the transient branch's validation and precision changes. Reject any resolved code containing `dynamic_power_scale`, `leakage_power_scale`, or fixed `10 ms` result-report text.

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' . --glob '!results/**' --glob '!runs/**'
rg -n 'dynamic_power_scale|leakage_power_scale|10 ms' workflow tests docs
```

Expected: no merge markers; no restored calibration fields; any remaining `10 ms` mention is explanatory or an example rather than a hard-coded report value.

- [ ] **Step 3: Run focused regression tests before committing the merge**

Run:

```bash
python3 -m unittest -v tests.test_workflow tests.test_transient tests.test_lambda_wire_exploratory_config
```

Expected: all available tests pass; HotSpot-dependent tests may be skipped only when the executable is unavailable.

- [ ] **Step 4: Commit the merge**

Run:

```bash
git add README.md configs docs patches tests workflow
git commit -m "merge: integrate transient and traffic-weighted workflows"
```

Expected: a two-parent merge commit whose second parent is `feature/matmul-transient-validation`.

### Task 2: Verify cross-feature contracts

**Files:**
- Verify: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_exploratory.json`
- Verify: `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json`
- Verify: `tests/test_lambda_wire_exploratory_config.py`
- Verify: `tests/test_transient.py`
- Verify: `tests/test_workflow.py`
- Verify: `patches/hotspot/0001-six-decimal-temperature-output.patch`

**Interfaces:**
- Consumes: the integrated tree from Task 1 and the local ignored lambda evidence artifact.
- Produces: test evidence that each requested feature remains reachable and strict-P1 global-bound code is absent.

- [ ] **Step 1: Verify configuration classification and option wiring**

Run:

```bash
python3 -m unittest -v \
  tests.test_lambda_wire_exploratory_config \
  tests.test_workflow.FormalGuardTests.test_traffic_weighted_exploratory_config_is_non_formal \
  tests.test_workflow.GridTests.test_optimizer_and_r2_use_the_same_traffic_weighted_aggregate
```

Expected: all selected tests pass.

- [ ] **Step 2: Verify transient precision and sampling contracts**

Run:

```bash
python3 -m unittest -v tests.test_transient
```

Expected: all transient tests pass, with only explicitly environment-dependent tests skipped.

- [ ] **Step 3: Verify repository scope**

Run:

```bash
test -f patches/hotspot/0001-six-decimal-temperature-output.patch
test ! -e workflow/analysis/strict_p1_global_bound.py
git log --oneline --all -- workflow/analysis/strict_p1_global_bound.py
```

Expected: the precision patch exists, the strict-P1 tool is absent from the integration tree, and any strict-P1 history shown belongs only to its separate branch.

### Task 3: Run full integration verification

**Files:**
- Verify: all tracked Python workflow, tests, configurations, documentation, and patch files.

**Interfaces:**
- Consumes: the committed integration branch.
- Produces: fresh test, syntax, whitespace, history, and scope evidence suitable for merging to `main`.

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: zero failures and zero errors; only documented missing-tool skips are allowed.

- [ ] **Step 2: Compile all Python modules**

Run:

```bash
python3 -m compileall -q workflow scripts configs/gem5 tests
```

Expected: exit status 0.

- [ ] **Step 3: Check diff hygiene and branch ancestry**

Run:

```bash
git diff --check main...HEAD
git merge-base --is-ancestor feature/traffic-weighted-wire HEAD
git merge-base --is-ancestor feature/matmul-transient-validation HEAD
test ! -e workflow/analysis/strict_p1_global_bound.py
git status --short --branch
```

Expected: all commands return zero and no tracked working-tree changes remain.

### Task 4: Preserve the dirty checkout and integrate into main

**Files:**
- Preserve: all tracked and untracked files currently present in `/home/zyjiang/Agenticflow/CLIP`.
- Merge into: local branch `main`.

**Interfaces:**
- Consumes: verified `integration/main-feature-integration` and the existing dirty main checkout.
- Produces: local `main` containing both requested feature branches while retaining a named safety snapshot of the pre-merge checkout.

- [ ] **Step 1: Prove the existing tracked checkout matches the traffic branch before preserving it**

Run from the main checkout:

```bash
git diff --exit-code feature/traffic-weighted-wire -- . ':!docs/superpowers/plans/2026-08-06-main-feature-integration.md'
cmp \
  configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json \
  <(git show feature/traffic-weighted-wire:configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json)
git status --short --branch
```

Expected: tracked contents and the tracked traffic configuration match the traffic branch; unrelated untracked user documents remain visible.

- [ ] **Step 2: Create a non-destructive safety branch if the main checkout is still dirty**

Run:

```bash
git switch -c safety/pre-integration-main-20260806
git add -u
git add configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json
git commit -m "chore: preserve pre-integration main worktree"
git switch main
```

Expected: the prior tracked working state and traffic configuration are recoverable from the safety branch; unrelated untracked files remain in place.

- [ ] **Step 3: Merge the verified integration branch into main**

Run:

```bash
git merge --no-ff integration/main-feature-integration \
  -m "merge: integrate nonzero lambda, transient thermal, and traffic weighting"
```

Expected: merge succeeds without importing `workflow/analysis/strict_p1_global_bound.py`.

- [ ] **Step 4: Verify the actual merged main tree**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q workflow scripts configs/gem5 tests
git merge-base --is-ancestor feature/traffic-weighted-wire main
git merge-base --is-ancestor feature/matmul-transient-validation main
test ! -e workflow/analysis/strict_p1_global_bound.py
git status --short --branch
```

Expected: tests and compilation pass, both requested branches are ancestors of `main`, strict-P1 remains excluded, and only the previously preserved unrelated untracked documents may remain.
