# Canonical Balanced-50 final cross-cutting fix report

Date: 2026-08-07

Fix base: `c8d609e24dc19eef939cfd31de9b1b8afedbb1bc`

## Outcome

The four Important findings in `final-review-findings.md` are implemented and
covered by focused regressions. The final scoped re-review reported no residual
Critical or Important findings and judged this fix wave merge-ready for its
scope. No canonical R1 point, real R2 simulation, HotSpot executable, or other
external scientific tool was run.

## Implemented fixes

### 1. Fail-closed canonical R1 plan certification

- `planned_jobs.json` is required; absence or malformed top-level content is a
  certification failure.
- `job_count` must be a typed integer equal to `len(jobs)` and to the expected
  canonical count.
- Every ordered job must have the exact canonical architecture key, selected
  profile, and resolved canonical output directory.
- Missing, extra, duplicate, reordered, wrong-profile, and quarantined-output
  jobs are rejected by the shared catalogue validation used by audit and
  refresh. Audit output is schema 3.

### 2. Type-strict R2 override identity

- The override mapping has exactly the canonical key set.
- Every latency is a positive, non-boolean integer in the uint64 domain.
- `gem5_args` must be the exact canonical ordered rendering of the mapping.
- One shared validator/equality operation is used by R2 execution, exact reuse,
  paired branch selection, and lifting compatibility checks. Python loose
  equality (`True == 1`, `8 == 8.0`) cannot select reuse.

### 3. Physical/local attachment coherence

- Added `workflow/r2/attachment_validation.py` as the shared physical validator
  and point-lock owner.
- Module architecture and R1 source are bound to canonical metadata.
- IPC1 is recomputed from the captured, stability-checked R1 `stats.txt` bytes.
- Module dynamic/leakage/total power records are internally checked and summed
  back to the declared totals; gamma is recomputed from those totals.
- HotSpot ambient/cooling inputs are bound to the captured manifest.
- Tmax Kelvin/Celsius, peak unit, and sample count are recomputed from the
  captured, stability-checked `steady.txt` bytes.
- Performance and every derived summary field are recomputed from the captured
  modules, thermal evidence, selected config, and validated IPC2.
- Local attachment publishes performance/summary atomically with rollback.
  Scheduler resume and final reporting call the same validator and consume its
  captured snapshot. Reuse provenance now hashes the manifest and steady output.
- Reporting accepts an explicit `--status-root`, matching the runner's status
  namespace instead of assuming only the default sibling directory.

### 4. Concurrency and completion certification

The global acquisition order is:

1. physical-root sweep locks (resolved path order);
2. physical-point pair locks (resolved path order);
3. physical point attachment lock.

Sweep locks are stored as `.paired-r2-sweep.lock` in both fixed and CLIP roots.
Pair locks are stored as `.paired-r2-pair.lock` in both selected physical point
directories. Therefore changing `status_root` cannot bypass execution locks,
and partially shared physical roots still contend safely.

- A sweep takes exclusive, fail-fast root locks before preflight and holds them
  through worker shutdown, live pair revalidation, and final experiment-status
  publication.
- A direct `run_pair` takes shared root locks, so it waits for an active sweep;
  it then takes both point-level pair locks before reading cached status or
  launching work.
- Internal sweep workers skip only the already-owned outer lock level and still
  take both pair locks.
- Reuse-state clearing takes the physical attachment lock.
- JSON writes use collision-resistant, same-directory `mkstemp` files followed
  by `os.replace`.
- Intermediate experiment statuses cannot claim `complete=true`. Final status
  rereads each persisted pair under its physical pair locks, validates live
  fixed/CLIP/R1/config evidence, and counts only certified records. The outer
  sweep lock prevents a direct pair writer from changing an already-certified
  pair before final publication.

## TDD evidence

### Finding 1

Focused regressions covered a missing plan, duplicate count-preserving jobs,
quarantined output redirection, and missing/extra/reordered/wrong-profile jobs.
They failed against the fail-open implementation and passed after exact ordered
plan validation was shared by audit and refresh.

### Finding 2

Focused regressions covered float-versus-int, bool-versus-int, zero/overflow,
missing/extra keys, noncanonical argument spelling, and scheduler branch
selection. They failed against loose mapping equality and passed after the
shared strict latency-vector validator was installed.

### Finding 3: final raw-evidence strengthening

RED command:

```text
python -m unittest \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_coherent_ipc1_not_derived_from_r1_stats \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_coherent_tmax_not_derived_from_steady_output \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_thermal_identity_not_derived_from_steady_output \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_hotspot_manifest_not_bound_to_config \
  tests.test_balanced_experiment.StrictR2ReuseTests.test_local_attach_rejects_power_totals_not_derived_from_module_records \
  tests.test_balanced_experiment.PairedAggregationTests.test_report_rejects_fabricated_summary_and_performance_temperature -v
```

Before implementation, all six test methods rejected the current behavior
(the peak/sample method contains two failing subcases). After implementation:
`Ran 6 tests ... OK`. The complete strict reuse class then passed 44/44.

### Finding 4: final physical-lock strengthening

RED command covered seven test methods:

- physical sweep fail-fast;
- sweep lock held through worker and status publication;
- pair lock before cache validation;
- shared pair identity across different status roots;
- direct pair blocked by an active physical sweep;
- pair lock before local/reuse attachment;
- point lock while clearing reuse state.

Before implementation: `Ran 7 tests ... FAILED (failures=7)`.
After implementation: `Ran 7 tests ... OK`. The complete paired-runner class
then passed 40/40. An additional explicit status-root reporting regression moved
from missing-API RED to `Ran 1 test ... OK`.

Earlier Finding-4 RED/GREEN work also covered collision-resistant publication,
persisted-status tampering, pair-lock final validation, and valid 50-pair
completion (nine focused failures before implementation, then nine passing).

## Fresh verification

```text
python -m unittest \
  tests.test_balanced_experiment.AtomicJsonPublicationTests \
  tests.test_balanced_experiment.StrictR2ReuseTests \
  tests.test_balanced_experiment.PairedR2RunnerTests \
  tests.test_balanced_experiment.PairedAggregationTests
```

Result: `Ran 103 tests in 8.527s` — `OK`.

The single final full-discovery run was:

```text
python -m unittest discover -s tests -v
```

Result: `Ran 271 tests in 15.448s` — `OK (skipped=2)`.

```text
python -m compileall -q workflow scripts tests
```

Result: exit 0, no output.

```text
git diff --check
```

Result: exit 0, no output.

## Files in the fix wave

- `tests/test_balanced_experiment.py`
- `workflow/analysis/audit_r1.py`
- `workflow/analysis/refresh_r1_plan.py`
- `workflow/analysis/summarize_paired_sweep.py`
- `workflow/common.py`
- `workflow/r1_catalog.py`
- `workflow/r2/attachment_validation.py` (new)
- `workflow/r2/attach_result.py`
- `workflow/r2/reuse_result.py`
- `workflow/r2/run_paired_sweep.py`
- `workflow/r2/run_r2.py`
- `workflow/run_lifting_pipeline.py`
- `workflow/thermal/run_hotspot.py`
- `workflow/thermal/sustainable_frequency.py`
- this report

The untracked `results` symlink and all generated experiment outputs are
intentionally excluded.

## Self-review and residual concerns

- Scoped read-only re-review after the final fixes found no residual Critical or
  Important issues.
- The report CSV/JSON two-file publication remains rollback-capable for ordinary
  exceptions but not process-death atomic across both renames. The requested
  versioned-bundle/current-pointer improvement remains deferred as a Minor to
  avoid changing public report paths in this fix wave.
- The direct symlink-escape regression remains deferred as a Minor; resolved-path
  containment is already enforced.
- Filesystem locking relies on POSIX `flock` semantics and on both processes
  seeing the same physical-root filesystem. This matches the supported Linux
  execution environment.
- Real one-pair smoke and full operational experiment execution remain pending
  post-merge by design; this fix wave deliberately ran no scientific tools.
