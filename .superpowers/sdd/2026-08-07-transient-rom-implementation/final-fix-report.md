# Transient-ROM final fix report

Date: 2026-08-08

Base commit: `d95e3b440255e3e65e9cad36309281795fc72e07`

Latest replay-hardening round base:
`bff1704444f1ec4fec5394f2c3869b658ba6f2ca`

Scope: final correctness, reuse-integrity, failure-semantics, and auditability
fixes for the non-formal transient-ROM path. The steady Eq. 13 implementation
was intentionally left unchanged.

## Findings and root causes

1. The accepted package did not persist the complete calibration design.
   Anchors, holdouts, interpolation domains/simplices, PRBS inputs, and the base
   layout could therefore be regenerated differently while an old fitted
   model was reused. The package identity hashed too little state, and the
   optimizer rebuilt rather than loaded that state.
2. Package reuse discarded acceptance evidence and made historical calibration
   calls look like zero total calls. The reuse path retained neither the exact
   8 training plus 2 holdout records nor the validation metrics needed to audit
   why the package had been accepted.
3. Discrete-to-continuous conversion recorded numerical diagnostics without
   enforcing all of them. A poorly conditioned matrix logarithm or inaccurate
   exp/log round trip could therefore produce an accepted model.
4. Final real-HotSpot validation errors were conflated. Tool execution failure,
   malformed validation evidence, PSS nonconvergence, and a converged but
   genuinely unsafe frequency grid did not have distinct outcomes; some paths
   could also escape without an auditable summary or proceed ambiguously toward
   R2.
5. Direct optimizer reuse initially trusted a shallower contract than pipeline
   reuse, and case artifacts used paths tied to their original workspace. A
   self-consistent but incomplete package could pass direct reuse, while a
   legitimate package could fail after relocation.
6. Source provenance was initially asserted rather than derived. Manifest
   source hashes, holdout power identity, and PRBS metadata were progressively
   bound to the accepted identity, but the last review found that rehashed
   training powers could still retain truthful PRBS metadata without actually
   being generated from the bound source trace. The root cause was byte-level
   inventory checking without a semantic reconstruction of the excitation.
7. Reuse still treated each holdout's `full_grid_converged` flag and validation
   report as declarations. It hashed the temperature trace but did not parse it,
   so a rehashed one-row trace, absent frequency, forged convergence metrics,
   or peak/safety claim could retain an accepted package marker.
8. The final-search exception boundary covered the call but not its returned
   value. A non-dictionary return escaped through `.get()`, while missing or
   malformed evaluations, non-finite sustainable frequency, and inconsistent
   state/safety could be mislabeled or reach the R2 boundary.
9. The first semantic holdout-reuse fix still replayed only the real-HotSpot
   half of each comparison. Reuse derived the trace peak, safety, PSS, and grid
   identity, but continued to trust persisted ROM peak/safety, RMSE, peak
   error, per-gate booleans, and accepted flags. The root cause was a second,
   partial implementation of the holdout contract in `contracts.py` rather
   than reuse of the calibration-time comparison itself.
10. The first returned-search validator checked selected invariants but did not
    prove that the recorded midpoint evaluations and refined brackets were the
    output of the canonical local-boundary search. It also trusted the recorded
    `converged` bit without validating its complete PSS evidence, and direct
    `float()` conversion of an arbitrarily large JSON integer could escape as
    `OverflowError`. The root cause was validating a summary of the search
    instead of replaying the search algorithm from safely normalized evidence.

## Implemented behavior

- `anchors.json` now contains the exact accepted calibration design, including
  training and holdout points, legal domains and simplices, PRBS multipliers,
  allowed tiers, and base layout. Its canonical hash participates in the ROM
  input identity, calibration manifest, fit report, model metadata, and
  optimizer reuse gate. The optimizer loads this saved design.
- Calibration packages are self-contained and relocatable. Module bytes are
  copied into the package, case artifact paths are package-relative, and reuse
  rejects absolute paths, parent traversal, containment escapes, symlinks,
  missing files, changed hashes, and incomplete artifact inventories.
- Both pipeline and direct optimizer reuse call the same deep package
  validator. It requires the exact 8+2 IDs, points, layouts, source identities,
  artifact hashes, training hash maps, POD/conditioning/conversion evidence,
  holdout thresholds and gates, saved model metadata, and complete manifest.
- Source binding now compares `modules_sha256`, `power_trace_identity`,
  `config_sha256`, and `hotspot_sha256` with the accepted identity. Holdout
  source identity is recomputed from packaged bytes. For every training case,
  the validator rebuilds the complete PRBS power windows from the bound holdout
  source with the persisted multipliers and requires exact equality with the
  packaged excitation.
- Calibration now packages the exact hashed config bytes. Reuse resolves that
  package-relative config, validates the two specified holdout frequencies
  (`f0` and `0.6*f0 + 0.4*fmin`), re-parses every holdout temperature trace,
  and calls the existing `summarize_period_end_convergence()` and
  `last_period_peak()` helpers. It requires the exact repeated-period sample
  count, full-grid convergence, exact recorded PSS metrics/grid order, model
  grid identity, and validation-report HotSpot peak/safety derived from the
  trace and bound `tsafe_c`.
- Reuse preserves historical calibration evidence and reports the historical
  8 training, 2 holdout, and 10 total HotSpot calls separately from zero calls
  made by the current reuse invocation.
- Model fitting now verifies the case-recorded training artifact hashes and
  enforces finite upper bounds for matrix-logarithm input conditioning, the
  `logm` error estimate, and exp/log reconstruction error, in addition to the
  existing pole-stability and imaginary-component checks.
- Final HotSpot validation occurs before R2. Tool errors, validation-contract
  errors, PSS nonconvergence, and true thermal infeasibility are recorded as
  distinct failure reasons under the failed final-validation outcome. R2 is
  skipped on all validation failures, ROM predictions remain in the summary,
  and partial validation calls are counted.
- The result of `search_layout_frequency()` is now validated inside the same
  recovery boundary as the call. The validator requires a dictionary, a
  non-empty ordered evaluation list covering the configured grid, finite
  in-range frequencies/peaks, boolean convergence, safety consistent with
  convergence and `tsafe_c`, matching sustainable frequency/state, and
  consistent grid/search/PSS metadata. Contract failures write
  `final_validation_failure.json`, preserve ROM predictions, and never build an
  R2 vector or launch R2.
- Calibration holdout validation now has a non-publishing mode. Reuse loads the
  saved POD model, persisted design, packaged config, two case layouts/powers,
  and immutable temperature traces, then calls the exact calibration-time
  evaluator and gate implementation without rewriting any package artifact.
  The complete recomputed report must equal `validation_report.json`, including
  ROM and HotSpot peaks/safety, RMSE, peak error, every gate, failure reasons,
  and accepted flags; the recomputed result must independently be accepted.
- Final HotSpot validation now checks the complete
  `period_end_convergence` structure for every grid and midpoint evaluation,
  derives convergence from the repeated-period count and final full-grid delta,
  and derives safety from that convergence plus the inclusive final-period
  peak. It maps every recorded frequency and replays
  `find_sustainable_frequency()` with the saved evaluator results. The entire
  rebuilt search—including midpoint set, grid
  evaluations, local brackets, monotonic flag, state, and sustainable
  frequency—must equal the returned search before R2 can start. Numeric
  normalization converts non-finite and non-convertible/overflowing JSON
  numbers into a validation-contract `ValueError` inside the recovery boundary.
- The standalone `calibrate_rom` CLI now finalizes
  `rom_artifact_manifest.json` through the same shared package-finalization
  helpers as pipeline calibration.
- Configuration and both design/usage documents now describe the additional
  continuous-conversion thresholds, persisted design, package-reuse contract,
  and final-validation semantics.

Primary implementation files:

- `workflow/transient/rom/calibration_design.py`
- `workflow/transient/rom/materialize_calibration.py`
- `workflow/transient/rom/calibrate_rom.py`
- `workflow/transient/rom/contracts.py`
- `workflow/transient/rom/pod_state_space.py`
- `workflow/transient/rom/optimize_layout.py`
- `workflow/transient/rom/run_pipeline.py`
- `configs/experiments/clip3d_transient_rom_exploratory.json`
- `docs/superpowers/specs/2026-08-07-transient-rom-design.md`
- `docs/transient_rom_usage_zh.md`
- `tests/test_transient_rom.py`

## TDD and review regressions

The fixes were driven by regressions covering persisted-design identity,
reuse evidence/counts, numerical gates, training artifact hashes, relocatable
packages, direct-optimizer deep validation, CLI manifest finalization, final
validation failure classes, R2 suppression, partial call accounting, source
identity, and PRBS derivation.

The prior review condition was reproduced independently:

```text
RED: test_optimizer_rejects_training_power_not_derived_from_bound_source
     failed because the tampered, rehashed package reached ROM search:
     AssertionError: ROM search ran before the current identity gate

GREEN: the same test passed after the validator rebuilt the expected PRBS
       trace from the packaged holdout source and compared every training
       trace exactly.
```

Representative package/validation regressions also prove that reuse rejects a
changed saved design, changed training point or layout, changed conversion
threshold, changed manifest source identity, holdout power from another source,
training power not derived from the bound source, missing holdout evidence, and
an incomplete/stale artifact manifest. A relocated self-contained package is
accepted and reused without regeneration or HotSpot inside the optimizer.

The preceding re-review round added two independent RED groups before production
changes:

```text
Holdout RED (5 tests): missing frequency, one-row rehashed trace, forged PSS
metrics, forged trace peak, and forged safety all passed the package gate and
reached ROM search (`ROM search ran before the current identity gate`).

Final-search RED (7 tests): non-dict returned an uncaught AttributeError;
missing/invalid evaluations were mislabeled `missing_pss_evidence`; missing or
non-finite sustainable frequency and inconsistent state/safety were accepted or
reached R2.

GREEN: all 12 focused regressions pass. Existing pss_nonconvergence remains a
distinct validation failure, and an all-converged unsafe grid remains genuine
`thermally_infeasible` rather than a contract/tool error.
```

The latest round added seven more RED regressions before its production
changes:

```text
Persisted-report RED (4 tests): forged ROM peak, forged ROM safety, forged
peak error, and a self-consistent forged safety gate/accepted flag all reached
ROM search (`ROM search ran before the current identity gate`).

Canonical-search RED (3 tests): a forged refined-safe bracket and PSS evidence
inconsistent with the recorded converged bit both reached the R2 boundary; a
10**400 JSON integer escaped the recovery boundary as OverflowError.

GREEN: the four persisted-report regressions are rejected before ROM search,
and all three canonical-search regressions become validation_contract_error
with no latency-vector construction or R2 launch. The validator does not modify
the forged persisted report while checking it.
```

## Fresh verification

Focused package-reuse, final-pipeline, and calibration-case suites:

```bash
/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom.ROMOptimizerTests \
  tests.test_transient_rom.ROMPipelineTests \
  tests.test_transient_rom.ROMCalibrationCaseTests -v
```

Result: `52 tests` passed.

Complete transient-ROM test module:

```bash
/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom -v
```

Result: `89 tests` passed.

Relevant integration suite, after the replay-hardening fixes:

```bash
/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest \
  tests.test_transient_rom tests.test_transient tests.test_workflow
```

Result: `225 tests` passed; `2` expected skips for unavailable live HotSpot
tests.

Full discovery, after the replay-hardening fixes:

```bash
/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest discover -s tests
```

Result: `373 tests` run; `1` error and `2` expected skips. The only error is the
pre-existing historical fixture gap in
`test_lambda_wire_exploratory_config.py`:

```text
results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/
lambda_wire_report.json
```

That test/config are unchanged from the base commit, and `git cat-file` confirms
the required report is absent from the base tree. The artifact was not
fabricated.

Additional checks:

```text
git diff --check                                                        PASS
python -m compileall -q workflow tests                                  PASS
git diff --exit-code d95e3b4 -- workflow/thermal/sustainable_frequency.py \
  workflow/floorplan/optimize_layout.py                                 PASS
```

The last command confirms no change to the steady sustainable-frequency or
steady layout-optimizer implementation relative to the requested base.

## Review status and remaining concerns

No Critical review finding remains. All confirmed Important findings were
implemented. The latest conditions—complete two-sided holdout recomputation,
canonical final-search replay, and overflow-safe contract failure—are covered
by the seven-test RED/GREEN round above, in addition to the preceding 12-test
reuse/final-validation round.

No live HotSpot, gem5, McPAT, CACTI, R1, R2, or EDA tool was invoked during
these fixes or verification. Tests use synthetic artifacts and mocks at the
external execution boundaries. Consequently, the remaining validation concern
is physical rather than contractual: this change set has no live physical
HotSpot run and no single real end-to-end physical fixture spanning source
trace, all 8+2 solves, fitting, optimization, final HotSpot validation, and R2.
The synthetic package, relocation, CLI, and orchestration regressions cover the
software contracts but cannot replace that future physical validation run.
