# Task 1 Report: Strict Candidate/Formal Configuration Gates

## Result

Implemented the strict-P1 candidate configuration, validation gate, report-backed promotion utility, and atomic fixed-R1 manifest preparation. No R1 scripts, gem5 runs, or HotSpot long runs were changed or executed.

## Changed Files

- `configs/experiments/clip3d_constrained_5p0_raw_power_p1_candidate.json`: direct-McPAT/local-CACTI candidate copied from `clip3d_constrained_5p0.json`, with `[1]`, `paper-single`, zero beta, candidate parameters, provenance, and formal-validation fields.
- `workflow/run_lifting_pipeline.py`: `validate_config` rejects strict-P1 configurations not using exactly tier `[1]`, `paper-single`, and beta `0.0`.
- `workflow/analysis/promote_validated_config.py`: added `promote(...)`; it requires accepted proxy/wire/frequency reports and fixed-unidentifiable beta, promotes only the report values, and records absolute report paths plus SHA-256 hashes.
- `workflow/analysis/prepare_raw_power_validation.py`: added `prepare(...)`; it requires exactly `("fft", "matmul", "stencil", "stream")`, validates successful matching 32kB/512kB R1 points, records hashes/paths/scopes, and writes only after all four pass.
- `tests/test_workflow.py`: added strict candidate, failed-proxy promotion, and missing-FFT manifest tests.

## TDD Evidence

Production break named before the tests: removing the candidate or either new public utility must make strict configuration loading, promotion, or manifest preparation unavailable or unsafe.

RED command:

```bash
python -m unittest tests.test_workflow.FormalGuardTests.test_raw_power_p1_candidate_has_only_top_tier_l2 tests.test_workflow.FormalGuardTests.test_promotion_rejects_failed_proxy -v
```

RED output (exit 1):

```text
ERROR: test_workflow (unittest.loader._FailedTest.test_workflow)
ImportError: Failed to import test module: test_workflow
ModuleNotFoundError: No module named 'workflow.analysis.prepare_raw_power_validation'
Ran 2 tests in 0.000s
FAILED (errors=2)
```

The expected missing-public-module failure was observed before implementation.

GREEN command:

```bash
python -m unittest tests.test_workflow.FormalGuardTests.test_raw_power_p1_candidate_has_only_top_tier_l2 tests.test_workflow.FormalGuardTests.test_promotion_rejects_failed_proxy tests.test_workflow.FormalGuardTests.test_prepare_validation_manifest_requires_the_four_named_r1_points -v
```

GREEN output (exit 0):

```text
test_raw_power_p1_candidate_has_only_top_tier_l2 ... ok
test_promotion_rejects_failed_proxy ... ok
test_prepare_validation_manifest_requires_the_four_named_r1_points ... ok
Ran 3 tests in 0.001s
OK
```

## Full Suite Evidence

Command run immediately before committing:

```bash
python -m unittest discover -s tests -v
```

Output: exit 0; `Ran 35 tests in 1.417s`; `OK`. This included all three new `FormalGuardTests` and the existing transient, parser, frequency, grid, and formal-guard tests. No tests were skipped in this environment.

## Commit

- `5736952 feat: guard raw-power strict P1 configuration promotion`

## Self-Review

- Candidate strict-P1 constraints are checked by the normal pipeline validator before any lifting operation.
- Promotion cannot treat bare numbers as formal values: it reads report files, requires their acceptance/status gates, and hashes the exact source artifacts.
- Manifest construction validates all four input points before calling the atomic writer, so a failure does not create a partial output manifest.
- Only Task 1 hunks were staged from the already-dirty shared files; unrelated raw-power-calibration-removal changes remain unstaged.
- Positive promotion and complete-manifest paths were not run against real reports/R1 outputs because Task 1 must not launch gem5/HotSpot experiments; later tasks produce the required accepted artifacts.

## Fix Round 1: Reviewer Findings

### Root Cause

`validate_config` previously constrained only strict-P1 tier/policy/beta values.  It did not inspect an accepted configuration's promotion artifacts or tie optimizer numbers back to the accepted proxy/wire reports.  `prepare` also used caller-provided cache-size strings to construct input paths and allowed a missing or blank instruction-window scope into a manifest.

### TDD Evidence

Added four behavior tests before production changes: a forged accepted formal config is rejected; promotion emits report-derived, hash-recorded provenance that validates; noncanonical cache sizes are rejected before point discovery; and an empty instruction-window scope fails without a manifest.

RED command (exit 1):

```bash
python -m unittest tests.test_workflow.FormalGuardTests.test_accepted_formal_config_rejects_manual_parameters_without_artifact_provenance tests.test_workflow.FormalGuardTests.test_promotion_emits_report_derived_formal_config_that_validates tests.test_workflow.FormalGuardTests.test_prepare_rejects_noncanonical_cache_sizes_before_point_discovery tests.test_workflow.FormalGuardTests.test_prepare_rejects_missing_instruction_window_scope_without_manifest -v
```

The pre-fix run had three assertion failures (manual accepted values, prose-only promotion provenance, and blank scope) and one point-discovery `FileNotFoundError` instead of the requested cache-size rejection.

GREEN command (exit 0): the same four tests; `Ran 4 tests in 0.004s; OK`.

### Fix Details

- Accepted strict-P1 configs now require accepted proxy/wire/frequency artifacts with absolute paths, lowercase 64-character SHA-256 values, matching on-disk hashes, and accepted report contents.  The validator verifies the proxy beta status and exact numeric alpha/cross-tier/lambda derivation through structured parameter-provenance entries; beta remains a fixed zero provenance record.
- Promotion now writes that structured provenance before its final validation.
- Preparation now accepts only the four canonical 32kB/512kB points and requires each metadata scope to be a nonblank string before it writes the manifest.

### Verification

```bash
python -m unittest tests.test_workflow.FormalGuardTests -v
# Ran 14 tests in 0.006s; OK
python -m unittest discover -s tests -v
# Ran 39 tests in 1.425s; OK
```

No R1/gem5/HotSpot execution was performed.
