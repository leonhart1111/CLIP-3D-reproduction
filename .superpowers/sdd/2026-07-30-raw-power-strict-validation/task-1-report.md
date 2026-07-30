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
