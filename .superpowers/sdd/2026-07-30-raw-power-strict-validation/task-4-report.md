# Task 4 Report — Matched R2 Wire-Sensitivity Runner

## Status

Implemented and committed the matched R2 layout-wire sensitivity workflow. No gem5 R2 workload was launched during implementation or verification.

Commit:

```
d3bd3c9 feat: add matched R2 wire sensitivity validation
```

The commit contains only Task 4 production files and Task 4 test hunks:

- `workflow/r2/run_wire_sensitivity.py` (new)
- `workflow/r2/calibrate_lambda_wire.py`
- `tests/test_workflow.py` (Task 4 hunks only; pre-existing unrelated dirty changes remain unstaged)

## Delivered behavior

- `build_sensitivity_vectors()` creates independent vectors and changes only `components_cycles.layout_wire`, the critical L1D-to-L2 path, xbar forward latency, and regenerated gem5 arguments.
- `run_series()` reconstructs its base vector from the completed lifting point, requires the same candidate config, an exact source R1, raw-power module provenance, completed artifacts, and matching workload.
- Each cycle records SHA-256 hashes for R1 `stats.txt` and metadata, lifted modules, CACTI result, lifting performance, candidate config, and generated vector. Successful resumes require the identical hashes and the identical referenced latency-vector path; changed provenance is rejected.
- Only successful actual R2 results become OLS samples. The OLS denominator inputs are `modules.json.ipc1` and `performance.json.sustainable_frequency_ghz`.
- The summary requires exactly FFT, MATMUL, STENCIL, and STREAM. It accepts only accepted local OLS reports whose lambdas all lie within ±25% of their median; otherwise `selected_lambda_wire` is `null`.
- The legacy calibration CLI now refuses its local `--input-config/--output-config` promotion path, preventing a workload-local estimate from becoming a formal lambda.

## TDD evidence

The required vector/selection tests were written before the runner existed. Their recorded RED command/output was:

```text
$ python -m unittest tests.test_workflow.FrequencyTests.test_wire_sensitivity_vectors_change_only_injected_wire tests.test_workflow.FrequencyTests.test_wire_sensitivity_global_acceptance_rule -v
ModuleNotFoundError: No module named 'workflow.r2.run_wire_sensitivity'
FAILED (errors=2)
```

After the minimal runner and global selector were added, the same focused tests produced:

```text
Ran 2 tests in 0.001s
OK
```

The local-formal-promotion regression was then written before closing the legacy CLI path. Its RED run was:

```text
AssertionError: SystemExit not raised
FAILED (failures=1)
lambda_wire=0.05; IPC loss=0.1 per added cycle
```

After the CLI refusal was added, it produced:

```text
calibrate_lambda_wire: error: a local workload lambda cannot write a formal config; use workflow.r2.run_wire_sensitivity --summary after all four workloads
Ran 1 test in 0.002s
OK
```

## Final verification

Executed after the final code changes:

```text
$ python -m unittest discover -s tests -v
Ran 53 tests in 1.443s
OK

$ python -m py_compile workflow/r2/run_wire_sensitivity.py workflow/r2/calibrate_lambda_wire.py
$ git diff --check
```

Also checked the new CLI without executing gem5:

```text
$ python -m workflow.r2.run_wire_sensitivity --help
$ python -m workflow.r2.run_wire_sensitivity --summary --output /tmp/wire-summary-empty.json
selected_lambda_wire=None
$ python -m json.tool /tmp/wire-summary-empty.json >/dev/null
```

## Concerns / follow-up

- Real R2 execution was intentionally not run. The planned FFT/MATMUL/STENCIL/STREAM 0/1/2/3-cycle runs still need to be performed against the raw-power lifting outputs.
- An incomplete or failed cycle can be executed again only if its recorded hashes still match. A changed input causes an explicit resume rejection rather than silently reusing a stale measurement.
- The test working tree contains unrelated pre-existing edits in `tests/test_workflow.py`; they were preserved and not included in this Task 4 commit.

## Strict cycle-level validation fix

Review found that one or two distinct wire-cycle levels could pass the old
non-empty/unique validation. `run_series()` then constructed a base vector
before the too-small series later yielded no calibration. The public vector
builder and the `run_series()` entry boundary now require at least three
distinct, non-negative levels.

TDD evidence:

```text
$ python -m unittest tests.test_workflow.FrequencyTests.test_wire_sensitivity_rejects_fewer_than_three_distinct_cycles tests.test_workflow.FrequencyTests.test_wire_sensitivity_series_rejects_short_cycles_before_preparation -v
FAILED (failures=3)
```

The RED failures showed one/two-level builder calls reaching unrelated missing
base-component validation and `run_series()` reaching the patched `_base_vector`
boundary. After centralizing the cycle validation and calling it before path or
base-vector preparation, the focused check passed:

```text
Ran 4 tests in 0.002s
OK
```

The full suite was then run without gem5 or HotSpot execution:

```text
$ python -m unittest discover -s tests -v
Ran 55 tests in 1.435s
OK
```

## Strict validation boundary regression fix — round 2

Review identified that the prior short-cycle entry test patched only
`_base_vector`. It did not prove that validation runs before output-directory
creation, so an ordering regression could create filesystem state before
raising `ValueError`.

Added `test_wire_sensitivity_series_rejects_short_cycles_before_output_or_execution`.
It invokes `run_series(..., cycles=[0, 1], execute=True)` with a nonexistent
output path; it asserts the `ValueError`, asserts the path was not created,
and installs sentinels at `_base_vector` and `run_r2` to ensure preparation
and external R2 execution have no opportunity to run.

TDD / mutation RED evidence: temporarily moving `output_dir.mkdir()` before
`_validated_cycles()` made the new focused test fail exactly on the observable
filesystem boundary:

```text
$ python -m unittest tests.test_workflow.FrequencyTests.test_wire_sensitivity_series_rejects_short_cycles_before_output_or_execution -v
FAIL: test_wire_sensitivity_series_rejects_short_cycles_before_output_or_execution
AssertionError: True is not false
Ran 1 test in 0.001s
FAILED (failures=1)
```

Restoring the existing validation-first ordering produced this focused result:

```text
$ python -m unittest tests.test_workflow.FrequencyTests.test_wire_sensitivity_rejects_fewer_than_three_distinct_cycles tests.test_workflow.FrequencyTests.test_wire_sensitivity_series_rejects_short_cycles_before_preparation tests.test_workflow.FrequencyTests.test_wire_sensitivity_series_rejects_short_cycles_before_output_or_execution -v
Ran 3 tests in 0.001s
OK
```

Fresh full-suite verification after the test addition:

```text
$ python -m unittest discover -s tests -v
Ran 56 tests in 1.434s
OK
```

No gem5 or HotSpot executable was run for this fix.
