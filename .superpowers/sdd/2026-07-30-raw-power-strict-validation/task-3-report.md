# Task 3 report: spatially separated raw-power frequency validation

## Implementation

- Added `compose_separated_ptrace(dynamic, leakage, destination, frequency, f0)`.
  It writes every cell as `P_leak + (f/f0) P_dyn` and returns the dynamic
  scale plus dynamic, leakage, composed, and available f0-total trace sums.
- Power trace input checks reject malformed one-sample traces, header/order or
  length disagreement, non-finite values, and (when `power.ptrace` exists)
  any cell whose total differs from dynamic plus leakage by more than `1e-9 W`.
- `validate_case` now defaults to `separated-dynamic-leakage`; it runs HotSpot
  from `power_dynamic.ptrace` and `power_leakage.ptrace`, reports the actual
  HotSpot temperature and trace sums, and keeps `paper-uniform-gamma` as an
  explicit alternate mode.  Each run separately records the scalar
  uniform-gamma closed-form comparison.
- Frequency settings are validated and accepted from the anchor manifest.
  When `f_sus < f0`, the extra safety solve is marked accepted only for a
  finite result within `1.0 C` of `T_safe`; skipping that required safety solve
  makes the recommendation unaccepted.
- Updated the example anchor manifest with the requested raw-power frequency
  settings.  The example does not claim that anchor experiments were run.

## TDD evidence

1. RED (before implementation):

   ```text
   python -m unittest tests.test_workflow.FrequencyTests.test_separated_frequency_trace_scales_only_dynamic_power -v
   ImportError: cannot import name 'compose_separated_ptrace'
   FAILED (errors=1)
   ```

2. GREEN after the initial composition implementation:

   ```text
   Ran 2 tests in 0.001s
   OK
   ```

3. The default-validation and manifest-forwarding tests were added next and
   initially failed against the uniform-gamma default / unforwarded settings.
   They pass with the raw-power implementation.  The real HotSpot smoke test
   also initially exposed its obsolete uniform-scaling error threshold
   (`0.278325 C` versus `<0.05 C`); it now asserts the separated-mode contract
   directly.

## Fresh verification

```text
python -m unittest discover -s tests -v
Ran 48 tests in 1.433s
OK

python -m py_compile workflow/thermal/validate_frequency.py workflow/thermal/run_anchor_validation.py
exit 0

git diff --check
exit 0
```

No gem5 or anchor HotSpot experiment was run.  The existing small unit-test
HotSpot smoke test executed as part of the requested suite.

## Fix round 1: strict separated-model naming and safety-solve auditability

### Implementation

- Renamed the primary per-case, anchor-summary, and CLI metric to
  `max_abs_uniform_gamma_comparison_error_c`.  The removed
  `max_abs_linear_error_c` name is not retained as an alias: in
  `separated-dynamic-leakage` mode it would incorrectly imply linearity of the
  spatially separated physical model.  The per-run
  `uniform_gamma_comparison.error_vs_hotspot_c` remains explicitly labelled as
  a paper uniform-gamma comparison.
- A failure from the actual `run_hotspot` call for a mandatory below-`f0`
  safety solve is now caught at that narrow boundary.  `validate_case` writes
  and returns a result with `solution_validation.accepted: false`, a textual
  `solution_validation.error`, no fabricated temperature/error value, and a
  rejected recommendation.  Requested-frequency HotSpot failures still
  propagate normally.  Trace composition/validation happens before that
  catch, unchanged.
- Anchor summaries now report `requested_frequency_hotspot_run_count`,
  `fsus_safety_solve_count`, `hotspot_run_count`, aggregate
  `recommendation.accepted`, and the renamed primary comparison metric.
  Failed safety attempts are counted as actual HotSpot invocations, while
  `max_safe_error_c` ignores their absent numeric error.

### TDD evidence

1. RED, before the production changes:

   ```text
   python -m unittest tests.test_workflow.FrequencyTests.test_frequency_validation_defaults_to_separated_hotspot_trace tests.test_workflow.FrequencyTests.test_below_f0_hotspot_failure_writes_a_rejected_validation_result tests.test_workflow.FrequencyTests.test_anchor_summary_reports_actual_hotspot_run_counts_and_acceptance -v
   FAILED (failures=1, errors=2)

   AssertionError: 'max_abs_uniform_gamma_comparison_error_c' not found
   RuntimeError: injected f_sus HotSpot failure
   KeyError: 'max_abs_linear_error_c'
   ```

2. GREEN after the focused implementation:

   ```text
   Ran 3 tests in 0.003s
   OK
   ```

   The tests exercise result schemas and the on-disk JSON output.  The anchor
   aggregation test uses lightweight patched `validate_case` result objects,
   rather than inspecting source text.

### Fresh verification

```text
python -m unittest discover -s tests -v
Ran 50 tests in 1.434s
OK

python -m py_compile workflow/thermal/validate_frequency.py workflow/thermal/run_anchor_validation.py
exit 0

git diff --check
exit 0
```

No R1, gem5, anchor experiment, or long-running job was started.  The full
unit suite includes the repository's existing small HotSpot tests.
