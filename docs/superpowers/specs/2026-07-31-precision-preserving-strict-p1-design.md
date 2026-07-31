# Precision-Preserving Strict-P1 Revalidation

## Objective

Re-run the strict-P1 HotSpot/R2 parameter study without modifying or
re-running R1.  Correct two measurement-path defects that currently prevent
an evidence-based decision: HotSpot truncates steady-state temperatures to
0.01 K, and power-trace text rounding breaks a physically exact per-cell
`total = dynamic + leakage` invariant.  Make cross-tier-weight selection
consistent with the already-declared spatial-rank acceptance rule.

This work may either produce accepted strict-P1 parameters or a stronger,
fully reproducible negative result.  It must never force a parameter merely
to make the formal configuration promotable.

## Fixed scope and invariants

- Do not edit `configs/gem5/clip_r1.py`, `scripts/run_r1_sweep.py`, or
  `configs/experiments/r1_cache_sweep.json`.
- Do not start, terminate, change, or replace any of the 100 R1 sweep jobs.
  The existing successful exact inputs remain the only inputs:
  `fft`, `matmul`, `stencil`, and `stream`, all at L1D=32kB/L2=512kB.
- Continue to use only direct McPAT Runtime Dynamic plus Subthreshold and
  Gate Leakage, and only local CACTI data.  No power scale, paper-table cache
  value, or manually inserted thermal/wire parameter is permitted.
- Preserve all earlier failed experiments as evidence.  New output roots are
  distinct and contain the binary/source provenance needed to reproduce them.
- Strict P1 remains cores on tier 0, L2 legal only on tier 1,
  `paper-single`, and beta exactly zero with status
  `fixed_unidentifiable_under_p1`.

## Diagnosis being corrected

The current HotSpot source formats both steady and grid steady-state
temperatures with `%.2f` (`temperature_block.c` and `temperature_grid.c`).
The observed location response is only 0.89–1.52 C, so this output
quantisation creates ties and unstable rank statistics.  The simulator itself
is not being changed; only the lossless representation of its already
computed double-precision temperatures is changed.

The generated total, dynamic, and leakage traces each currently use `.12g`.
They are rounded independently.  For example, FFT cell 289 has a parsed
total-minus-components residual of about -6.39e-9 W, which is a text
serialization artefact but exceeds the 1e-9-W invariant threshold.  All three
traces will be written with a round-trip-safe precision instead.

Finally, the existing cross-tier grid search minimises a continuous combined
error while the formal protocol separately demands spatial rank at least 0.8.
The selection must treat the declared rank condition as a feasibility
constraint before comparing continuous score.  This changes selection logic,
not a numerical threshold.

## Design

### 1. High-fidelity HotSpot output

Patch only the local HotSpot output formatting sites used for the steady and
grid-steady files from two-decimal display to a documented high-precision
format (at least 12 significant decimal digits).  Rebuild the existing local
HotSpot binary with its normal Makefile.  Record the pre/post source hash and
rebuilt executable SHA-256 in each precision-study manifest.

No thermal material, geometry, grid dimensions, power input, solver option,
or cooling parameter changes.  The output format is the sole HotSpot change.

### 2. Round-trip-safe raw power traces

Use 17 significant digits when writing total, dynamic, and leakage `.ptrace`
files.  Preserve the existing per-cell equality validation at 1e-9 W, because
the regenerated parsed traces now represent their calculated values without
the earlier formatting loss.  Add a focused test using a value pair that
previously fails at 12 significant digits and passes after round-trip
serialization.

### 3. Feasible-first parameter selection

For each cross-tier-weight candidate in the unchanged inner grid, fit alpha
(beta stays fixed) and calculate the existing validation metrics.  Define
the feasible set as candidates whose non-tied inner spatial Spearman is at
least 0.8 and whose weight is strictly inside `(0, 1)`.  Select the minimum
existing continuous validation score only inside that set.  Record the full
candidate table, the feasibility flag for every candidate, and the reason
when no feasible candidate exists.

The outer independent STREAM 32x32 cases and leave-one-workload-out tests
remain acceptance gates; they cannot be used to choose a candidate.  If no
candidate survives any required gate, the report is rejected and promotion
remains impossible.

### 4. Isolated revalidation evidence

Create a new root such as
`results/parameter_studies/raw_power_strict_20260731_precision/`.  Copy only
the immutable input-manifest hashes or reference the existing R1 directories;
do not regenerate R1.  Recreate the four lifting/HotSpot outputs from the
same R1 data so new manifests reflect the precision-controlled tool.  Re-run:

1. FFT/MATMUL/STENCIL 3x3 training samples at grid 16;
2. STREAM 3x3 independent target samples at grid 32;
3. final training with exactly the three named STREAM external cases;
4. separated dynamic/leakage frequency validation; and
5. the matched 0/1/2/3 wire-cycle R2 series.

The R2 series is independent of the temperature-print precision, but is
recorded under the new study root for a single complete formal artifact set.
Long jobs use one named tmux session, one `flock` lock, and an empty-output
precondition; this is execution control only.

## Acceptance and failure behaviour

Formal promotion is allowed only when the predeclared proxy, frequency, and
four-workload wire rules all pass.  In particular the final proxy must retain
beta's strict-P1 fixed status, select a strictly interior feasible
cross-tier-weight, and pass every spatial-rank and target-grid gate.  Failed
reports are first-class results and remain in the new root.  There is no
fallback that writes alpha, cross-tier weight, or lambda into a formal config.

If precision-preserving revalidation still rejects the proxy, the next work
is a separately approved exploratory response-surface design, not a hidden
change to this strict paper-formula study.

## Verification

- Focused tests prove high-precision trace round trips and feasible-first
  selection rejects an otherwise lower-score rank-infeasible candidate.
- Existing unit suite remains green; tests do not run gem5 or long HotSpot
  workloads.
- A small HotSpot fixture demonstrates values differing below 0.01 K are
  preserved in both steady output forms.
- New study manifests contain command lines, source/binary hashes, R1 input
  hashes, and candidate feasibility diagnostics.
- `git diff --check` and the full unit suite run before any completion claim.
