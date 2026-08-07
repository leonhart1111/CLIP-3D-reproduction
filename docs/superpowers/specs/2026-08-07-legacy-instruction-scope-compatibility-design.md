# Legacy instruction-window scope compatibility

## Purpose

Allow historical canonical R1 metadata that predates the explicit
`instruction_window_scope` field to participate in the current R2 attachment
and paired-sweep validation without rewriting R1 evidence or rerunning gem5.

## Scope and invariants

- A missing scope in historical R1 metadata means `cpu0`, matching the
  existing `paper` profile, R1 catalogue, and R2 command construction.
- A scope explicitly present in metadata remains authoritative.  In
  particular, an explicit value other than the expected value must continue
  to be rejected.
- New module-model outputs must record the normalized `cpu0` value both in
  their copied architecture metadata and communication profile.
- Existing R1 metadata, statistics, layout artifacts, and successful R2
  `gem5_r2/` files are immutable inputs.  The recovery path must attach the
  already measured R2 result rather than invoke gem5 again.

## Design

Introduce one small scope-normalization helper in the module-model boundary:
it returns `cpu0` only when the field is absent, and otherwise returns the
explicit field unchanged.  `build_model` uses a normalized metadata copy for
all generated module-model and communication-profile records.

The physical attachment validator applies the same normalization when it
compares a historical module architecture with canonical R1 metadata.  This
accepts a legacy pair where both original records omit the field, while still
rejecting an architecture that explicitly says `all-cores` when canonical R1
means `cpu0`.

## Tests and recovery

Add focused tests for:

1. Building a module model from R1 metadata without the field writes `cpu0`.
2. Local R2 attachment accepts an otherwise coherent legacy R1/module pair
   with the field absent.
3. Local R2 attachment still rejects an explicit incompatible scope.

After the code tests pass, run a fixed-only local attachment smoke against
`fixed_bin/fft/l1d_16kB/l2_128kB` in the existing operational roots.  It uses
the hash-guarded `python -m workflow.r2.attach_result --point-dir ...` command
to validate and attach the already successful fixed-bin R2 cache; it must not
execute gem5 or claim a complete paired status.  Before and after the command,
the SHA-256 values of that fixed point's `gem5_r2/stats.txt`,
`gem5_r2/r2_result.json`, and `gem5_r2/status.json` must match exactly.  The
command publishes only fixed-point attachment-derived outputs such as
`performance.json` and `pipeline_summary.json`.

The full paired sweep is outside this no-gem5 compatibility smoke.  It
continues to follow the `50+K` rule: when a fixed and CLIP point have different
complete latency-override vectors, CLIP requires its own R2 result and cannot
be represented as a reuse of the fixed result.
