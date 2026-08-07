# Final documentation-fix report

## Modified files

- `docs/superpowers/specs/2026-08-07-legacy-instruction-scope-compatibility-design.md`
- `docs/superpowers/plans/2026-08-07-legacy-instruction-scope-compatibility.md`

The operational smoke is now precisely a fixed-only, hash-guarded local
attachment of `fixed_bin/fft/l1d_16kB/l2_128kB` using
`python -m workflow.r2.attach_result --point-dir ...`.  The documents state
that it publishes only fixed attachment-derived `performance.json` and
`pipeline_summary.json`, compares pre/post SHA-256 values for `stats.txt`,
`r2_result.json`, and `status.json`, and does not create pair status or claim
pair completion.  They also defer the full paired sweep and retain the `50+K`
requirement for unequal complete latency-override vectors.

## Validation

Executed:

```bash
git diff --check
rg -n "run_paired_sweep|clip_complete|complete paired|--limit 1|attach_result|50\\+K|legacy-scope-fixed-r2-before" \
  docs/superpowers/specs/2026-08-07-legacy-instruction-scope-compatibility-design.md \
  docs/superpowers/plans/2026-08-07-legacy-instruction-scope-compatibility.md
```

Result: `git diff --check` exited 0.  The targeted inspection found only the
intended attachment command, hash-guard reference, and `50+K` explanation;
there is no post-merge `run_paired_sweep`, `--limit 1`, or `clip_complete`
instruction.

## Commit

Documentation fix commit: `fbb525f960f4caf1051def0255c61cb2cc0bc921`
(`docs: scope legacy compatibility smoke`).

## Operational boundary

No real R1, R2, gem5, HotSpot, or layout flow was run.  No `runs/` or
`gem5_r2/` file was modified.
