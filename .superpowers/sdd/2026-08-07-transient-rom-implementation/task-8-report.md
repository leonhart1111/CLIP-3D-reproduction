# Task 8 report

Status: DONE

Initial implementation commit: `0e69380 feat: add optional transient ROM pipeline`

Files changed:

- `workflow/transient/rom/run_pipeline.py`
- `workflow/run_lifting_pipeline.py`
- `configs/experiments/clip3d_transient_rom_exploratory.json`
- `tests/test_workflow.py`
- `tests/test_transient_rom.py`

Implementation summary:

- Added explicit `--thermal-mode {steady,transient-rom}` dispatch while keeping
  steady mode as the default and lazily importing the optional ROM pipeline.
- Runs or reuses an R2-disabled fixed-bin steady preflight, prepares one set of
  periodic power windows, calibrates or validates a reusable ROM package,
  optimizes without HotSpot calls, derives R2 from the selected layout, and
  performs final real-HotSpot sustainable-frequency validation.
- Writes ROM-owned artifacts below the fixed sibling `OUTPUT/transient_rom`,
  classifies them as non-formal and not paper-equivalent, and binds package and
  output inventories with SHA-256 manifests.
- Supports `--rerun-r2` only as a retry of R2 work that failed before final
  HotSpot, reusing validated power windows, calibration, and optimization.
- Reports the unambiguous transient metrics
  `f_sus_trans_rom_pred_ghz`, `f_sus_trans_hotspot_ghz`,
  `bips1_trans_rom_pred`, and `bips2_trans`; bare `bips2` is forbidden.

Initial TDD and verification evidence:

- Dispatch tests were observed RED before `--thermal-mode` existed and GREEN
  after the opt-in steady/ROM dispatch was implemented.
- End-to-end tool boundaries are mocked; no live HotSpot, R1, R2, gem5, McPAT,
  or CACTI execution was authorized or performed.
- Initial targeted verification passed 135 tests with 2 expected HotSpot skips.

## Review fix round

Fix commit: the commit containing this report.

Review findings addressed:

- The steady pipeline now records the CACTI characterization SHA-256. ROM mode
  requires the steady summary's CACTI artifact path to identify exactly
  `steady_preflight/cacti/cacti_characterization.json` and verifies the current
  file against the recorded hash before power preparation or `build_vector()`.
- Removed `--transient-rom-dir`; ROM output is always the sibling
  `OUTPUT/transient_rom`. `--transient-rom-package-dir` remains available for
  read-only accepted-package reuse.
- `bips2_trans` is always present. It is `null` unless both a successful R2
  result and a final real-HotSpot sustainable frequency exist, and numeric only
  when both measurements exist. Bare `bips2` remains forbidden.
- `--rerun-r2` now refuses a nonempty pre-existing
  `final_hotspot_validation/` with an auditable error explaining that retry is
  limited to failures before final HotSpot.

Fix-round TDD evidence:

- Wrong-path and replaced-content CACTI tests were observed RED by reaching the
  missing-package error, proving the pipeline lacked the provenance gate. Both
  are GREEN and assert that power preparation and R2 vector construction are
  not called.
- The arbitrary-output CLI test was observed RED because the option parsed and
  dispatched; it is GREEN after removing the option and checks argparse exit 2
  plus zero pipeline dispatches.
- The incomplete-summary test was observed RED because `bips2_trans` was
  absent; it is GREEN with an explicit null, while the complete R2 plus final
  HotSpot test remains numeric at 2.7.
- The final-validation retry test was observed RED by reaching steady-preflight
  validation; it is GREEN with an early refusal and downstream call sentinels.

Fix-round verification:

- `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_workflow tests.test_transient_rom -v`
  — PASS: 139 tests, 2 expected live-HotSpot skips.
- `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m py_compile workflow/run_lifting_pipeline.py workflow/transient/rom/run_pipeline.py tests/test_workflow.py tests/test_transient_rom.py`
  — PASS.
- `/home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m json.tool configs/experiments/clip3d_transient_rom_exploratory.json`
  — PASS.
- `git diff --check` — PASS.
- Production scan for `transient-rom-dir`/`transient_rom_dir` — PASS: no
  production or configuration matches; only the CLI rejection test names the
  removed option.

Scope:

- No live HotSpot, R1, R2, gem5, McPAT, CACTI, or EDA tool was run.
- No non-Task-8 source or test file was modified.
