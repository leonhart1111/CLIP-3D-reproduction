# HotSpot temperature-precision final fix report

Base reviewed: `eef5e05` (`docs: parameterize transient sampling guidance`).

## Review findings and resolutions

1. **Patch procedure and rebuild:** `docs/DOWNLOAD_TOOLS.md` now uses a
   three-way `patch` procedure. A successful reverse dry-run recognizes an
   already-patched source; otherwise a successful forward dry-run is required
   before applying; failure of both aborts before `make`. The build command is
   outside the conditional, so it runs for both valid source states and cannot
   run after a failed application.
2. **Proxy-calibration CLI precision:** `workflow/thermal/calibrate_proxy.py`
   now sends `validation["rmse_c"]` through the shared
   `format_temperature_c` formatter. A captured-stdout test requires the live
   CLI text `validation RMSE=12.345679 C`.
3. **Tracked patch applicability:** `tests/test_transient.py` now creates a
   temporary source fixture from the patch's exact recorded preimage hunk lines
   at their recorded line numbers. It verifies `patch -p1 --dry-run`, applies
   the patch, then verifies reverse dry-run. This has no network or vendor
   snapshot dependency.

## TDD evidence

RED command:

```bash
python -m unittest tests.test_workflow.WorkflowTests.test_proxy_calibration_cli_prints_temperature_rmse_with_six_decimals tests.test_transient.HotSpotPrecisionPatchTests.test_hotspot_patch_applies_and_reverse_checks_on_recorded_hunk_fixture -v
```

Output: one expected failure and one pass. The CLI printed
`validation RMSE=12.3457 C`; the test required `validation RMSE=12.345679 C`.
The new patch applicability/round-trip test passed because the existing patch
artifact already applied correctly; it was added to cover the previously
untested contract.

GREEN command:

```bash
python -m unittest tests.test_workflow.WorkflowTests.test_proxy_calibration_cli_prints_temperature_rmse_with_six_decimals tests.test_transient.HotSpotPrecisionPatchTests.test_hotspot_patch_applies_and_reverse_checks_on_recorded_hunk_fixture tests.test_transient.HotSpotPrecisionPatchTests.test_hotspot_patch_tracks_six_decimal_temperature_outputs -v
```

Output: `Ran 3 tests ... OK`.

Local patched-source check and rebuild:

```bash
patch -d tools/src/hotspot -p1 --dry-run -R < patches/hotspot/0001-six-decimal-temperature-output.patch
make -C tools/src/hotspot hotspot
```

Output: all three source files checked successfully; `hotspot` was up to date.

Required full-suite command:

```bash
python -m unittest tests.test_transient tests.test_workflow -v
```

Output: `Ran 108 tests in 1.706s` and `OK`.

## Files changed

- `docs/DOWNLOAD_TOOLS.md`
- `workflow/thermal/calibrate_proxy.py`
- `tests/test_workflow.py`
- `tests/test_transient.py`
- `.superpowers/sdd/2026-08-05-hotspot-temperature-precision/final-fix-report.md`

## Commit

`fix: close HotSpot temperature precision review gaps` (this report is included
in that single review-fix commit).

## Self-review

- `git diff --check` passed.
- The precision scan found no live `RMSE`/temperature `.0f` through `.5f`
  formatter matching the targeted output patterns.
- The existing patch contract test still proves exactly eleven textual format
  changes (2 + 4 + 5) and the patch itself is unchanged, so no HotSpot equation
  or tolerance changed.
- The diff is limited to the three review findings and their regressions; it
  does not touch R1, McPAT, R2, layout, or historical results.

## Residual concern

The offline fixture proves that the tracked `-p1` patch applies to its recorded
hunk preimages and round-trips, but it cannot establish that every arbitrary
future HotSpot download matches beyond those hunks. The documented source
archive checksum remains the compatibility gate; the documented three-way
dry-run procedure aborts incompatible or partially patched trees.
