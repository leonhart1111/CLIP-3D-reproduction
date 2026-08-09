# Transient ROM Matched Steady Initialization Design

## Goal

Make transient-ROM calibration and final real-HotSpot validation produce scientifically valid output without weakening the full-grid periodic-steady-state (PSS) threshold of `0.01 C`. Preserve the original eight ambient-start PRBS training anchors and the rule that no HotSpot calls occur inside layout optimization.

## Observed Failure and Root Cause

The MATMUL ROM run completed McPAT extraction, all eight training anchors, and the first holdout HotSpot run. The holdout used 239 windows per period, a 2 ms interval, and 20 repeats, corresponding to 9.56 s of physical time. Its adjacent-period full-grid maximum difference was `0.085164 C`, above the required `0.010000 C`.

HotSpot completed successfully and storage was not exhausted. The failure is therefore a valid PSS rejection. The cause is the hard-coded ambient initial condition in both holdout materialization and final sustainable-frequency validation. HotSpot's source confirms that a steady-only invocation averages all rows of the supplied power trace and solves the steady thermal system for that average power. For the failed holdout, this solution peaked near `116.353 C`, whereas the ambient-start trajectory reached only about `47.616 C` after 9.56 s. The transient run was still warming toward its DC equilibrium.

## Considered Approaches

1. **Matched average-power steady preconditioning (selected).** Run a steady-only HotSpot invocation for each holdout or final frequency case using that case's exact layout, frequency-scaled power trace, stack, cooling configuration, grid, and HotSpot binary. Feed the resulting block steady temperature file into the periodic transient invocation with `-init_file`. This costs two additional calibration calls and one additional call per final frequency evaluation, but removes artificial ambient warm-up while retaining a strict PSS test.
2. **Longer ambient warm-up.** Increase the number of period repeats until the trajectory converges. This preserves the ten-call calibration count but may require hundreds of periods, produces very large traces, and wastes time integrating an uninformative startup transient.
3. **Reuse the fixed-layout steady preflight.** This is inexpensive, but its geometry and sometimes frequency differ from the evaluated case. It introduces an uncontrolled physical mismatch and is rejected.

## Architecture and Data Flow

### Training anchors

The eight PRBS training anchors remain unchanged:

1. Materialize the anchor layout and PRBS power trace.
2. Run one ambient-start transient HotSpot call.
3. Record the full-grid trajectory used for POD/state-space identification.

Ambient start is part of the training contract because the identified temperature-rise state begins at zero. These traces are not required to be in PSS.

### Holdout validation

Each of the two real-workload holdouts follows this sequence:

1. Materialize the layout and frequency-scaled periodic power trace once.
2. Run HotSpot without `-o`, with the exact same thermal inputs, to compute `initialization.steady.txt` and its grid steady companion from the trace's average power.
3. Validate the initialization artifacts and bind their hashes and input identities to the calibration package.
4. Run transient HotSpot with `initial_temperature="steady"` and `steady_source=initialization.steady.txt`.
5. Compare every grid cell at the final two period ends. Accept only when the maximum absolute difference is at most `0.01 C`.

Calibration therefore uses exactly 12 HotSpot calls:

- 8 training transient calls;
- 2 holdout steady-initialization calls;
- 2 holdout transient-validation calls.

### Final fixed-bin and CLIP-3D validation

Every frequency evaluated by the real-HotSpot sustainable-frequency search uses the same two-call pair: a case-matched average-power steady initialization followed by a periodic transient run. The frequency search, safety classification, and strict PSS gate remain unchanged. Initialization and transient calls are reported separately so the evidence cannot hide the increased tool budget.

### ROM-side initialization

The reduced model must not remain ambient-start while the reference HotSpot path is preconditioned. For a candidate layout and frequency, let the continuous reduced model be

\[
\dot{x}=Ax+B(l)u(t), \qquad T(t)=T_{amb}+Vx(t).
\]

For one power period with duration-weighted mean input

\[
\bar{u}=\frac{\sum_k \Delta t_k u_k}{\sum_k \Delta t_k},
\]

initialize the reduced temperature-rise state by solving

\[
A x_{ss}=-B(l)\bar{u}.
\]

Then simulate the configured 20 periodic repeats exactly as before and apply the same adjacent-period full-grid PSS test. This matches HotSpot's average-power steady-start semantics while preserving the periodic peaks caused by time-varying power. The solve must be fail-closed: reject non-finite matrices, a condition number above the configured numerical limit, or a normalized residual above a strict documented tolerance. There is no silent fallback to ambient initialization.

## Artifacts and Provenance

Each holdout and final frequency directory records:

- the steady-only command and return code;
- `initialization.steady.txt` and its grid steady companion;
- SHA-256 hashes for layout, scaled power trace, stack, materials, configuration, HotSpot binary, and initialization outputs;
- the subsequent transient command showing `-init_file`;
- full-grid PSS evidence and the last-period peak.

Reusable ROM packages validate these records fail-closed. A legacy ten-call package is not silently treated as a new twelve-call package; it must be recalibrated because the holdout evidence contract changed.

The call-accounting schema distinguishes:

- `training_hotspot_calls = 8`;
- `holdout_initialization_hotspot_calls = 2`;
- `holdout_transient_hotspot_calls = 2`;
- `calibration_hotspot_calls = 12`;
- `final_initialization_hotspot_calls`;
- `final_transient_hotspot_calls`;
- `final_validation_hotspot_calls`, equal to the sum of the preceding two fields.

## Error Handling

Failures are separated into actionable categories:

1. steady-initialization tool failure;
2. missing, malformed, or identity-mismatched initialization artifact;
3. transient HotSpot tool failure;
4. PSS nonconvergence after matched initialization;
5. ROM steady-state solve conditioning or residual failure.

Only category 4 is evidence that the configured 20 periods are physically insufficient. None of these categories is converted to thermal infeasibility, and no partial result is published as a valid paired comparison.

## Testing and Acceptance

Unit and regression tests must first reproduce the current ambient-start failure contract, then prove:

- training anchors still use exactly one ambient-start transient call each;
- holdouts run steady initialization before transient validation;
- final frequency cases use the same paired calls;
- steady and transient inputs are byte/hash bound to the same case;
- package validation requires the 8+2+2 call evidence;
- call accounting remains correct for fresh and reused packages;
- ROM evaluation starts from the checked average-power reduced equilibrium;
- ill-conditioned or high-residual reduced solves fail explicitly;
- existing frequency search, discrete wire-partition selection, R2 gating, and paired-evidence behavior do not regress.

The real MATMUL acceptance run succeeds only if both fixed-bin and CLIP-3D final branches produce full-grid-converged HotSpot evaluations at the unchanged `0.01 C` tolerance, report finite sustainable frequencies, and—when R2 is requested—produce finite `BIPS2_trans`. A normal output is not accepted merely because files exist.

## Scope

This change does not modify R1, McPAT power extraction, HotSpot source code, the eight-anchor training design, the discrete wire partition search, the nonzero `lambda_wire` configuration, or the steady-state paper reproduction path. It only repairs initial-state consistency and evidence accounting in the optional transient-ROM path.
