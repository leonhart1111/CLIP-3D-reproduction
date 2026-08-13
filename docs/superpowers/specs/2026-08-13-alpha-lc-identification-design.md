# Alpha and Thermal-Length Identification Design

## Objective

Identify a reproducible strict-P1 steady-state thermal proxy from the local
CACTI/McPAT/HotSpot toolchain.  Paper temperatures, paper areas, and desired
BIPS improvements are not calibration targets.  The campaign ends after
estimating `alpha` and the planar thermal characteristic length `lc_mm`, with
held-out validation and uncertainty estimates.

This phase reuses completed gem5 R1 results.  It must not edit, delete, or
rerun R1, and it does not perform the later gem5 R2/lambda experiment.

## Evidence boundary

The reference truth is the configured local toolchain:

- local CACTI supplies cache area and timing;
- unscaled McPAT supplies module geometry and raw dynamic/leakage power;
- HotSpot supplies steady-state temperatures for explicit package settings;
- gem5 R1 supplies workload statistics already present on disk.

The resulting parameters are valid for this declared toolchain and package
scenario.  They are not claimed to be silicon-measurement calibration.

## Fixed physical contract

Before sampling, every model must pass the existing physical-alignment
contract:

- no global die/module area scaling;
- one cache characterization identity is shared by CACTI, McPAT, modules, and
  R2 metadata;
- raw McPAT Runtime Dynamic and leakage values have no calibration multiplier;
- `local_resistance_scale` is exactly 1.0;
- layer thickness, material resistivity, ambient temperature, cooling
  resistance, HotSpot version, grid size, and utilization are recorded in a
  content-addressed experiment manifest.

Changing any of these fields creates a different calibration campaign rather
than silently reusing samples.

## Why beta is excluded in this phase

For strict P1, the L2 tier, all fixed-module tiers, and module powers are held
constant while only L2 `(x, y)` changes.  Therefore total power and bottom-tier
power are constant within a work point.  For reference layout `x0`:

```text
Delta T_HS(x) = Tmax_HS(x) - Tmax_HS(x0)
```

The proxy is fitted in difference form:

```text
Delta T_hat(x) = alpha * [H(x; Lc, wcross) - H(x0; Lc, wcross)]
```

Consequently ambient temperature, `Rconv * Ptotal`, and `beta * Pbottom`
cancel.  This phase neither asserts `beta = 0` nor attempts to estimate an
unidentifiable parameter.  A future tier-swap experiment is required for
beta.

## Spatial proxy

For receiver sample point `i` and source sample point `j`:

```text
K(dij; Lc) = 1 / sqrt(1 + (dij / Lc)^2)
H(x; Lc, wcross) = max_i sum_j Pj * K(dij; Lc) * w(zi, zj)
```

where `w = 1` for the same tier and `w = wcross` for different tiers.  Module
rectangles use area quadrature; they must not be collapsed to center points.

The existing `side / 2` characteristic length is a historical assumption and
must not be used as fitted evidence.  The implementation adds explicit
`lc_mm` plumbing and provenance.

## Cross-tier coupling treatment

`wcross` is not jointly optimized with alpha and Lc in the first formal fit.
Jointly fitting all three from ordinary workload maps can be ill-conditioned.
Instead, matched unit-power HotSpot experiments estimate a fixed cross-tier
coupling ratio:

1. apply one watt to a source rectangle with all other sources at zero;
2. measure temperature rise at matched receiver regions on the same tier and
   the other tier;
3. repeat at center, edge, corner, and near-core positions on small, medium,
   and large geometries;
4. use a robust median of normalized cross-tier/same-tier response ratios;
5. bootstrap cases to report a 95% interval.

If the ratio is not stable across position and geometry, the campaign records
that the scalar cross-tier model is inadequate and does not promote formal
alpha/Lc values.

## Experiment 1: HotSpot grid convergence

Use six layouts on the medium MATMUL 64 kB / 512 kB point: four legal corner
placements, center, and the placement closest to the hottest core.  Run the
same cases with grid sizes 32, 64, and 128.

The smallest grid is accepted when, for every case:

```text
abs(Tmax(grid) - Tmax(128)) <= 0.10 C
```

and layout ordering is unchanged relative to grid 128.  All subsequent cases
use the accepted grid.  Failure at grid 64 promotes grid 128; failure at grid
128 stops the campaign and requests a finer convergence study.

## Experiment 2: unit-power response

Use three architecture anchors:

- small: L1D 16 kB, L2 128 kB;
- medium: L1D 64 kB, L2 512 kB;
- large: L1D 128 kB, L2 2048 kB.

For core-like and L2-like source rectangles, sample center, four edge/corner
regions, and near-core legal locations on both tiers.  Generate at least 72
matched responses.  Each case records the full HotSpot temperature field, not
only Tmax.

Unit-power inputs are system-identification stimuli.  They never overwrite
the raw McPAT power model used in final workload validation.

## Experiment 3: real-power spatial campaign

Use all five workloads:

- FFT;
- Cholesky;
- MATMUL;
- STENCIL;
- STREAM.

Use nine size anchors covering the Cartesian product:

```text
L1D in {16 kB, 64 kB, 128 kB}
L2  in {128 kB, 512 kB, 2048 kB}
```

For every one of the 45 workload/architecture points, generate 13
deterministic legal L2 placements:

- center;
- four corner-directed placements;
- four edge-midpoint-directed placements;
- four placements nearest the four core rectangles.

Duplicate placements produced by geometry constraints are removed and
replaced by the next deterministic legal candidate.  The campaign therefore
targets 585 steady HotSpot cases and requires at least 11 distinct placements
per work point.  The fixed-bin placement is the reference `x0` when legal;
otherwise the manifest records the deterministic fallback reference.

Model generation reuses completed R1 and reruns only the corrected local
CACTI/McPAT conversion needed to produce unscaled modules.  HotSpot cases are
parallel and independently reusable by content identity.

## Fitting method

For fixed candidate `Lc`, compute each sample feature:

```text
q_n(Lc) = H(x_n; Lc, wcross) - H(x0_n; Lc, wcross)
```

Fit nonnegative alpha with robust Huber loss:

```text
minimize_alpha>=0 sum_n Huber(alpha * q_n(Lc) - DeltaT_HS_n)
```

Search `Lc` in log space over a geometry-relative range from 0.02 to 4.0 times
the die side.  Refine around the best interval with bounded scalar
optimization.  Parameters touching a boundary are rejected rather than
promoted.

Bootstrap whole work points, not individual neighboring placements, for 1000
resamples.  Report median estimates and 95% confidence intervals for alpha
and Lc.

## Data splits

Random per-placement splitting is prohibited.  Validation has three levels:

1. spatial holdout: per work point, fit on center/edge/core-near cases and
   hold out designated corner-directed cases;
2. architecture holdout: rotate whole size combinations out of fitting;
3. leave-one-workload-out: fit on four workloads and evaluate the fifth.

The final parameters are fitted once using all non-final-test groups only
after bounds and model form are selected.  A locked final test group is read
only after parameter freezing.

## Acceptance criteria

Formal alpha/Lc promotion requires all of the following:

- grid convergence passes;
- unit-response `wcross` has a finite 95% interval and is not on an imposed
  bound;
- active-parameter Jacobian has rank two with acceptable conditioning;
- aggregate held-out Delta-T MAE <= 0.50 C;
- aggregate held-out Delta-T RMSE <= 1.00 C;
- aggregate held-out spatial Spearman >= 0.80;
- every leave-one-workload-out Spearman >= 0.70;
- median HotSpot selection regret <= 0.50 C;
- 95th-percentile selection regret <= 1.00 C;
- neither alpha nor Lc touches a search boundary;
- bootstrap intervals are finite and reported;
- results are reproducible from the frozen manifest.

Failure is a valid outcome.  It means the scalar kernel is structurally
insufficient; acceptance thresholds must not be relaxed after viewing test
results.

## Outputs

The campaign writes a new result root without deleting previous studies:

```text
results/parameter_studies/unscaled_alpha_lc_<timestamp>/
  campaign_manifest.json
  grid_convergence/
  unit_response/
  real_power_cases/
  samples.csv
  fit_report.json
  cross_validation.json
  bootstrap.json
  acceptance.json
  plots/
```

Every report distinguishes measured HotSpot outputs, derived features,
fitted values, held-out metrics, and acceptance status.  Formal configuration
files are updated only by a separate promotion command after acceptance.

## Non-goals

This phase does not:

- fit beta;
- fit lambda_wire;
- run gem5 R2;
- rerun gem5 R1;
- calibrate McPAT power multipliers;
- match paper temperature, area, or BIPS values;
- alter the transient ROM flow.
