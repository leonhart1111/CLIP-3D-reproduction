# Five-State Transient Thermal Proxy Design

## Status and scientific boundary

This is an optional, non-formal extension of the CLIP-3D reproduction.  It
does not change the paper-equivalent steady flow.  The optimizer uses no
HotSpot calls; the selected fixed-bin and CLIP-3D layouts continue through the
existing real-HotSpot periodic validation and optional gem5 R2 measurement.

The default backend for `--thermal-mode transient-rom` becomes `five-state`.
The existing POD state-space backend remains available as `pod-rom`.  Its
files, calibration package format, and reports are retained.

## Model

The five receivers are `core0`, `core1`, `core2`, `core3`, and `shared_l2`.
For receiver \(i\), layout \(p\), frequency scale \(s=f/f_0\), and power
window \(k\), the spatial equilibrium proxy is

\[
T^{\mathrm{eq}}_{i,k}(p,s)=T_{\mathrm{amb}}+
r_{\mathrm{global}}P_{\mathrm{total},k}(s)+
\alpha\sum_j K_{ij}(p)P_{j,k}(s)+
\beta P_{\mathrm{bottom},k}(s).
\]

The area-quadrature spatial kernel and same-tier/cross-tier factor are shared
with the existing steady proxy:

\[
K_{ij}(p)=\frac{w_{\mathrm{tier}}(i,j)}
{\sqrt{1+(d_{ij}(p)/L)^2}}.
\]

Each receiver has one first-order inertia state:

\[
T_{i,k+1}=a_{i,k}(s)T_{i,k}+
[1-a_{i,k}(s)]T^{\mathrm{eq}}_{i,k}(p,s),
\qquad
a_{i,k}(s)=\exp[-\Delta t_k^0/(s\tau_i)].
\]

Dynamic power scales with frequency and leakage is held fixed:

\[
P_{j,k}(s)=P^{\mathrm{leak}}_{j,k}+sP^{\mathrm{dyn}}_{j,k}.
\]

The model uses the exact scalar periodic steady state for each receiver.  If
one complete ROI maps a receiver state as \(T_N=A_iT_0+B_i\), then

\[
T_{i,0}^{\mathrm{PSS}}=B_i/(1-A_i).
\]

One ROI traversal starting at this state yields
\(T_{\mathrm{peak}}=\max_{i,k}T_{i,k}\).  The existing bracket-refinement
frequency search finds the largest evaluated safe frequency and records all
evaluations.  The objective remains

\[
-\mathrm{IPC1}f_{\mathrm{sus,trans}}+
\lambda_{\mathrm{wire}}\mathrm{IPC1}\tau_{\mathrm{wire}}.
\]

## Receiver/source aggregation

All window modules carrying `core=0..3` are aggregated into the corresponding
core source.  The sole `kind=l2` module is the L2 source.  Remaining modules,
such as the NoC, contribute to total and bottom-tier power but are not a local
source state.  Receiver geometry is the area-weighted centroid and bounding
rectangle of each core's placed modules; the L2 uses its own rectangle.

Every source keeps dynamic and leakage components separate.  A source's tier
comes from the supplied candidate layout.  Geometry and module names must
match the power-window evidence; malformed or missing power components are
rejected rather than silently inferred.

## Configuration contract

`transient_rom.backend` accepts:

- `five-state` (default): uses the closed-form proxy;
- `pod-rom`: preserves the existing matrix-ROM calibration and optimizer.

The five-state backend consumes:

```json
"five_state": {
  "tau_core_s": 0.166,
  "tau_l2_s": 0.166,
  "spatial_model": "area-quadrature",
  "quadrature_order": 2,
  "parameter_status": "provisional"
}
```

Both time constants must be finite and positive.  `parameter_status` must be
`provisional` or `measured`; results record it verbatim.  The initial values
are explicit provisional experiment controls, not claimed measurements.  A
later identification experiment may replace them without changing code.

The spatial coefficients are read from the existing `layout_optimizer`
section.  Frequency bounds, ambient temperature, and safety threshold are
read from `frequency`.  No duplicate scientific parameters are introduced.

## Pipeline behavior

The common prefix remains:

1. validate canonical R1 and fixed-bin steady preflight;
2. reuse or generate periodic-statistics R1;
3. prepare one set of windowed McPAT power records.

For `five-state`, no POD calibration package is created or required.  The
five-state optimizer directly searches legal L2 layouts using the power
windows.  For `pod-rom`, the existing calibration/holdout/package path is
unchanged.  Both backends then use the existing real-HotSpot final periodic
validation, integer-cycle latency construction, paired fixed/CLIP-3D report,
and optional R2 execution.

Reports use backend-neutral predicted fields while retaining legacy POD field
aliases where required for backward compatibility.  Every report records the
backend, equations, parameters, zero optimizer HotSpot calls, and
`paper_equivalent=false`.

## Validation

Unit tests must establish:

- constant equilibrium power converges to the equilibrium temperature;
- zero dynamic power makes frequency affect duration but not window power;
- dynamic power scales by \(s\) and duration by \(1/s\);
- periodic initial state closes exactly after one period;
- changing L2 location changes the receiver equilibrium vector;
- hotspot identity can move between core receivers;
- configuration omission selects `five-state`, while `pod-rom` remains valid;
- optimizer output contains no HotSpot call and records provisional status;
- invalid time constants and malformed power windows are rejected.

An integration test uses synthetic modules/layout/windows and a small discrete
partition grid.  Existing transient-ROM and steady-flow regressions must remain
green.  Scientific acceptance still depends on later real-HotSpot comparison;
unit-test success alone does not validate the provisional time constants.
