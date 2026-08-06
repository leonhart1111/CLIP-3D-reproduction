# HotSpot Temperature Precision and Sampling-Report Design

## Goal

Remove the 0.01 K output quantization from the steady and transient HotSpot
temperature path, and make every live transient report describe the sampling
interval that was actually requested. The change must not alter gem5 R1,
windowed McPAT, layout optimization, R2, or HotSpot's thermal equations.

## Scope

The locally downloaded HotSpot source will print computed temperatures with
exactly six digits after the decimal point in all machine-readable temperature
files used by this project:

- the block/grid transient trace selected by `-o`;
- detailed-grid steady and transient dumps;
- named steady temperature files used for transient initialization; and
- block-model temperature dumps, so the project does not silently regress if
  that model is used later.

Human-readable diagnostic output such as ambient temperature and package
thickness is outside this change because it is not parsed as a temperature
trajectory.

The complete Python steady/transient lifting path will print reported
temperatures with at least six decimal places. Temperature-bearing CSV fields
in the lifting, transient-comparison, and thermal-proxy reports will use six
decimal places for Celsius values. JSON values remain JSON numbers;
fixed-width decimal formatting is a textual presentation concern, not part of
JSON number semantics.

Every live limitation string will interpolate the actual `sample_ms` value.
The command-line default remains 10 ms for backward compatibility, while a
run requested with `--sample-ms 2` must report 2 ms everywhere. Current user
documentation will describe a configurable interval and show the present 2 ms
example. Historical design and plan documents retain their original 10 ms
statements because those are provenance records of the earlier experiment.

## Reproducible Vendor Modification

`tools/src/` is ignored by Git. Editing the downloaded HotSpot checkout alone
would therefore create an unreproducible local dependency. The repository will
contain a versioned patch under `patches/hotspot/`, and the same patch will be
applied to the shared local HotSpot source before rebuilding the binary.
Download/build documentation will explain how to apply the patch. The patch
must be safe to detect as already applied so repeated setup does not invite an
accidental second application.

## Data Flow

HotSpot continues to solve temperatures internally as `double`. Only the C
format strings used at file-output boundaries change from two to six decimal
places. Python then parses those six-decimal Kelvin tokens, converts them to
Celsius, writes six-decimal CSV presentation values, and preserves numeric
values in JSON summaries. No padding with artificial zeroes is allowed as a
substitute for rebuilding HotSpot.

The transient report generators receive `sample_ms` through their existing
interfaces. Limitation text is constructed from that value at the point where
each summary dictionary is created; no new global configuration value is
introduced.

## Compatibility and Existing Results

Existing two-decimal result directories remain historical artifacts and are
not rewritten in place. Corrected results require rerunning the steady HotSpot
initialization and the two transient HotSpot layout solves with the rebuilt
binary. Existing 2 ms gem5 periodic statistics and windowed McPAT power windows
are reusable, so no multi-hour R1 rerun is required.

## Tests and Verification

Tests will be added before implementation to establish these contracts:

1. a 2 ms transient summary contains a 2 ms sampling limitation and contains
   no stale 10 ms limitation;
2. the dual-layout and comparison summaries follow the same rule;
3. steady and transient CLI temperature presentation uses six digits after
   the decimal point;
4. temperature-bearing CSV presentation uses six digits after the decimal
   point throughout the lifting, comparison, and proxy-report paths;
5. the tracked HotSpot patch changes each machine-readable temperature format
   to six decimal places; and
6. a rebuilt HotSpot executable emits six-decimal temperature tokens in a
   small real invocation.

Verification consists of the focused transient test module, the complete
workflow test suite, rebuilding `tools/src/hotspot/hotspot`, and a small
HotSpot output smoke test. The implementation is not considered complete if
only Python tests pass while the real binary still emits two-decimal values.

## Non-Goals

This change does not implement temperature-dependent leakage, a
temperature/leakage/DVFS feedback loop, variable-duration final HotSpot steps,
new peak-tie semantics, or a new default sampling interval. Those remain
separate modeling and reporting improvements.
