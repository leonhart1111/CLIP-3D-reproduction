# Operational Raw-Power P1 Parameter Profile Design

## Objective

Provide a usable, evidence-derived CLIP-3D parameter configuration for completing the full experimental workflow without modifying HotSpot source code or representing the outcome as a strict/formal reproduction result.

## Observed evidence

The preserved strict-P1 study at `results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json` used only local McPAT power, local CACTI geometry, L2 tier 1, beta fixed to zero, and the strict 3×3 FFT/MATMUL/STENCIL training design.

Its held-out score selected the interior cross-tier weight `0.995`. Refitting alpha with that selected weight gives:

| Parameter | Value | Source |
| --- | ---: | --- |
| `alpha` | `1.5643788695171585` | `fit.parameters.alpha` |
| `beta` | `0.0` | fixed because it is unidentifiable under strict P1 |
| `cross_tier_weight` | `0.995` | held-out continuous-score minimizer |
| `lambda_wire` | not set by this profile | must come from the separate matched R2 study; retain zero until that study accepts a value |

The fitted proxy improved held-out temperature RMSE and centered spatial RMSE relative to defaults, but it does not meet the strict rank rule: internal spatial Spearman is `0.6571428571428573`, minimum leave-one-workload-out rank is `0.28571428571428575`, and independent STREAM target rank is `0.5`. Thus the strict result remains rejected.

## Chosen approach

Add a separate **operational** acceptance policy rather than weakening the strict policy.

- The existing strict report, hard-coded `0.8` requirements, and formal promotion path stay unchanged.
- A new operational config uses the measured alpha/beta/cross-tier map above and has `formal_validation.strict_p1: false`, `accepted: false`, and an explicit prohibition on formal-promotion claims.
- A new config section, `operational_validation`, makes the non-strict decision rules declarative:
  - `minimum_validation_spatial_spearman: 0.5`;
  - `minimum_external_target_spatial_spearman: 0.5`;
  - require lower validation RMSE and lower centered spatial RMSE than the default proxy;
  - require lower external target RMSE and lower external target centered spatial RMSE than default;
  - require an interior cross-tier weight;
  - record leave-one-workload-out ranks but do not make them a release gate.
- The calibration script reads this policy only for a config whose `mode` is `operational`. It writes a separate `operational_recommendation`, including policy, checks, values, and a clear non-formal action string. The existing `recommendation` remains the strict result whenever strict P1 is requested.
- The full lifting/R2 workflow may consume the operational config. Every generated run config and summary must carry the mode and parameter provenance.

## HotSpot source constraint

No file under `tools/src/hotspot/` will remain tracked or edited by this work. The prior precision commit will be corrected so that the downloaded local source returns to its upstream two-decimal formatting and is again ignored by Git. The project-owned Python writers retain `.17g` raw-power serialization because that prevents an artificial failure of the existing `total = dynamic + leakage` trace check without changing a computed physical quantity.

## Safety and reporting

Operational acceptance is a workflow-enablement decision, not a claim that the paper's undisclosed model parameters were reproduced. Documentation and JSON use the words `operational` and `non-formal`; they must not use `formal`, `strict accepted`, or `paper-equivalent` for this profile.

The strict report and current failed checks remain immutable evidence. A later strict revalidation can be compared against this operational profile, but may not overwrite it.

## Verification

1. Tests prove the strict policy stays at `0.8` and remains rejected on the preserved report.
2. Tests prove the operational policy accepts boundary Spearman `0.5`, records the leave-one-out failure as a diagnostic, and rejects lower rank or non-improving RMSE.
3. Tests prove the operational config is accepted by normal pipeline validation but rejected by formal-promotion validation.
4. Tests prove no HotSpot source file is tracked and no project test requires a HotSpot source edit.
5. The raw-power serialization regression test and full Python test suite pass.
6. A one-point fixed-bin and one-point `clip3d` pipeline run using the operational config complete before launching the remaining full experiments.

