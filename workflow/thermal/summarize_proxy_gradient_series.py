#!/usr/bin/env python3
"""Aggregate frozen-parameter proxy-gradient diagnostics across work points.

This tool intentionally consumes only completed ``gradient_diagnostic.json``
reports.  It never refits proxy parameters, reruns HotSpot, or changes a
report's acceptance boundary.  Labels make the fit/holdout status explicit so
in-fit regression cases cannot be accidentally presented as transfer evidence.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from workflow.common import read_json, write_json


def parse_report_spec(text: str) -> tuple[str, Path]:
    """Parse ``LABEL=PATH`` while keeping evidence labels explicit."""
    if "=" not in text:
        raise argparse.ArgumentTypeError("report must be LABEL=PATH")
    label, path = text.split("=", 1)
    if not label or not path or any(character in label for character in "/\\"):
        raise argparse.ArgumentTypeError("report must be LABEL=PATH")
    return label, Path(path)


def _mean_median(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def _contract_signature(document: dict) -> tuple:
    """Return the report fields that must not differ in one evidence series."""
    variants = document.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("report has no variant metadata")
    normalized_variants = []
    for variant in variants:
        try:
            normalized_variants.append((
                str(variant["name"]), str(variant["proxy_spatial_model"]),
                float(variant["lc_die_side_ratio"]),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("report has invalid variant metadata") from error
    return (
        document.get("config"), document.get("hotspot"),
        document.get("grid_points_per_axis"),
        tuple(document.get("allowed_l2_tiers", [])), tuple(normalized_variants),
    )


def summarize_reports(reports: list[tuple[str, Path]]) -> dict:
    """Summarize reports without averaging per-report sign rates incorrectly."""
    if not reports:
        raise ValueError("at least one report is required")
    labels = [label for label, _ in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("report labels must be unique")

    loaded = []
    expected_variants: tuple[str, ...] | None = None
    expected_contract: tuple | None = None
    for label, path in reports:
        document = read_json(path)
        if document.get("method") != "common-HotSpot-grid equation-(14) gradient diagnostic":
            raise ValueError(f"unsupported report method: {path}")
        evaluations = document.get("evaluations")
        if not isinstance(evaluations, dict) or not evaluations:
            raise ValueError(f"report has no evaluations: {path}")
        variants = tuple(evaluations)
        if expected_variants is None:
            expected_variants = variants
        elif variants != expected_variants:
            raise ValueError("all reports must evaluate the same ordered variants")
        contract = _contract_signature(document)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError("all reports must use the same physical/proxy contract")
        loaded.append((label, path.resolve(), document))

    assert expected_variants is not None
    summaries = {}
    for variant in expected_variants:
        per_report = []
        sign_comparable = 0
        sign_agreement = 0
        candidate_count = 0
        spearman = []
        regrets = []
        active_reports = 0
        observability_keys = (
            "raw_thermal_frequency_term_active",
            "hotspot_frequency_varies",
            "anchored_proxy_frequency_varies",
            "raw_proxy_frequency_varies",
        )
        observability = {
            key: {"observed": 0, "true": 0} for key in observability_keys
        }
        for label, path, document in loaded:
            values = document["evaluations"][variant]
            comparable = int(values["sign_comparable_count"])
            agreement = int(values["sign_agreement_count"])
            if comparable < 0 or agreement < 0 or agreement > comparable:
                raise ValueError(f"invalid sign counts for {variant} in {path}")
            sign_comparable += comparable
            sign_agreement += agreement
            candidate_count += int(values["candidate_count"])
            raw_spearman = values.get("spearman")
            if raw_spearman is not None:
                numeric = float(raw_spearman)
                if not math.isfinite(numeric):
                    raise ValueError(f"non-finite Spearman for {variant} in {path}")
                spearman.append(numeric)
            regret = float(values["proxy_selected"]["selection_regret_c"])
            if not math.isfinite(regret) or regret < 0.0:
                raise ValueError(f"invalid selection regret for {variant} in {path}")
            regrets.append(regret)
            active = bool(values["thermal_frequency_term_active"])
            active_reports += int(active)
            observed_observability = {}
            for key in observability_keys:
                value = values.get(key)
                if value is not None and type(value) is not bool:
                    raise ValueError(f"invalid {key} for {variant} in {path}")
                if type(value) is bool:
                    observability[key]["observed"] += 1
                    observability[key]["true"] += int(value)
                observed_observability[key] = value
            per_report.append({
                "label": label,
                "report": str(path),
                "model": document.get("model"),
                "config": document.get("config"),
                "candidate_count": int(values["candidate_count"]),
                "sign_comparable_count": comparable,
                "sign_agreement_count": agreement,
                "sign_agreement_rate": values.get("sign_agreement_rate"),
                "spearman": raw_spearman,
                "selection_regret_c": regret,
                "thermal_frequency_term_active": active,
                **observed_observability,
            })
        summaries[variant] = {
            "report_count": len(per_report),
            "candidate_count": candidate_count,
            "weighted_sign_comparable_count": sign_comparable,
            "weighted_sign_agreement_count": sign_agreement,
            "weighted_sign_agreement_rate": (
                sign_agreement / sign_comparable if sign_comparable else None
            ),
            "spearman": _mean_median(spearman),
            "selection_regret_c": {
                **_mean_median(regrets), "max": max(regrets),
            },
            "thermal_frequency_term_active_report_count": active_reports,
            "frequency_observability": {
                key: {
                    "observed_report_count": values["observed"],
                    "true_report_count": values["true"],
                }
                for key, values in observability.items()
            },
            "per_report": per_report,
        }

    return {
        "schema_version": 1,
        "method": "frozen-parameter proxy-gradient series summary",
        "purpose": (
            "Aggregate completed same-method diagnostics without refitting. "
            "Labels preserve in-fit versus holdout evidence boundaries."
        ),
        "reports": [
            {
                "label": label, "path": str(path), "model": document.get("model"),
                "config": document.get("config"),
                "grid_points_per_axis": document.get("grid_points_per_axis"),
            }
            for label, path, document in loaded
        ],
        "common_contract": {
            "config": expected_contract[0], "hotspot": expected_contract[1],
            "grid_points_per_axis": expected_contract[2],
            "allowed_l2_tiers": list(expected_contract[3]),
            "variants": [
                {"name": name, "proxy_spatial_model": spatial_model,
                 "lc_die_side_ratio": ratio}
                for name, spatial_model, ratio in expected_contract[4]
            ],
        },
        "variants": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", type=parse_report_spec, required=True,
        help="repeat LABEL=PATH; use labels such as l2-holdout or in-fit-regression",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_reports([(label, path.resolve()) for label, path in args.report])
    write_json(args.output, report)
    for name, values in report["variants"].items():
        print(
            f"{name}: weighted-sign={values['weighted_sign_agreement_rate']}, "
            f"mean-regret={values['selection_regret_c']['mean']:.6f} C"
        )


if __name__ == "__main__":
    main()
