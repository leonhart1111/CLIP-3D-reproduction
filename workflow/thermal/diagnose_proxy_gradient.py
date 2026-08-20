#!/usr/bin/env python3
"""Probe equation-(14) thermal-gradient variants against a common HotSpot grid.

This diagnostic deliberately separates a placement heuristic from a detailed
thermal model.  It materializes one fixed-bin reference and a deterministic
grid of legal L2 positions using the *same* HotSpot contract, then compares
the candidates' relative temperatures under one or more equation-(14)
variants.  It is not a parameter-fitting or paper-promotion mechanism.

The primary outputs are directional: sign agreement against the fixed-bin
reference, rank correlations over a common set of positions, and the HotSpot
selection regret of each proxy's coolest candidate.  Absolute proxy
temperature is anchored to the fixed-bin HotSpot temperature solely to make
equation (13)'s thermal-frequency regime observable; the anchor adds the same
constant to every candidate and cannot change a variant's ordering.
"""

from __future__ import annotations

import argparse
import fcntl
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import baseline_layout
from workflow.floorplan.optimize_layout import proxy_temperature
from workflow.thermal.calibrate_proxy import candidate_layouts, run_one
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT
from workflow.thermal.sustainable_frequency import closed_form_frequency


@dataclass(frozen=True)
class ProxyVariant:
    """One equation-(14) geometry choice, holding fitted weights constant."""

    name: str
    spatial_model: str
    lc_die_side_ratio: float


OUTPUT_LOCK_NAME = ".proxy_gradient.lock"


def parse_variant(text: str) -> ProxyVariant:
    """Parse ``NAME=MODEL,LC_DIE_SIDE_RATIO`` without hidden defaults."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "variant must be NAME=SPATIAL_MODEL,LC_DIE_SIDE_RATIO"
        )
    name, raw = text.split("=", 1)
    fields = raw.split(",")
    if not name or any(character in name for character in "/\\") or len(fields) != 2:
        raise argparse.ArgumentTypeError(
            "variant must be NAME=SPATIAL_MODEL,LC_DIE_SIDE_RATIO"
        )
    spatial_model = fields[0]
    if spatial_model not in ("center", "area-quadrature"):
        raise argparse.ArgumentTypeError(
            "variant spatial model must be center or area-quadrature"
        )
    try:
        ratio = float(fields[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("variant Lc ratio must be numeric") from error
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise argparse.ArgumentTypeError("variant Lc ratio must be finite and positive")
    return ProxyVariant(name, spatial_model, ratio)


def prepare_output_directory(output_dir: Path, resume: bool) -> None:
    """Create an output directory, retaining matching completed probes on resume."""
    entries = (
        entry for entry in output_dir.iterdir()
        if entry.name != OUTPUT_LOCK_NAME
    ) if output_dir.exists() else ()
    if output_dir.exists() and any(entries) and not resume:
        raise ValueError(
            "output directory must be new or empty; pass --resume to reuse "
            f"matching completed HotSpot probes: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def exclusive_output_directory(output_dir: Path):
    """Prevent two ``--resume`` drivers from deleting each other's partial case.

    ``run_one`` deliberately removes a non-reusable, incomplete case before it
    materializes it again.  That is correct after interruption, but destructive
    if a second diagnostic process operates on the same output tree while a
    HotSpot child still owns that directory.  The lock covers the complete
    probe, not just file preparation, and is released automatically on exit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / OUTPUT_LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another proxy-gradient diagnostic already owns output directory "
                f"{output_dir}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _frequency(temperature_c: float, model: dict, config: dict) -> dict:
    frequency = config["frequency"]
    value, state, unconstrained = closed_form_frequency(
        float(temperature_c), float(model["gamma"]),
        float(frequency["f0_ghz"]), float(frequency["fmin_ghz"]),
        float(frequency["tsafe_c"]), float(frequency["ambient_c"]),
    )
    return {
        "sustainable_frequency_ghz": value,
        "state": state,
        "unclamped_frequency_ghz": unconstrained,
    }


def proxy_for_layout(layout: dict, config: dict, variant: ProxyVariant) -> float:
    """Evaluate one proxy variant on a materialized module layout."""
    optimizer = config["layout_optimizer"]
    physical = config["physical"]
    frequency = config["frequency"]
    return proxy_temperature(
        layout["modules"], float(layout["die_width_mm"]),
        float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
        float(optimizer["alpha"]), float(optimizer["beta"]),
        float(optimizer["cross_tier_weight"]), variant.spatial_model,
        int(optimizer.get("proxy_quadrature_order", 2)),
        float(layout["die_width_mm"]) * variant.lc_die_side_ratio,
    )


def _correlation(first: list[float], second: list[float], kind: str) -> float | None:
    if len(first) < 2 or math.isclose(max(first), min(first)) \
            or math.isclose(max(second), min(second)):
        return None
    if kind == "spearman":
        def ranks(values: list[float]) -> list[float]:
            result = [0.0] * len(values)
            ordered = sorted(range(len(values)), key=lambda index: values[index])
            start = 0
            while start < len(ordered):
                end = start + 1
                while end < len(ordered) and math.isclose(
                        values[ordered[start]], values[ordered[end]],
                        rel_tol=0.0, abs_tol=1.0e-12):
                    end += 1
                average = (start + 1 + end) / 2.0
                for index in ordered[start:end]:
                    result[index] = average
                start = end
            return result

        left, right = ranks(first), ranks(second)
        left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
        numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        return numerator / denominator if denominator > 0.0 else None
    concordant = discordant = first_ties = second_ties = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            delta_first = first[left] - first[right]
            delta_second = second[left] - second[right]
            tied_first = math.isclose(delta_first, 0.0, abs_tol=1.0e-12)
            tied_second = math.isclose(delta_second, 0.0, abs_tol=1.0e-12)
            if tied_first and not tied_second:
                first_ties += 1
            elif tied_second and not tied_first:
                second_ties += 1
            elif not tied_first and not tied_second:
                if delta_first * delta_second > 0.0:
                    concordant += 1
                else:
                    discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + first_ties)
        * (concordant + discordant + second_ties)
    )
    return (concordant - discordant) / denominator if denominator > 0.0 else None


def summarize_variant(name: str, records: list[dict], anchor: dict,
                      sign_tolerance_c: float,
                      tsafe_c: float | None = None) -> dict:
    """Summarize rank, signs, frequency regime, and selection regret.

    ``records`` must contain actual HotSpot temperatures and this variant's raw
    proxy temperatures.  Keeping this pure lets tests cover the scientific
    metrics without invoking HotSpot.
    """
    if not records:
        raise ValueError("cannot summarize an empty probe grid")
    if sign_tolerance_c < 0.0:
        raise ValueError("sign tolerance must be non-negative")
    if tsafe_c is not None and not math.isfinite(float(tsafe_c)):
        raise ValueError("tsafe_c must be finite when supplied")
    raw_anchor = float(anchor["variants"][name]["raw_proxy_tmax_c"])
    hotspot_anchor = float(anchor["tmax_c"])
    actual_deltas = [float(record["tmax_c"]) - hotspot_anchor for record in records]
    proxy_deltas = [float(record["variants"][name]["raw_proxy_tmax_c"]) - raw_anchor
                    for record in records]
    considered = [
        (actual, predicted) for actual, predicted in zip(actual_deltas, proxy_deltas)
        if abs(actual) > sign_tolerance_c
    ]
    signs = [
        abs(predicted) > 1.0e-12 and (actual > 0.0) == (predicted > 0.0)
        for actual, predicted in considered
    ]
    actual_values = [float(record["tmax_c"]) for record in records]
    raw_values = [float(record["variants"][name]["raw_proxy_tmax_c"])
                  for record in records]
    proxy_best = min(range(len(records)), key=lambda index: (
        raw_values[index], records[index]["row"], records[index]["column"]
    ))
    hotspot_best = min(range(len(records)), key=lambda index: (
        actual_values[index], records[index]["row"], records[index]["column"]
    ))
    anchor_frequency = anchor["variants"][name]["anchored_frequency"]
    actual_states = [
        record.get("hotspot_frequency", {}).get("state") for record in records
    ]
    proxy_states = [
        record["variants"][name].get("anchored_frequency", {}).get("state")
        for record in records
    ]
    raw_proxy_states = [
        record["variants"][name].get("raw_frequency", {}).get("state")
        for record in records
    ]
    state_pairs = [
        (actual, proxy) for actual, proxy in zip(actual_states, proxy_states)
        if isinstance(actual, str) and isinstance(proxy, str)
    ]
    raw_state_pairs = [
        (actual, proxy) for actual, proxy in zip(actual_states, raw_proxy_states)
        if isinstance(actual, str) and isinstance(proxy, str)
    ]
    raw_anchor_state = anchor["variants"][name].get("raw_frequency", {}).get("state")

    def count_states(states: list[object]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in states:
            if isinstance(state, str):
                counts[state] = counts.get(state, 0) + 1
        return counts

    def frequency_values(entries: list[dict], field: str) -> list[float]:
        values = []
        for entry in entries:
            frequency = entry.get(field, {})
            value = frequency.get("sustainable_frequency_ghz") \
                if isinstance(frequency, dict) else None
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))):
                values.append(float(value))
        return values

    hotspot_frequencies = frequency_values(records, "hotspot_frequency")
    anchored_frequencies = frequency_values(
        [record["variants"][name] for record in records], "anchored_frequency"
    )
    raw_frequencies = frequency_values(
        [record["variants"][name] for record in records], "raw_frequency"
    )

    def frequency_range(values: list[float]) -> list[float] | None:
        return [min(values), max(values)] if values else None

    def frequency_varies(values: list[float]) -> bool:
        return len(values) > 1 and max(values) - min(values) > 1.0e-12

    result = {
        "candidate_count": len(records),
        "hotspot_delta_range_c": [min(actual_deltas), max(actual_deltas)],
        "proxy_delta_range_c": [min(proxy_deltas), max(proxy_deltas)],
        "delta_mae_c": sum(abs(actual - predicted) for actual, predicted in
                           zip(actual_deltas, proxy_deltas)) / len(records),
        "spearman": _correlation(actual_values, raw_values, "spearman"),
        "kendall_tau": _correlation(actual_values, raw_values, "kendall"),
        "sign_tolerance_c": sign_tolerance_c,
        "sign_comparable_count": len(considered),
        "sign_agreement_count": sum(signs),
        "sign_agreement_rate": sum(signs) / len(signs) if signs else None,
        "proxy_selected": {
            "row": records[proxy_best]["row"],
            "column": records[proxy_best]["column"],
            "tmax_c": actual_values[proxy_best],
            "raw_proxy_tmax_c": raw_values[proxy_best],
            "selection_regret_c": actual_values[proxy_best] - min(actual_values),
        },
        "hotspot_best": {
            "row": records[hotspot_best]["row"],
            "column": records[hotspot_best]["column"],
            "tmax_c": actual_values[hotspot_best],
            "raw_proxy_tmax_c": raw_values[hotspot_best],
        },
        "hotspot_frequency_states": count_states(actual_states),
        "anchored_proxy_frequency_states": count_states(proxy_states),
        "raw_proxy_frequency_states": count_states(raw_proxy_states),
        "hotspot_sustainable_frequency_range_ghz": frequency_range(hotspot_frequencies),
        "anchored_proxy_sustainable_frequency_range_ghz": frequency_range(
            anchored_frequencies
        ),
        "raw_proxy_sustainable_frequency_range_ghz": frequency_range(raw_frequencies),
        "hotspot_frequency_varies": frequency_varies(hotspot_frequencies),
        "anchored_proxy_frequency_varies": frequency_varies(anchored_frequencies),
        "raw_proxy_frequency_varies": frequency_varies(raw_frequencies),
        "frequency_state_comparable_count": len(state_pairs),
        "frequency_state_agreement_count": sum(
            actual == proxy for actual, proxy in state_pairs
        ),
        "frequency_state_agreement_rate": (
            sum(actual == proxy for actual, proxy in state_pairs) / len(state_pairs)
            if state_pairs else None
        ),
        "raw_frequency_state_comparable_count": len(raw_state_pairs),
        "raw_frequency_state_agreement_count": sum(
            actual == proxy for actual, proxy in raw_state_pairs
        ),
        "raw_frequency_state_agreement_rate": (
            sum(actual == proxy for actual, proxy in raw_state_pairs)
            / len(raw_state_pairs) if raw_state_pairs else None
        ),
        "raw_thermal_frequency_term_active": (
            (isinstance(raw_anchor_state, str)
             and raw_anchor_state != "thermal_headroom")
            or any(state != "thermal_headroom" for state in raw_proxy_states
                   if isinstance(state, str))
        ),
        "thermal_frequency_term_active": (
            anchor_frequency["state"] != "thermal_headroom"
            or any(state != "thermal_headroom" for state in actual_states if isinstance(state, str))
            or any(state != "thermal_headroom" for state in proxy_states if isinstance(state, str))
        ),
    }
    if tsafe_c is not None:
        hottest = max(actual_values)
        coolest = min(actual_values)
        # A spatial surrogate cannot create a frequency gain if the complete
        # sampled HotSpot envelope lies below the DTM threshold.  Reporting
        # this margin distinguishes a genuinely inactive physical term from a
        # ranking or parameter failure in Equation (14).
        result["hotspot_frequency_observability"] = {
            "safe_temperature_c": float(tsafe_c),
            "hotspot_tmax_range_c": [coolest, hottest],
            "hottest_headroom_to_safe_c": float(tsafe_c) - hottest,
            "sampled_positions_cross_safe_threshold": (
                coolest <= float(tsafe_c) < hottest
            ),
        }
    return result


def _sample_from_layout(model_path: Path, label: str, layout: dict,
                        case_dir: Path, row: int, column: int,
                        fx: float, fy: float, tier: int) -> dict:
    return {
        "model": str(model_path.resolve()), "model_label": label,
        "tier": tier, "row": row, "column": column, "fx": fx, "fy": fy,
        "case_dir": str(case_dir.resolve()), "layout": layout,
    }


def _run_probe_unlocked(model_path: Path, config_path: Path, output_dir: Path,
                        variants: list[ProxyVariant], grid_points: int = 3,
                        workers: int = 1, hotspot: Path = DEFAULT_HOTSPOT,
                        sign_tolerance_c: float = 0.02, resume: bool = False) -> dict:
    """Run one shared HotSpot grid and evaluate every requested proxy variant."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    if workers < 1:
        raise ValueError("workers must be positive")
    if sign_tolerance_c < 0.0:
        raise ValueError("sign tolerance must be non-negative")
    if not variants:
        raise ValueError("at least one proxy variant is required")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("proxy variant names must be unique")
    config = read_json(config_path)
    model = read_json(model_path)
    prepare_output_directory(output_dir, resume)
    utilization = float(config["physical"]["utilization"])
    baseline = baseline_layout(model, utilization)
    baseline_l2 = next(module for module in baseline["modules"] if module["kind"] == "l2")
    anchor_sample = _sample_from_layout(
        model_path, "fixed-bin", baseline, output_dir / "fixed_bin",
        -1, -1, float(baseline_l2["x_mm"]), float(baseline_l2["y_mm"]),
        int(baseline_l2["tier"]),
    )
    anchor_hotspot = run_one(anchor_sample, config, hotspot, force=False)
    anchor_layout = read_json(Path(anchor_sample["case_dir"]) / "layout.json")
    anchor = {
        "layout": str((Path(anchor_sample["case_dir"]) / "layout.json").resolve()),
        "hotspot_dir": str(Path(anchor_sample["case_dir"]).resolve()),
        "tmax_c": float(anchor_hotspot["tmax_c"]),
        "peak_unit": anchor_hotspot["peak_unit"], "variants": {},
    }
    for variant in variants:
        raw = proxy_for_layout(anchor_layout, config, variant)
        anchor["variants"][variant.name] = {
            "raw_proxy_tmax_c": raw,
            "raw_frequency": _frequency(raw, model, config),
            "anchored_proxy_tmax_c": float(anchor_hotspot["tmax_c"]),
            "anchored_frequency": _frequency(float(anchor_hotspot["tmax_c"]), model, config),
        }

    samples = []
    for candidate in candidate_layouts(model_path, grid_points, utilization, (1,)):
        samples.append(_sample_from_layout(
            model_path, "grid", candidate["layout"],
            output_dir / "positions" /
            f"tier{candidate['tier']}_r{candidate['row']:02d}_c{candidate['column']:02d}",
            int(candidate["row"]), int(candidate["column"]),
            float(candidate["fx"]), float(candidate["fy"]), int(candidate["tier"]),
        ))

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, sample, config, hotspot, False): sample
            for sample in samples
        }
        total = len(futures)
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"HotSpot proxy-gradient probes: {len(completed)}/{total}", flush=True)
    completed.sort(key=lambda item: (item["tier"], item["row"], item["column"]))

    records = []
    for sample in completed:
        layout = read_json(Path(sample["case_dir"]) / "layout.json")
        record = {
            "tier": int(sample["tier"]), "row": int(sample["row"]),
            "column": int(sample["column"]), "fx": float(sample["fx"]),
            "fy": float(sample["fy"]), "x_mm": next(
                module["x_mm"] for module in layout["modules"] if module["kind"] == "l2"
            ), "y_mm": next(
                module["y_mm"] for module in layout["modules"] if module["kind"] == "l2"
            ), "tmax_c": float(sample["tmax_c"]), "peak_unit": sample["peak_unit"],
            "hotspot_dir": sample["case_dir"], "variants": {},
        }
        actual_frequency = _frequency(record["tmax_c"], model, config)
        record["hotspot_frequency"] = actual_frequency
        for variant in variants:
            raw = proxy_for_layout(layout, config, variant)
            delta = raw - anchor["variants"][variant.name]["raw_proxy_tmax_c"]
            anchored = anchor["tmax_c"] + delta
            record["variants"][variant.name] = {
                "raw_proxy_tmax_c": raw,
                "raw_frequency": _frequency(raw, model, config),
                "proxy_delta_from_fixed_bin_c": delta,
                "anchored_proxy_tmax_c": anchored,
                "anchored_frequency": _frequency(anchored, model, config),
            }
        records.append(record)

    report = {
        "schema_version": 1,
        "method": "common-HotSpot-grid equation-(14) gradient diagnostic",
        "purpose": (
            "Qualitatively validate proxy placement gradients, not absolute thermal "
            "accuracy or exact top-K rank agreement."
        ),
        "model": str(model_path.resolve()), "config": str(config_path.resolve()),
        "hotspot": str(hotspot.resolve()), "grid_points_per_axis": grid_points,
        "allowed_l2_tiers": [1], "workers": workers, "resume_requested": resume,
        "variants": [
            {"name": variant.name, "proxy_spatial_model": variant.spatial_model,
             "lc_die_side_ratio": variant.lc_die_side_ratio}
            for variant in variants
        ],
        "anchor": anchor, "records": records,
        "evaluations": {
            variant.name: summarize_variant(
                variant.name, records, anchor, sign_tolerance_c,
                float(config["frequency"]["tsafe_c"]),
            ) for variant in variants
        },
        "interpretation": {
            "anchor_policy": (
                "fixed-bin HotSpot anchor supplies only a common offset; all rank and "
                "gradient differences originate in the analytical proxy"
            ),
            "acceptance_boundary": (
                "This diagnostic does not require exact HotSpot ranking or top-three "
                "overlap. Inspect sign agreement, rank trend, and selection regret "
                "before changing a shared optimizer configuration."
            ),
        },
    }
    write_json(output_dir / "gradient_diagnostic.json", report)
    return report


def run_probe(model_path: Path, config_path: Path, output_dir: Path,
              variants: list[ProxyVariant], grid_points: int = 3,
              workers: int = 1, hotspot: Path = DEFAULT_HOTSPOT,
              sign_tolerance_c: float = 0.02, resume: bool = False) -> dict:
    """Run one diagnostic while holding exclusive ownership of its output tree."""
    with exclusive_output_directory(output_dir):
        return _run_probe_unlocked(
            model_path, config_path, output_dir, variants, grid_points, workers,
            hotspot, sign_tolerance_c, resume,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True,
                        help="repeat NAME=SPATIAL_MODEL,LC_DIE_SIDE_RATIO")
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--sign-tolerance-c", type=float, default=0.02)
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse matching completed HotSpot probes in a non-empty output directory",
    )
    args = parser.parse_args()
    report = run_probe(
        args.modules.resolve(), args.config.resolve(), args.output_dir.resolve(),
        args.variant, args.grid_points, args.workers, args.hotspot.resolve(),
        args.sign_tolerance_c, args.resume,
    )
    for name, values in report["evaluations"].items():
        print(
            f"{name}: Spearman={values['spearman']}, "
            f"sign={values['sign_agreement_rate']}, "
            f"regret={values['proxy_selected']['selection_regret_c']:.6f} C"
        )


if __name__ == "__main__":
    main()
