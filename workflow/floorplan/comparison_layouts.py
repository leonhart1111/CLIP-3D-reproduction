#!/usr/bin/env python3
"""Deterministic reproductions of the paper's under-specified layout baselines.

The paper names ``cool3d-standard`` and ``SA + lambda`` but does not publish
their implementation or numerical search parameters.  This module therefore
keeps the search finite, seeded, and fully recorded instead of pretending the
authors' exact layouts are available.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import baseline_layout, check_geometry, overlap
from workflow.floorplan.layout_metrics import mean_wire_cycles
from workflow.floorplan.optimize_layout import proxy_temperature


METHODS = ("cool3d-standard", "sa-lambda")


def collision_area(candidate: dict, fixed: list[dict]) -> float:
    return sum(
        overlap(candidate, module["x_mm"], module["y_mm"],
                module["x_mm"] + module["width_mm"],
                module["y_mm"] + module["height_mm"])
        for module in fixed if module["tier"] == candidate["tier"]
    )


def finite_candidates(model: dict, utilization: float, grid: int,
                      ambient: float, r_convec: float, alpha: float,
                      beta: float, cross_tier_weight: float,
                      f0_ghz: float) -> tuple[dict, list[dict]]:
    if grid < 2:
        raise ValueError("candidate grid must be at least 2")
    base = baseline_layout(model, utilization)
    side = base["die_width_mm"]
    original = next(module for module in base["modules"] if module["kind"] == "l2")
    fixed = [dict(module) for module in base["modules"] if module["kind"] != "l2"]
    xmax = side - original["width_mm"]
    ymax = side - original["height_mm"]
    candidates = []
    for tier in (0, 1):
        for yi in range(grid):
            for xi in range(grid):
                x = xmax * xi / (grid - 1)
                y = ymax * yi / (grid - 1)
                l2 = dict(original, tier=tier, x_mm=x, y_mm=y)
                if collision_area(l2, fixed) > 1e-8:
                    continue
                modules = fixed + [l2]
                candidates.append({
                    "index": len(candidates),
                    "tier": tier,
                    "grid_x": xi,
                    "grid_y": yi,
                    "x_mm": x,
                    "y_mm": y,
                    "proxy_tmax_c": proxy_temperature(
                        modules, side, ambient, r_convec, alpha, beta,
                        cross_tier_weight
                    ),
                    "mean_wire_cycles": mean_wire_cycles(modules, f0_ghz)[0],
                    "layout": dict(base, modules=modules),
                })
    if not candidates:
        raise RuntimeError("comparison layout search found no legal L2 placement")
    for field in ("proxy_tmax_c", "mean_wire_cycles"):
        values = [candidate[field] for candidate in candidates]
        low, high = min(values), max(values)
        for candidate in candidates:
            candidate["normalized_" + field] = (
                (candidate[field] - low) / (high - low) if high > low else 0.0
            )
    return base, candidates


def anneal(candidates: list[dict], weight: float, iterations: int,
           randomizer: random.Random) -> tuple[int, int]:
    current = randomizer.randrange(len(candidates))

    def cost(index: int) -> float:
        candidate = candidates[index]
        return (weight * candidate["normalized_proxy_tmax_c"] +
                (1.0 - weight) * candidate["normalized_mean_wire_cycles"])

    current_cost = cost(current)
    best, best_cost = current, current_cost
    accepted = 0
    for iteration in range(max(iterations, 1)):
        temperature = max(1e-4, 1.0 - iteration / max(iterations, 1))
        proposal = randomizer.randrange(len(candidates))
        proposal_cost = cost(proposal)
        delta = proposal_cost - current_cost
        if delta <= 0 or randomizer.random() < math.exp(-delta / temperature):
            current, current_cost = proposal, proposal_cost
            accepted += 1
        if current_cost < best_cost:
            best, best_cost = current, current_cost
    return best, accepted


def generate(model_path: Path, output_dir: Path, method: str,
             utilization: float = 0.70, candidate_grid: int = 11,
             top_k: int = 3, ambient: float = 25.0,
             r_convec: float = 5.0, alpha: float = 0.3,
             beta: float = 0.1, cross_tier_weight: float = 0.65,
             f0_ghz: float = 2.0, sa_iterations: int = 600,
             sa_seed: int = 20260725) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model = read_json(model_path)
    base, candidates = finite_candidates(
        model, utilization, candidate_grid, ambient, r_convec, alpha, beta,
        cross_tier_weight, f0_ghz
    )
    search = []
    if method == "cool3d-standard":
        ordered_indices = [candidate["index"] for candidate in sorted(
            candidates,
            key=lambda item: (item["proxy_tmax_c"], item["mean_wire_cycles"],
                              item["tier"], item["grid_y"], item["grid_x"]),
        )]
    else:
        randomizer = random.Random(sa_seed)
        winner_indices = []
        for step in range(11):
            weight = step / 10.0
            winner, accepted = anneal(candidates, weight, sa_iterations, randomizer)
            winner_indices.append(winner)
            search.append({"lambda_thermal": weight, "winner": winner,
                           "accepted_moves": accepted})
        # Preserve the thermal/wire knee and both endpoints before the rest of
        # the 11-point lambda grid.  Duplicate winners are evaluated once.
        preferred_steps = (5, 10, 0, 4, 6, 3, 7, 2, 8, 1, 9)
        ordered_indices = [winner_indices[step] for step in preferred_steps]

    selected_indices = []
    for index in ordered_indices:
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) == top_k:
            break
    if len(selected_indices) < top_k:
        for candidate in sorted(candidates, key=lambda item: (
                item["normalized_proxy_tmax_c"] + item["normalized_mean_wire_cycles"],
                item["index"])):
            if candidate["index"] not in selected_indices:
                selected_indices.append(candidate["index"])
            if len(selected_indices) == top_k:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    emitted = []
    by_index = {candidate["index"]: candidate for candidate in candidates}
    for rank, index in enumerate(selected_indices):
        candidate = by_index[index]
        layout = dict(candidate["layout"])
        layout["policy"] = method
        layout["comparison_candidate"] = {
            key: value for key, value in candidate.items() if key != "layout"
        }
        check_geometry(layout["modules"], layout["die_width_mm"])
        path = output_dir / f"candidate_{rank:02d}.json"
        write_json(path, layout)
        emitted.append({"rank": rank, "candidate_index": index,
                        "layout": str(path.resolve()),
                        "proxy_tmax_c": candidate["proxy_tmax_c"],
                        "mean_wire_cycles": candidate["mean_wire_cycles"]})

    report = {
        "schema_version": 1,
        "method": method,
        "candidate_count": len(candidates),
        "hotspot_candidate_count": len(emitted),
        "emitted": emitted,
        "search": search,
        "parameters": {
            "candidate_grid": candidate_grid,
            "top_k": top_k,
            "ambient_c": ambient,
            "r_convec_k_per_w": r_convec,
            "alpha": alpha,
            "beta": beta,
            "cross_tier_weight": cross_tier_weight,
            "f0_ghz": f0_ghz,
            "sa_iterations": sa_iterations,
            "sa_seed": sa_seed,
        },
        "reproduction_boundary": (
            "The paper does not publish these baseline algorithms or parameters; "
            "this deterministic finite/seeded implementation is an explicit reproduction."
        ),
        "base_policy": base["policy"],
    }
    write_json(output_dir / "search_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--candidate-grid", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sa-iterations", type=int, default=600)
    parser.add_argument("--sa-seed", type=int, default=20260725)
    args = parser.parse_args()
    result = generate(args.modules.resolve(), args.output_dir.resolve(), args.method,
                      candidate_grid=args.candidate_grid, top_k=args.top_k,
                      sa_iterations=args.sa_iterations, sa_seed=args.sa_seed)
    print(f"{result['method']}: emitted {result['hotspot_candidate_count']} candidates")


if __name__ == "__main__":
    main()
