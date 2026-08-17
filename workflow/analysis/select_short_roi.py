"""Select the shortest converged all-core R1 measurement window.

Candidates (1M/2M/5M) are compared against the 10M local reference with 1%
aggregate-IPC and 3% power/traffic thresholds.  A candidate is promoted only
when it and every larger candidate pass, otherwise the reference is used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from workflow.common import (
    aggregate_ipc,
    parse_gem5_stats_text,
    read_json,
    sha256_file,
)
from workflow.floorplan.build_module_model import extract_communication_profile
from workflow.r1_protocol import canonical_protocol
from workflow.transient.sampling import measured_roi_duration


CANDIDATE_TARGETS = (1_000_000, 2_000_000, 5_000_000, 10_000_000)


@dataclass(frozen=True)
class ConvergenceLimits:
    ipc: float = 0.01
    total_power: float = 0.03
    module_distribution: float = 0.03
    l2_traffic_distribution: float = 0.03


def _relative(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else math.inf


def collect_candidate(r1_dir: Path, modules_path: Path) -> dict:
    """Validate one candidate R1 point and return its convergence features."""
    r1_dir = Path(r1_dir).resolve()
    modules_path = Path(modules_path).resolve()
    metadata_path = r1_dir / "r1_metadata.json"
    stats_path = r1_dir / "stats.txt"
    if not metadata_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"candidate R1 lacks r1_metadata.json/stats.txt: {r1_dir}"
        )
    metadata = read_json(metadata_path)
    protocol = canonical_protocol(metadata["r1_protocol"])
    if protocol["warmup_insts"] != 2_000_000:
        raise ValueError("candidate R1 must use 2,000,000 warmup instructions")
    if protocol["instruction_window_scope"] != "all-cores":
        raise ValueError("candidate R1 must use all-cores measurement scope")
    target = int(protocol["measure_insts"])
    if target not in CANDIDATE_TARGETS:
        raise ValueError(f"candidate R1 measurement target {target} is unsupported")
    stats = parse_gem5_stats_text(stats_path.read_text(encoding="utf-8"))
    counts = [
        int(stats[f"system.cpu{core}.commitStats0.numInsts"])
        for core in range(4)
    ]
    if any(count < target for count in counts):
        raise ValueError(
            f"candidate R1 cores below measurement target: {counts} < {target}"
        )
    roi = measured_roi_duration(r1_dir)
    if metadata.get("r1_protocol_id") != roi["r1_protocol_id"]:
        raise ValueError("candidate R1 protocol identity is inconsistent")
    ipc = aggregate_ipc(stats)
    if not math.isfinite(ipc) or ipc <= 0:
        raise ValueError("candidate R1 aggregate IPC must be finite positive")
    modules = read_json(modules_path)
    module_names = [
        module["name"] for module in modules.get("modules", [])
    ]
    if len(module_names) != len(set(module_names)):
        raise ValueError("candidate module names must be unique")
    total_power = sum(
        float(module.get("total_power_w", math.nan))
        for module in modules.get("modules", [])
    )
    dynamic_power = sum(
        float(module.get("dynamic_power_w", math.nan))
        for module in modules.get("modules", [])
    )
    leakage_power = sum(
        float(module.get("leakage_power_w", math.nan))
        for module in modules.get("modules", [])
    )
    if any(not math.isfinite(value) or value <= 0
           for value in (total_power, dynamic_power, leakage_power)):
        raise ValueError("candidate module power totals must be finite positive")
    profile = extract_communication_profile(
        stats, num_cores=4, stats_path=stats_path,
        instruction_window_scope=metadata.get(
            "instruction_window_scope", "all-cores"
        ),
        required=True,
    )
    per_core_traffic = [
        float(record["raw_demand_accesses"])
        for record in profile["per_core"].values()
    ]
    return {
        "measure_insts": target,
        "r1_protocol_id": metadata["r1_protocol_id"],
        "r1_dir": str(r1_dir),
        "stats_sha256": sha256_file(stats_path),
        "aggregate_ipc": ipc,
        "total_power_w": total_power,
        "dynamic_power_w": dynamic_power,
        "leakage_power_w": leakage_power,
        "module_names": module_names,
        "module_total_power_w": [
            float(module["total_power_w"]) for module in modules["modules"]
        ],
        "per_core_l2_traffic": per_core_traffic,
    }


def _tv_distance(candidate: list[float], reference: list[float]) -> float:
    total_candidate = sum(candidate)
    total_reference = sum(reference)
    if total_candidate <= 0 or total_reference <= 0:
        return math.inf
    return 0.5 * sum(
        abs(value / total_candidate - reference[index] / total_reference)
        for index, value in enumerate(candidate)
    )


def compare_candidate(candidate: dict, reference: dict,
                      limits: ConvergenceLimits | None = None) -> dict:
    """Compare one candidate to the 10M reference feature by feature."""
    limits = limits or ConvergenceLimits()
    if candidate["module_names"] != reference["module_names"]:
        raise ValueError("candidate and reference module name sets differ")
    metrics = {
        "ipc": _relative(
            candidate["aggregate_ipc"], reference["aggregate_ipc"]
        ),
        "total_power": _relative(
            candidate["total_power_w"], reference["total_power_w"]
        ),
        "module_distribution": _tv_distance(
            candidate["module_total_power_w"],
            reference["module_total_power_w"],
        ),
        "l2_traffic_distribution": _tv_distance(
            candidate["per_core_l2_traffic"],
            reference["per_core_l2_traffic"],
        ),
    }
    thresholds = {
        "ipc": limits.ipc,
        "total_power": limits.total_power,
        "module_distribution": limits.module_distribution,
        "l2_traffic_distribution": limits.l2_traffic_distribution,
    }
    return {
        "candidate_measure_insts": candidate["measure_insts"],
        "reference_measure_insts": reference["measure_insts"],
        "metrics": metrics,
        "thresholds": thresholds,
        "passes": {
            name: math.isfinite(value) and value <= thresholds[name]
            for name, value in metrics.items()
        },
    }


def select_measurement_target(comparisons: dict[int, dict]) -> int:
    """Select the shortest target whose suffix of larger candidates all pass."""
    for target in CANDIDATE_TARGETS[:3]:
        gates = [
            comparisons[larger]
            for larger in CANDIDATE_TARGETS[:3]
            if larger >= target
        ]
        if gates and all(
            all(comparison["passes"].values()) for comparison in gates
        ):
            return target
    return CANDIDATE_TARGETS[-1]


def write_convergence_report(
    candidates: dict[int, tuple[Path, Path]],
    output: Path,
    limits: ConvergenceLimits | None = None,
) -> dict:
    """Collect candidates, compare to 10M, and persist the selection report."""
    limits = limits or ConvergenceLimits()
    reference_dir, reference_modules = candidates[CANDIDATE_TARGETS[-1]]
    reference = collect_candidate(reference_dir, reference_modules)
    comparisons = {}
    collected = {}
    for target in CANDIDATE_TARGETS[:3]:
        candidate_dir, candidate_modules = candidates[target]
        candidate = collect_candidate(candidate_dir, candidate_modules)
        collected[target] = candidate
        comparisons[target] = compare_candidate(candidate, reference, limits)
    selected = select_measurement_target(comparisons)
    report = {
        "schema_version": 1,
        "limits": asdict(limits),
        "selected_measure_insts": selected,
        "comparisons": {
            str(target): comparisons[target] for target in comparisons
        },
        "candidates": {
            str(target): {
                "r1_dir": collected[target]["r1_dir"],
                "r1_protocol_id": collected[target]["r1_protocol_id"],
                "stats_sha256": collected[target]["stats_sha256"],
            }
            for target in collected
        },
        "reference_r1_dir": str(reference_dir),
        "reference_r1_protocol_id": reference["r1_protocol_id"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        import json
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report
