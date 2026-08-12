#!/usr/bin/env python3
"""Five-receiver closed-form transient thermal proxy.

This non-formal optimizer model keeps one scalar first-order temperature state
for each of four cores and the shared L2.  It never invokes HotSpot; real
HotSpot remains the final validator for selected layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from workflow.transient.verify_sustainable_frequency import (
    find_sustainable_frequency,
)


RECEIVERS = ("core0", "core1", "core2", "core3", "shared_l2")


@dataclass(frozen=True)
class FiveStateSettings:
    backend: str = "five-state"
    tau_core_s: float = 0.166
    tau_l2_s: float = 0.166
    spatial_model: str = "area-quadrature"
    quadrature_order: int = 2
    parameter_status: str = "provisional"


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or positive and result <= 0.0:
        suffix = " positive" if positive else ""
        raise ValueError(f"{label} must be a finite{suffix} number")
    return result


def parse_five_state_settings(config: dict) -> FiveStateSettings:
    """Parse backend selection and the intentionally small inertia contract."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a dictionary")
    transient = config.get("transient_rom", {})
    if transient is None:
        transient = {}
    if not isinstance(transient, dict):
        raise ValueError("transient_rom must be a dictionary")
    backend = transient.get("backend", "five-state")
    if backend not in ("five-state", "pod-rom"):
        raise ValueError("transient_rom.backend must be five-state or pod-rom")
    controls = transient.get("five_state", {})
    if controls is None:
        controls = {}
    if not isinstance(controls, dict):
        raise ValueError("transient_rom.five_state must be a dictionary")
    unknown = set(controls) - {
        "tau_core_s", "tau_l2_s", "spatial_model", "quadrature_order",
        "parameter_status",
    }
    if unknown:
        raise ValueError(f"unsupported five-state settings: {sorted(unknown)}")
    tau_core = _finite(
        controls.get("tau_core_s", 0.166), "five_state.tau_core_s",
        positive=True,
    )
    tau_l2 = _finite(
        controls.get("tau_l2_s", 0.166), "five_state.tau_l2_s",
        positive=True,
    )
    spatial_model = controls.get("spatial_model", "area-quadrature")
    if spatial_model not in ("center", "area-quadrature"):
        raise ValueError("five_state.spatial_model must be center or area-quadrature")
    order = controls.get("quadrature_order", 2)
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2, 3):
        raise ValueError("five_state.quadrature_order must be 1, 2, or 3")
    status = controls.get("parameter_status", "provisional")
    if status not in ("provisional", "measured"):
        raise ValueError("five_state.parameter_status must be provisional or measured")
    return FiveStateSettings(
        backend=backend, tau_core_s=tau_core, tau_l2_s=tau_l2,
        spatial_model=spatial_model, quadrature_order=order,
        parameter_status=status,
    )


def _rect(module: dict, label: str) -> dict:
    values = {
        field: _finite(module.get(field), f"{label}.{field}")
        for field in ("x_mm", "y_mm", "width_mm", "height_mm")
    }
    if values["width_mm"] <= 0.0 or values["height_mm"] <= 0.0:
        raise ValueError(f"{label} dimensions must be positive")
    tier = module.get("tier")
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in (0, 1):
        raise ValueError(f"{label}.tier must be 0 or 1")
    return {**values, "tier": tier}


def _receiver_geometry(layout: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    if not isinstance(layout, dict) or not isinstance(layout.get("modules"), list):
        raise ValueError("layout must contain a module list")
    width = _finite(layout.get("die_width_mm"), "layout.die_width_mm", positive=True)
    height = _finite(layout.get("die_height_mm"), "layout.die_height_mm", positive=True)
    if not math.isclose(width, height, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("five-state proxy requires a square die")
    by_name: dict[str, dict] = {}
    core_rects: dict[int, list[dict]] = {core: [] for core in range(4)}
    l2s = []
    for index, module in enumerate(layout["modules"]):
        if not isinstance(module, dict):
            raise ValueError("layout modules must be dictionaries")
        name = module.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("layout module names must be unique non-empty strings")
        geometry = _rect(module, f"layout.modules[{index}]")
        by_name[name] = {**module, **geometry}
        core = module.get("core")
        if core is not None:
            if isinstance(core, bool) or not isinstance(core, int) or core not in range(4):
                raise ValueError(f"layout module {name} has invalid core")
            core_rects[core].append(geometry)
        if module.get("kind") == "l2":
            l2s.append((name, geometry))
    if len(l2s) != 1 or any(not core_rects[core] for core in range(4)):
        raise ValueError("layout must contain four core groups and exactly one L2")

    receivers: dict[str, dict] = {}
    for core, rects in core_rects.items():
        tiers = {rect["tier"] for rect in rects}
        if len(tiers) != 1:
            raise ValueError(f"core{core} modules must occupy one tier")
        left = min(rect["x_mm"] for rect in rects)
        bottom = min(rect["y_mm"] for rect in rects)
        right = max(rect["x_mm"] + rect["width_mm"] for rect in rects)
        top = max(rect["y_mm"] + rect["height_mm"] for rect in rects)
        receivers[f"core{core}"] = {
            "x_mm": left, "y_mm": bottom, "width_mm": right - left,
            "height_mm": top - bottom, "tier": next(iter(tiers)),
        }
    l2_name, l2_geometry = l2s[0]
    receivers["shared_l2"] = dict(l2_geometry)
    by_name[l2_name]["receiver"] = "shared_l2"
    return receivers, by_name


def _points(rectangle: dict, settings: FiveStateSettings) -> list[tuple[float, float]]:
    if settings.spatial_model == "center":
        fractions = (0.5,)
    else:
        fractions = {
            1: (0.5,), 2: (0.25, 0.75), 3: (1 / 6, 0.5, 5 / 6),
        }[settings.quadrature_order]
    return [
        (
            rectangle["x_mm"] + fx * rectangle["width_mm"],
            rectangle["y_mm"] + fy * rectangle["height_mm"],
        )
        for fy in fractions for fx in fractions
    ]


def _coupling(receivers: dict[str, dict], die_width: float,
              cross_tier_weight: float,
              settings: FiveStateSettings) -> dict[str, dict[str, float]]:
    length = die_width / 2.0
    result = {}
    for receiver_name, receiver in receivers.items():
        receiver_points = _points(receiver, settings)
        result[receiver_name] = {}
        for source_name, source in receivers.items():
            source_points = _points(source, settings)
            tier_weight = (
                1.0 if receiver["tier"] == source["tier"] else cross_tier_weight
            )
            values = []
            for xi, yi in receiver_points:
                values.append(sum(
                    tier_weight / math.sqrt(
                        1.0 + (math.hypot(xi - xj, yi - yj) / length) ** 2
                    )
                    for xj, yj in source_points
                ) / len(source_points))
            result[receiver_name][source_name] = max(values)
    return result


def _window_power(window: dict, by_name: dict[str, dict], scale: float) -> dict:
    modules = window.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("each power window must contain a non-empty module list")
    expected = set(by_name)
    observed: dict[str, dict] = {}
    source_power = {
        receiver: {"dynamic_power_w": 0.0, "leakage_power_w": 0.0}
        for receiver in RECEIVERS
    }
    totals = {"dynamic_power_w": 0.0, "leakage_power_w": 0.0}
    bottom = {"dynamic_power_w": 0.0, "leakage_power_w": 0.0}
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            raise ValueError("power-window modules must be dictionaries")
        name = module.get("name")
        if not isinstance(name, str) or name not in expected or name in observed:
            raise ValueError("power-window module names must exactly match the layout")
        dynamic = _finite(module.get("dynamic_power_w"), f"window module {name} dynamic")
        leakage = _finite(module.get("leakage_power_w"), f"window module {name} leakage")
        total = _finite(module.get("total_power_w"), f"window module {name} total")
        if min(dynamic, leakage, total) < 0.0 or not math.isclose(
            total, dynamic + leakage, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"window module {name} has an invalid power triplet")
        scaled_dynamic = scale * dynamic
        observed[name] = module
        totals["dynamic_power_w"] += scaled_dynamic
        totals["leakage_power_w"] += leakage
        if by_name[name]["tier"] == 0:
            bottom["dynamic_power_w"] += scaled_dynamic
            bottom["leakage_power_w"] += leakage
        core = by_name[name].get("core")
        receiver = f"core{core}" if core in range(4) else (
            "shared_l2" if by_name[name].get("kind") == "l2" else None
        )
        if receiver is not None:
            source_power[receiver]["dynamic_power_w"] += scaled_dynamic
            source_power[receiver]["leakage_power_w"] += leakage
    if set(observed) != expected:
        raise ValueError("power-window module names must exactly match the layout")
    for values in (*source_power.values(), totals, bottom):
        values["total_power_w"] = (
            values["dynamic_power_w"] + values["leakage_power_w"]
        )
    return {"sources": source_power, "totals": totals, "bottom": bottom}


def _controls(config: dict) -> dict:
    frequency = config.get("frequency")
    optimizer = config.get("layout_optimizer")
    if not isinstance(frequency, dict) or not isinstance(optimizer, dict):
        raise ValueError("config requires frequency and layout_optimizer dictionaries")
    return {
        "f0_ghz": _finite(frequency.get("f0_ghz"), "frequency.f0_ghz", positive=True),
        "fmin_ghz": _finite(frequency.get("fmin_ghz"), "frequency.fmin_ghz", positive=True),
        "tsafe_c": _finite(frequency.get("tsafe_c"), "frequency.tsafe_c"),
        "ambient_c": _finite(frequency.get("ambient_c"), "frequency.ambient_c"),
        "r_global": _finite(
            optimizer.get("r_convec_k_per_w"),
            "layout_optimizer.r_convec_k_per_w",
        ),
        "alpha": _finite(optimizer.get("alpha"), "layout_optimizer.alpha"),
        "beta": _finite(optimizer.get("beta"), "layout_optimizer.beta"),
        "cross": _finite(
            optimizer.get("cross_tier_weight"),
            "layout_optimizer.cross_tier_weight",
        ),
    }


def evaluate_five_state(layout: dict, power_windows: dict,
                        frequency_ghz: float, config: dict) -> dict:
    """Evaluate one layout/frequency at exact scalar periodic steady state."""
    settings = parse_five_state_settings(config)
    controls = _controls(config)
    frequency = _finite(frequency_ghz, "frequency_ghz", positive=True)
    if not controls["fmin_ghz"] <= frequency <= controls["f0_ghz"]:
        raise ValueError("frequency_ghz lies outside configured bounds")
    scale = frequency / controls["f0_ghz"]
    receivers, by_name = _receiver_geometry(layout)
    coupling = _coupling(
        receivers, float(layout["die_width_mm"]), controls["cross"], settings
    )
    windows = power_windows.get("windows") if isinstance(power_windows, dict) else None
    if not isinstance(windows, list) or not windows:
        raise ValueError("power_windows must contain at least one window")
    prepared = []
    tau = {
        receiver: settings.tau_l2_s if receiver == "shared_l2" else settings.tau_core_s
        for receiver in RECEIVERS
    }
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError("power windows must be dictionaries")
        nominal_duration = _finite(
            window.get("duration_s"), f"windows[{index}].duration_s", positive=True
        )
        duration = nominal_duration / scale
        power = _window_power(window, by_name, scale)
        equilibrium = {}
        inertia = {}
        for receiver in RECEIVERS:
            local = sum(
                coupling[receiver][source] * power["sources"][source]["total_power_w"]
                for source in RECEIVERS
            )
            equilibrium[receiver] = (
                controls["ambient_c"]
                + controls["r_global"] * power["totals"]["total_power_w"]
                + controls["alpha"] * local
                + controls["beta"] * power["bottom"]["total_power_w"]
            )
            inertia[receiver] = math.exp(-duration / tau[receiver])
        prepared.append({
            "index": window.get("index", index),
            "nominal_duration_s": nominal_duration,
            "duration_s": duration,
            "frequency_scale": scale,
            **power["totals"],
            "bottom_power_w": power["bottom"]["total_power_w"],
            "source_power_w": {
                receiver: values["total_power_w"]
                for receiver, values in power["sources"].items()
            },
            "equilibrium_c": equilibrium,
            "equilibrium_peak_receiver": max(equilibrium, key=equilibrium.get),
            "inertia": inertia,
        })

    cycle_a = {receiver: 1.0 for receiver in RECEIVERS}
    cycle_b = {receiver: 0.0 for receiver in RECEIVERS}
    for window in prepared:
        for receiver in RECEIVERS:
            a = window["inertia"][receiver]
            cycle_a[receiver] = a * cycle_a[receiver]
            cycle_b[receiver] = (
                a * cycle_b[receiver]
                + (1.0 - a) * window["equilibrium_c"][receiver]
            )
    initial = {}
    for receiver in RECEIVERS:
        denominator = 1.0 - cycle_a[receiver]
        if denominator <= 0.0:
            raise ArithmeticError("five-state periodic solution is singular")
        initial[receiver] = cycle_b[receiver] / denominator

    state = dict(initial)
    peak_temperature = max(state.values())
    peak_receiver = max(state, key=state.get)
    peak_phase = "period_initial_state"
    peak_window = -1
    for index, window in enumerate(prepared):
        for receiver in RECEIVERS:
            a = window["inertia"][receiver]
            state[receiver] = (
                a * state[receiver]
                + (1.0 - a) * window["equilibrium_c"][receiver]
            )
        window["temperature_c"] = dict(state)
        window["peak_receiver"] = max(state, key=state.get)
        window["peak_c"] = state[window["peak_receiver"]]
        if window["peak_c"] > peak_temperature:
            peak_temperature = window["peak_c"]
            peak_receiver = window["peak_receiver"]
            peak_phase = "window_end"
            peak_window = index
    closure = max(abs(state[name] - initial[name]) for name in RECEIVERS)
    return {
        "schema_version": 1,
        "thermal_backend": "five-state",
        "non_formal": True,
        "paper_equivalent": False,
        "frequency_ghz": frequency,
        "frequency_scale": scale,
        "parameter_status": settings.parameter_status,
        "parameters": {
            "tau_core_s": settings.tau_core_s,
            "tau_l2_s": settings.tau_l2_s,
            "spatial_model": settings.spatial_model,
            "quadrature_order": settings.quadrature_order,
            "r_global_k_per_w": controls["r_global"],
            "alpha": controls["alpha"],
            "beta": controls["beta"],
            "cross_tier_weight": controls["cross"],
        },
        "equations": {
            "recurrence": "T_next=a*T+(1-a)*T_eq",
            "inertia": "a=exp(-duration_nominal/(frequency_scale*tau))",
            "power": "P=leakage+frequency_scale*dynamic",
            "periodic_initial": "T0=B_cycle/(1-A_cycle)",
        },
        "periodic_initial_c": initial,
        "periodic_final_c": state,
        "pss_closure_max_abs_c": closure,
        "peak_c": peak_temperature,
        "peak_receiver": peak_receiver,
        "peak_phase": peak_phase,
        "peak_window_index": peak_window,
        "windows": prepared,
        "hotspot_calls": 0,
    }


def find_five_state_sustainable_frequency(
    layout: dict, power_windows: dict, frequencies_ghz: list[float], config: dict,
) -> dict:
    """Find the transient-safe frequency using only five-state evaluations."""
    settings = parse_five_state_settings(config)
    controls = _controls(config)
    transient = config.get("transient_rom", {})
    tolerance = _finite(
        transient.get("frequency_tolerance_ghz", 0.01),
        "transient_rom.frequency_tolerance_ghz", positive=True,
    )

    def evaluate(frequency: float) -> dict:
        value = evaluate_five_state(layout, power_windows, frequency, config)
        return {
            **value,
            "converged": value["pss_closure_max_abs_c"] <= 1e-9,
            "last_period_peak_c": value["peak_c"],
            "last_period_peak_receiver": value["peak_receiver"],
        }

    search = find_sustainable_frequency(
        evaluate, frequencies_ghz, controls["tsafe_c"], tolerance,
    )
    return {
        **search,
        "thermal_backend": "five-state",
        "parameter_status": settings.parameter_status,
        "hotspot_calls_inside_evaluator": 0,
    }
