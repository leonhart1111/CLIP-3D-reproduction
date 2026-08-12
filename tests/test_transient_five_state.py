from __future__ import annotations

import math
import unittest

from workflow.transient.five_state import (
    evaluate_five_state,
    find_five_state_sustainable_frequency,
    parse_five_state_settings,
)


class FiveStateFixture:
    @staticmethod
    def config() -> dict:
        return {
            "frequency": {
                "f0_ghz": 2.0,
                "fmin_ghz": 0.5,
                "tsafe_c": 80.0,
                "ambient_c": 25.0,
            },
            "layout_optimizer": {
                "r_convec_k_per_w": 1.0,
                "alpha": 2.0,
                "beta": 0.25,
                "cross_tier_weight": 0.5,
            },
            "transient_rom": {
                "five_state": {
                    "tau_core_s": 0.1,
                    "tau_l2_s": 0.2,
                    "spatial_model": "area-quadrature",
                    "quadrature_order": 1,
                    "parameter_status": "measured",
                },
                "frequency_tolerance_ghz": 0.01,
            },
        }

    @staticmethod
    def layout(l2_x: float = 8.0) -> dict:
        modules = []
        for core, (x, y) in enumerate(((0, 0), (6, 0), (0, 6), (6, 6))):
            modules.append({
                "name": f"core{core}_exec", "kind": "core_exec", "core": core,
                "tier": 0, "x_mm": float(x), "y_mm": float(y),
                "width_mm": 2.0, "height_mm": 2.0,
            })
        modules.extend((
            {
                "name": "shared_l2", "kind": "l2", "tier": 1,
                "x_mm": l2_x, "y_mm": 4.0, "width_mm": 2.0,
                "height_mm": 2.0,
            },
            {
                "name": "noc", "kind": "interconnect", "tier": 1,
                "x_mm": 0.0, "y_mm": 9.0, "width_mm": 1.0,
                "height_mm": 1.0,
            },
        ))
        return {
            "die_width_mm": 10.0,
            "die_height_mm": 10.0,
            "modules": modules,
        }

    @staticmethod
    def windows(dynamic_core: int = 0) -> dict:
        windows = []
        for index in range(2):
            powers = []
            for core in range(4):
                dynamic = 4.0 if core == (dynamic_core + index) % 4 else 0.0
                powers.append({
                    "name": f"core{core}_exec", "kind": "core_exec",
                    "core": core, "dynamic_power_w": dynamic,
                    "leakage_power_w": 1.0,
                    "total_power_w": dynamic + 1.0,
                })
            powers.extend((
                {
                    "name": "shared_l2", "kind": "l2",
                    "dynamic_power_w": 1.0, "leakage_power_w": 0.5,
                    "total_power_w": 1.5,
                },
                {
                    "name": "noc", "kind": "interconnect",
                    "dynamic_power_w": 0.0, "leakage_power_w": 0.5,
                    "total_power_w": 0.5,
                },
            ))
            windows.append({"index": index, "duration_s": 0.002, "modules": powers})
        return {
            "nominal_sample_interval_ms": 2.0,
            "windows": windows,
        }


class FiveStateSettingsTests(unittest.TestCase):
    def test_omitted_backend_defaults_to_provisional_five_state(self) -> None:
        # Catches silently retaining the failed matrix ROM as the default.
        settings = parse_five_state_settings({})
        self.assertEqual(settings.backend, "five-state")
        self.assertEqual(settings.parameter_status, "provisional")
        self.assertEqual(settings.tau_core_s, 0.166)
        self.assertEqual(settings.tau_l2_s, 0.166)

    def test_explicit_pod_backend_is_preserved(self) -> None:
        settings = parse_five_state_settings({
            "transient_rom": {"backend": "pod-rom"}
        })
        self.assertEqual(settings.backend, "pod-rom")

    def test_invalid_controls_are_rejected(self) -> None:
        for field, value in (
            ("tau_core_s", 0.0),
            ("tau_l2_s", -1.0),
            ("tau_l2_s", float("inf")),
            ("quadrature_order", 4),
            ("parameter_status", "fitted-for-bips"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse_five_state_settings({
                    "transient_rom": {"five_state": {field: value}}
                })
        with self.assertRaises(ValueError):
            parse_five_state_settings({"transient_rom": {"backend": "grid-rom"}})


class FiveStateNumericsTests(unittest.TestCase, FiveStateFixture):
    def test_frequency_scales_dynamic_power_and_stretches_time(self) -> None:
        # Catches reducing power without extending execution time at low frequency.
        result = evaluate_five_state(
            self.layout(), self.windows(), 1.0, self.config()
        )
        first = result["windows"][0]
        self.assertEqual(first["frequency_scale"], 0.5)
        self.assertEqual(first["duration_s"], 0.004)
        self.assertEqual(first["dynamic_power_w"], 2.5)
        self.assertEqual(first["leakage_power_w"], 5.0)
        self.assertEqual(first["total_power_w"], 7.5)
        self.assertAlmostEqual(first["inertia"]["core0"], math.exp(-0.004 / 0.1))
        self.assertAlmostEqual(first["inertia"]["shared_l2"], math.exp(-0.004 / 0.2))

    def test_periodic_initial_state_closes_after_one_roi(self) -> None:
        # Catches cold-start or arbitrary-repeat initialisation in a sustained run.
        result = evaluate_five_state(
            self.layout(), self.windows(), 2.0, self.config()
        )
        self.assertLess(result["pss_closure_max_abs_c"], 1e-10)
        self.assertEqual(
            set(result["periodic_initial_c"]),
            {"core0", "core1", "core2", "core3", "shared_l2"},
        )

    def test_constant_equilibrium_is_a_fixed_point(self) -> None:
        # Catches an incorrect affine composition or PSS denominator.
        windows = self.windows()
        windows["windows"][1]["modules"] = [
            dict(module) for module in windows["windows"][0]["modules"]
        ]
        result = evaluate_five_state(self.layout(), windows, 2.0, self.config())
        first = result["windows"][0]
        for receiver, equilibrium in first["equilibrium_c"].items():
            self.assertAlmostEqual(
                result["periodic_initial_c"][receiver], equilibrium, places=10
            )
            self.assertAlmostEqual(first["temperature_c"][receiver], equilibrium, places=10)

    def test_hotspot_receiver_can_move_between_windows(self) -> None:
        # Catches collapsing the model to one scalar Tmax history.
        config = self.config()
        config["layout_optimizer"]["r_convec_k_per_w"] = 0.0
        config["layout_optimizer"]["beta"] = 0.0
        result = evaluate_five_state(self.layout(), self.windows(), 2.0, config)
        identities = [window["equilibrium_peak_receiver"] for window in result["windows"]]
        self.assertEqual(identities, ["core0", "core1"])

    def test_l2_location_changes_receiver_equilibrium(self) -> None:
        # Catches a layout-insensitive thermal proxy.
        near = evaluate_five_state(self.layout(0.0), self.windows(), 2.0, self.config())
        far = evaluate_five_state(self.layout(8.0), self.windows(), 2.0, self.config())
        self.assertNotEqual(
            near["windows"][0]["equilibrium_c"]["core0"],
            far["windows"][0]["equilibrium_c"]["core0"],
        )

    def test_frequency_search_records_closed_form_evidence(self) -> None:
        # Catches replacing proxy calls with hidden HotSpot invocations.
        result = find_five_state_sustainable_frequency(
            self.layout(), self.windows(), [0.5, 2.0], self.config()
        )
        self.assertGreaterEqual(len(result["evaluations"]), 2)
        self.assertTrue(all(value["hotspot_calls"] == 0 for value in result["evaluations"]))
        self.assertTrue(all(value["converged"] for value in result["evaluations"]))
        self.assertEqual(result["thermal_backend"], "five-state")
        self.assertEqual(result["parameter_status"], "measured")

    def test_malformed_power_triplet_is_rejected(self) -> None:
        # Catches silently inventing dynamic/leakage components.
        windows = self.windows()
        windows["windows"][0]["modules"][0].pop("leakage_power_w")
        with self.assertRaises(ValueError):
            evaluate_five_state(self.layout(), windows, 2.0, self.config())


if __name__ == "__main__":
    unittest.main()
