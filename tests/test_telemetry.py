import unittest

from agent_lab.telemetry import EnergyCounter, Telemetry, hardware_profile


class TelemetryTests(unittest.TestCase):
    def test_phase_times_memory_and_unknown_energy_are_separate(self):
        now = [0.0]
        meter = Telemetry(
            clock=lambda: now[0],
            cpu_clock=lambda: now[0] / 2,
            memory_probe=lambda: (100, None),
            periodic=False,
        )
        meter.start()
        meter.emit("model_request", {})
        now[0] = 3
        meter.emit(
            "model_response",
            {
                "measurements": {
                    "load_duration": 1,
                    "prompt_eval_duration": 0.5,
                    "eval_duration": 1.5,
                }
            },
        )
        meter.emit("tool_request", {})
        now[0] = 5
        meter.emit("tool_result", {})
        now[0] = 6
        values, sources = meter.stop()
        self.assertEqual(values["wall_seconds"], 6)
        self.assertEqual(values["inference_seconds"], 2)
        self.assertEqual(values["setup_seconds"], 1)
        self.assertEqual(values["tool_seconds"], 2)
        self.assertEqual(values["peak_ram_bytes"], 100)
        self.assertIsNone(values["energy_joules"])
        self.assertIsNone(values["peak_vram_bytes"])
        self.assertIn("wall_seconds", sources)
        self.assertNotIn("energy_joules", sources)

    def test_energy_wrap_and_unsupported_reading(self):
        readings = iter([90, 10])
        counter = EnergyCounter(lambda: next(readings), max_microjoules=100)
        counter.start()
        self.assertEqual(counter.stop(), 20 / 1e6)
        self.assertIn("architecture", hardware_profile())

    def test_failed_inference_is_not_recorded_as_zero(self):
        meter = Telemetry(periodic=False, memory_probe=lambda: (None, None))
        meter.start()
        meter.emit("model_request", {})
        values, _ = meter.stop()
        self.assertIsNone(values["inference_seconds"])
