"""Measures per-event detector latency and peak memory on the machine it runs on.

Excludes OPC-UA transport and PLC scan time: this is the cost of DetectorService.process_phase_event.
Run: python tests/measure_latency.py
"""
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# pylint: disable=wrong-import-position
from detector.cusum_detector import PageCusumDetector
from detector.detector_service import DetectorService, DetectorServiceConfig
from simulator.press_state_machine import PressStateMachine

WARMUP_CYCLES = 50
MEASURED_CYCLES = 300


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def main():
    """Prints mean/p99 per-event latency for the CUSUM stage and for the full pipeline."""
    machine = PressStateMachine(seed=7)
    events = [e for c in range(1, WARMUP_CYCLES + MEASURED_CYCLES + 1) for e in machine.run_cycle(cycle_id=c)]
    with tempfile.TemporaryDirectory() as tmp:
        service = DetectorService(DetectorServiceConfig(telemetry_log_path=str(Path(tmp) / "t.csv")))
        timings = []
        for i, e in enumerate(events):
            start = time.perf_counter()
            service.process_phase_event(cycle_id=e.cycle_id, phase_name=e.phase_name, duration=e.duration_sec)
            if i >= WARMUP_CYCLES * 8:
                timings.append(time.perf_counter() - start)
        service.close()

    cusum = PageCusumDetector()
    ct = []
    for e in events[WARMUP_CYCLES * 8:]:
        start = time.perf_counter()
        cusum.evaluate_observation(cycle_id=e.cycle_id, phase_name=e.phase_name, duration=e.duration_sec,
                                   baseline_mu=e.nominal_duration_sec, baseline_sigma=0.1, sample_count=40)
        ct.append(time.perf_counter() - start)

    def line(name, arr):
        a = np.array(arr)
        print(f"{name}: mean {a.mean() * 1e3:.3f} ms, p99 {np.percentile(a, 99) * 1e3:.3f} ms ({len(a)} events)")

    print(f"{platform.platform()} | {platform.processor() or platform.machine()} | Python {platform.python_version()}")
    line("CUSUM stage      ", ct)
    line("Full pipeline    ", timings)
    print(f"Peak RSS (whole process, incl. imports): {_peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
