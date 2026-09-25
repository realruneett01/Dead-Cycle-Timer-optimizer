"""Physics-informed discrete-event extrusion press cycle simulator."""
import argparse
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional

from simulator.anomaly_injector import AnomalyEvent, AnomalyInjector, AnomalyType
from simulator.press_config import PhaseTiming, PressConfig

@dataclass
class CycleEvent:
    cycle_id: int
    phase_name: str
    phase_index: int
    start_time: str
    duration_sec: float
    nominal_duration_sec: float
    is_dead_cycle: bool
    pressure_bar: float
    valve_spool_pct: float
    ram_position_mm: float
    has_anomaly: bool
    anomaly_type: str
    injected_delay_sec: float

class PressStateMachine:
    """Simulates realistic kinematic and hydraulic state transitions of an industrial extrusion press."""

    def __init__(
        self,
        config: Optional[PressConfig] = None,
        injector: Optional[AnomalyInjector] = None,
        seed: Optional[int] = None
    ):
        self.config = config or PressConfig()
        self.injector = injector or AnomalyInjector(seed=seed)
        self.rng = random.Random(seed)
        self.current_cycle_id = 0
        self.simulated_clock_sec = 0.0

    def run_cycle(
        self,
        cycle_id: Optional[int] = None,
        wall_clock_start: Optional[datetime] = None
    ) -> List[CycleEvent]:
        """
        Executes a single complete press cycle (all dead-cycle phases + extrusion stroke).
        Returns a list of CycleEvent objects for the cycle.
        """
        if cycle_id is None:
            self.current_cycle_id += 1
            cycle_id = self.current_cycle_id
        else:
            self.current_cycle_id = cycle_id

        if wall_clock_start is None:
            base_time = datetime.now(timezone.utc)
        else:
            base_time = wall_clock_start

        phase_names = self.config.canonical_phase_names
        cycle_anomalies = self.injector.evaluate_cycle_anomalies(cycle_id, phase_names)
        
        cycle_events: List[CycleEvent] = []
        cycle_elapsed_sec = 0.0

        # Simulate kinematic position progression
        # container moves 0 -> 400mm -> 0mm
        # ram moves 0 -> 850mm -> 0mm
        for idx, name in enumerate(phase_names):
            timing: PhaseTiming = self.config.phases[name]
            
            # Base Gaussian jitter based on physical mechanical variance
            jitter = self.rng.gauss(0.0, timing.std_dev_sec)
            actual_duration = timing.nominal_sec + jitter
            
            # Check for anomaly injection
            anomaly_type, injected_delay = cycle_anomalies.get(name, (AnomalyType.NONE, 0.0))
            if anomaly_type != AnomalyType.NONE and injected_delay > 0.0:
                actual_duration += injected_delay
                has_anomaly = True
            else:
                has_anomaly = False

            # Strict physical constraints
            actual_duration = max(timing.min_sec * 0.5, actual_duration)
            actual_duration = round(actual_duration, 3)

            # Simulated physical sensor readings
            # Hydraulic cylinder pressure and proportional valve position
            pressure = round(timing.end_pressure_bar + self.rng.uniform(-3.0, 3.0), 1)
            pressure = max(5.0, min(315.0, pressure))
            
            valve_spool = round(timing.valve_spool_nominal_pct + self.rng.uniform(-2.0, 2.0), 1)
            valve_spool = max(0.0, min(100.0, valve_spool))

            # Approximate ram position per phase
            if name == "rapid_advance":
                ram_pos = 200.0
            elif name == "extrusion":
                ram_pos = 850.0
            else:
                ram_pos = 0.0

            # Calculate start timestamp string
            phase_start_dt = datetime.fromtimestamp(
                base_time.timestamp() + cycle_elapsed_sec,
                tz=timezone.utc
            )
            timestamp_str = phase_start_dt.isoformat()

            event = CycleEvent(
                cycle_id=cycle_id,
                phase_name=name,
                phase_index=idx,
                start_time=timestamp_str,
                duration_sec=actual_duration,
                nominal_duration_sec=timing.nominal_sec,
                is_dead_cycle=timing.is_dead_cycle,
                pressure_bar=pressure,
                valve_spool_pct=valve_spool,
                ram_position_mm=ram_pos,
                has_anomaly=has_anomaly,
                anomaly_type=anomaly_type.value if hasattr(anomaly_type, "value") else str(anomaly_type),
                injected_delay_sec=injected_delay
            )
            cycle_events.append(event)

            # Log to ground truth if an anomaly was injected
            if has_anomaly:
                gt_event = AnomalyEvent(
                    cycle_id=cycle_id,
                    timestamp=timestamp_str,
                    phase_name=name,
                    anomaly_type=anomaly_type,
                    injected_delay_sec=injected_delay,
                    nominal_duration=timing.nominal_sec,
                    actual_duration=actual_duration
                )
                self.injector.log_anomaly(gt_event)

            cycle_elapsed_sec += actual_duration

        self.simulated_clock_sec += cycle_elapsed_sec
        return cycle_events

    def stream_cycles(
        self,
        num_cycles: int,
        real_time_speedup: float = 1.0
    ) -> Generator[CycleEvent, None, None]:
        """
        Streams simulated cycles continuously. If real_time_speedup > 0, pauses
        proportionately to simulate real wall-clock execution.
        """
        for c in range(1, num_cycles + 1):
            events = self.run_cycle(cycle_id=c)
            for event in events:
                if real_time_speedup > 0.0:
                    time.sleep(event.duration_sec / real_time_speedup)
                yield event

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extrusion Press Simulator.")
    parser.add_argument("--cycles", type=int, default=100, help="Number of press cycles to simulate")
    parser.add_argument("--anomaly-prob", type=float, default=0.12, help="Probability of anomaly per cycle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Starting Extrusion Press Simulator: {args.cycles} cycles, p(anomaly)={args.anomaly_prob}...")
    cfg = PressConfig()
    inj = AnomalyInjector(anomaly_probability=args.anomaly_prob, seed=args.seed)
    sm = PressStateMachine(config=cfg, injector=inj, seed=args.seed)

    total_events = 0
    anomalies_count = 0
    start_wall = time.time()

    for c in range(1, args.cycles + 1):
        events = sm.run_cycle(cycle_id=c)
        total_events += len(events)
        for ev in events:
            if ev.has_anomaly:
                anomalies_count += 1

    wall_sec = time.time() - start_wall
    print(f"Completed {args.cycles} cycles ({total_events} phase events) in {wall_sec:.2f}s.")
    print(f"Injected anomalies: {anomalies_count} ({anomalies_count / args.cycles:.2f} anomalies/cycle).")
    print(f"Ground truth written to: {inj.ground_truth_path}")
