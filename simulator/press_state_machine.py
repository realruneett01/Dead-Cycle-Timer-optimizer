"""Physics-informed discrete-event extrusion press cycle simulator."""
import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, List, Optional, Tuple

from simulator.anomaly_injector import AnomalyEvent, AnomalyInjector, AnomalyType
from simulator.press_config import PhaseTiming, PressConfig

@dataclass
class PhaseEventContext:
    """Context parameters for instantiating a cycle phase event."""
    idx: int
    name: str
    active_id: int
    base_time: datetime
    cycle_elapsed_sec: float
    anomaly_info: Tuple[AnomalyType, float]


@dataclass
class CycleEvent:
    """Telemetry and anomaly metadata for a single simulated press cycle phase."""
    # pylint: disable=too-many-instance-attributes
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
    """Simulates realistic kinematic and hydraulic state transitions
    of an industrial extrusion press.
    """

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

    def _compute_phase_duration(
        self,
        timing: PhaseTiming,
        anomaly_info: Tuple[AnomalyType, float]
    ) -> Tuple[float, bool, AnomalyType, float]:
        """Calculates actual phase duration incorporating mechanical jitter and anomalies."""
        jitter = self.rng.gauss(0.0, timing.std_dev_sec)
        actual_duration = timing.nominal_sec + jitter
        anomaly_type, injected_delay = anomaly_info
        has_anomaly = bool(anomaly_type != AnomalyType.NONE and injected_delay > 0.0)
        if has_anomaly:
            actual_duration += injected_delay

        actual_duration = max(timing.min_sec * 0.5, actual_duration)
        return round(actual_duration, 3), has_anomaly, anomaly_type, injected_delay

    def _sample_telemetry(self, timing: PhaseTiming, phase_name: str) -> Tuple[float, float, float]:
        """Simulates sensor telemetry: pressure (bar), valve spool (%), and ram position (mm)."""
        pressure = round(timing.end_pressure_bar + self.rng.uniform(-3.0, 3.0), 1)
        pressure = max(5.0, min(315.0, pressure))
        valve_spool = round(timing.valve_spool_nominal_pct + self.rng.uniform(-2.0, 2.0), 1)
        valve_spool = max(0.0, min(100.0, valve_spool))

        if phase_name == "rapid_advance":
            ram_pos = 200.0
        elif phase_name == "extrusion":
            ram_pos = 850.0
        else:
            ram_pos = 0.0

        return pressure, valve_spool, ram_pos

    def _log_ground_truth(self, event: CycleEvent, timing: PhaseTiming, anomaly_type: AnomalyType):
        """Appends ground truth record for an injected anomaly."""
        gt_event = AnomalyEvent(
            cycle_id=event.cycle_id,
            timestamp=event.start_time,
            phase_name=event.phase_name,
            anomaly_type=anomaly_type,
            injected_delay_sec=event.injected_delay_sec,
            nominal_duration=timing.nominal_sec,
            actual_duration=event.duration_sec
        )
        self.injector.log_anomaly(gt_event)

    def _create_phase_event(self, ctx: PhaseEventContext) -> CycleEvent:
        # pylint: disable=too-many-locals
        """Constructs a CycleEvent and logs ground truth if anomalous."""
        timing: PhaseTiming = self.config.phases[ctx.name]
        duration, has_anomaly, anom_type, delay = self._compute_phase_duration(timing, ctx.anomaly_info)
        pressure, valve_spool, ram_pos = self._sample_telemetry(timing, ctx.name)

        offset = ctx.cycle_elapsed_sec
        phase_dt = datetime.fromtimestamp(ctx.base_time.timestamp() + offset, tz=timezone.utc)
        event = CycleEvent(
            cycle_id=ctx.active_id,
            phase_name=ctx.name,
            phase_index=ctx.idx,
            start_time=phase_dt.isoformat(),
            duration_sec=duration,
            nominal_duration_sec=timing.nominal_sec,
            is_dead_cycle=timing.is_dead_cycle,
            pressure_bar=pressure,
            valve_spool_pct=valve_spool,
            ram_position_mm=ram_pos,
            has_anomaly=has_anomaly,
            anomaly_type=anom_type.value if hasattr(anom_type, "value") else str(anom_type),
            injected_delay_sec=delay
        )
        if has_anomaly:
            self._log_ground_truth(event, timing, anom_type)
        return event

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
        else:
            self.current_cycle_id = cycle_id
        active_id = self.current_cycle_id
        base_time = wall_clock_start if wall_clock_start is not None else datetime.now(timezone.utc)

        phase_names = self.config.canonical_phase_names
        cycle_anomalies = self.injector.evaluate_cycle_anomalies(active_id, phase_names)

        cycle_events: List[CycleEvent] = []
        cycle_elapsed_sec = 0.0

        for idx, name in enumerate(phase_names):
            anomaly_info = cycle_anomalies.get(name, (AnomalyType.NONE, 0.0))
            ctx = PhaseEventContext(
                idx=idx,
                name=name,
                active_id=active_id,
                base_time=base_time,
                cycle_elapsed_sec=cycle_elapsed_sec,
                anomaly_info=anomaly_info
            )
            event = self._create_phase_event(ctx)
            cycle_events.append(event)
            cycle_elapsed_sec += event.duration_sec

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

def main():
    """CLI driver for the extrusion press simulator."""
    parser = argparse.ArgumentParser(description="Run Extrusion Press Simulator.")
    parser.add_argument(
        "--cycles", type=int, default=100, help="Number of press cycles to simulate"
    )
    parser.add_argument(
        "--anomaly-prob", type=float, default=0.12, help="Probability of anomaly per cycle"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(
        f"Starting Extrusion Press Simulator: {args.cycles} cycles, "
        f"p(anomaly)={args.anomaly_prob}..."
    )
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
    rate = anomalies_count / args.cycles if args.cycles > 0 else 0.0
    print(f"Completed {args.cycles} cycles ({total_events} phase events) in {wall_sec:.2f}s.")
    print(f"Injected anomalies: {anomalies_count} ({rate:.2f} anomalies/cycle).")
    print(f"Ground truth written to: {inj.ground_truth_path}")

if __name__ == "__main__":
    main()
