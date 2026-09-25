"""Industrial configuration and nominal operating parameters for Heavy Extrusion Press."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class PhaseTiming:
    name: str
    nominal_sec: float
    std_dev_sec: float
    min_sec: float
    max_sec: float
    is_dead_cycle: bool
    start_pressure_bar: float
    end_pressure_bar: float
    valve_spool_nominal_pct: float
    description: str

@dataclass
class PressConfig:
    """Nominal parameters for a 28 MN (2,800 MT) high-speed aluminum extrusion press."""
    press_id: str = "Extrusion_Press_28MN_01"
    rated_tonnage_mn: float = 28.0
    nominal_billet_length_mm: float = 850.0
    nominal_billet_diameter_mm: float = 228.0
    nominal_billet_weight_kg: float = 95.0
    
    # Phase sequence and physical nominal timings
    phases: Dict[str, PhaseTiming] = field(default_factory=lambda: {
        "decompression": PhaseTiming(
            name="decompression",
            nominal_sec=2.2,
            std_dev_sec=0.08,
            min_sec=1.8,
            max_sec=3.5,
            is_dead_cycle=True,
            start_pressure_bar=280.0,
            end_pressure_bar=15.0,
            valve_spool_nominal_pct=45.0,
            description="Controlled main cylinder decompression before container release"
        ),
        "container_shift_open": PhaseTiming(
            name="container_shift_open",
            nominal_sec=3.0,
            std_dev_sec=0.12,
            min_sec=2.4,
            max_sec=4.5,
            is_dead_cycle=True,
            start_pressure_bar=15.0,
            end_pressure_bar=60.0,
            valve_spool_nominal_pct=80.0,
            description="Container retracts 400mm from die face"
        ),
        "shear_stroke": PhaseTiming(
            name="shear_stroke",
            nominal_sec=2.5,
            std_dev_sec=0.10,
            min_sec=2.0,
            max_sec=4.0,
            is_dead_cycle=True,
            start_pressure_bar=40.0,
            end_pressure_bar=190.0,
            valve_spool_nominal_pct=90.0,
            description="Hydraulic shear cuts discard/butt end"
        ),
        "die_slide": PhaseTiming(
            name="die_slide",
            nominal_sec=1.8,
            std_dev_sec=0.06,
            min_sec=1.5,
            max_sec=3.0,
            is_dead_cycle=True,
            start_pressure_bar=20.0,
            end_pressure_bar=40.0,
            valve_spool_nominal_pct=60.0,
            description="Die cassette indexing & visual clearance check"
        ),
        "billet_load": PhaseTiming(
            name="billet_load",
            nominal_sec=3.2,
            std_dev_sec=0.15,
            min_sec=2.5,
            max_sec=5.0,
            is_dead_cycle=True,
            start_pressure_bar=20.0,
            end_pressure_bar=55.0,
            valve_spool_nominal_pct=75.0,
            description="Loader arm delivers preheated billet into press centerline"
        ),
        "container_shift_close": PhaseTiming(
            name="container_shift_close",
            nominal_sec=2.8,
            std_dev_sec=0.10,
            min_sec=2.2,
            max_sec=4.2,
            is_dead_cycle=True,
            start_pressure_bar=30.0,
            end_pressure_bar=160.0,
            valve_spool_nominal_pct=85.0,
            description="Container moves forward and seals against die face"
        ),
        "rapid_advance": PhaseTiming(
            name="rapid_advance",
            nominal_sec=2.5,
            std_dev_sec=0.08,
            min_sec=2.0,
            max_sec=3.8,
            is_dead_cycle=True,
            start_pressure_bar=25.0,
            end_pressure_bar=70.0,
            valve_spool_nominal_pct=95.0,
            description="Main ram rapid traverse until dummy block contacts billet"
        ),
        "extrusion": PhaseTiming(
            name="extrusion",
            nominal_sec=65.0,
            std_dev_sec=2.50,
            min_sec=50.0,
            max_sec=85.0,
            is_dead_cycle=False,
            start_pressure_bar=180.0,
            end_pressure_bar=285.0,
            valve_spool_nominal_pct=100.0,
            description="Main ram extrudes aluminum through die at controlled speed"
        )
    })

    @property
    def canonical_phase_names(self) -> List[str]:
        return list(self.phases.keys())

    @property
    def dead_cycle_phase_names(self) -> List[str]:
        return [name for name, p in self.phases.items() if p.is_dead_cycle]

    @property
    def nominal_dead_cycle_duration(self) -> float:
        return sum(p.nominal_sec for p in self.phases.values() if p.is_dead_cycle)

    @property
    def nominal_total_cycle_duration(self) -> float:
        return sum(p.nominal_sec for p in self.phases.values())
