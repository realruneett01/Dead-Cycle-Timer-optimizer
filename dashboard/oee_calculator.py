"""OEE, throughput tonnage, and financial capacity recovery calculator for Extrusion Press."""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PlantParameters:
    press_id: str = "Extrusion Press 28 MN #01"
    press_tonnage_rating_mn: float = 28.0
    nominal_billet_weight_kg: float = 95.0
    nominal_cycles_per_hour: float = 43.0
    operating_hours_per_year: float = 7200.0  # 24/7 3-shift operation
    plant_operating_cost_per_hour_eur: float = 950.0  # direct press hourly rate
    aluminum_value_add_per_ton_eur: float = 450.0   # conversion margin per MT

@dataclass
class OEERecoveryMetrics:
    total_cycles_analyzed: int
    total_anomalies_detected: int
    avg_recoverable_sec_per_cycle: float
    annual_planned_cycles: int
    annual_recovered_hours: float
    annual_direct_cost_savings_eur: float
    annual_additional_billets: int
    annual_additional_tonnage_mt: float
    annual_margin_gain_eur: float
    total_annual_economic_benefit_eur: float
    oee_availability_gain_pct: float

class OEECalculator:
    """Calculates industrial capacity and financial ROI from eliminated dead-cycle delays."""

    def __init__(self, params: Optional[PlantParameters] = None):
        self.params = params or PlantParameters()

    def calculate_recovery(
        self,
        total_excess_seconds: float,
        total_cycles: int,
        total_anomalies: int,
        capture_efficiency: float = 0.85
    ) -> OEERecoveryMetrics:
        """
        Translates detected excess seconds into annual press hours, metric tons, and EUR benefits.
        capture_efficiency: fraction of detected excess seconds physically recoverable by automation tuning (e.g. 85%).
        """
        if total_cycles <= 0:
            total_cycles = 1

        avg_excess_per_cycle = (total_excess_seconds / total_cycles) * capture_efficiency
        annual_cycles = int(self.params.operating_hours_per_year * self.params.nominal_cycles_per_hour)
        
        # Annual recovered productive machine hours
        annual_recovered_seconds = annual_cycles * avg_excess_per_cycle
        annual_recovered_hours = annual_recovered_seconds / 3600.0
        
        # Direct machine time cost recovery
        direct_cost_savings = annual_recovered_hours * self.params.plant_operating_cost_per_hour_eur
        
        # Additional billets produced in recovered hours
        additional_billets = int(annual_recovered_hours * self.params.nominal_cycles_per_hour)
        
        # Additional aluminum throughput in metric tons
        additional_tonnage_mt = (additional_billets * self.params.nominal_billet_weight_kg) / 1000.0
        
        # Margin gain from added throughput
        margin_gain = additional_tonnage_mt * self.params.aluminum_value_add_per_ton_eur
        
        # Total economic benefit
        total_benefit = direct_cost_savings + margin_gain
        
        # OEE Availability delta: recovered hours / planned operating hours
        oee_gain_pct = (annual_recovered_hours / self.params.operating_hours_per_year) * 100.0

        return OEERecoveryMetrics(
            total_cycles_analyzed=total_cycles,
            total_anomalies_detected=total_anomalies,
            avg_recoverable_sec_per_cycle=round(avg_excess_per_cycle, 2),
            annual_planned_cycles=annual_cycles,
            annual_recovered_hours=round(annual_recovered_hours, 1),
            annual_direct_cost_savings_eur=round(direct_cost_savings, 2),
            annual_additional_billets=additional_billets,
            annual_additional_tonnage_mt=round(additional_tonnage_mt, 1),
            annual_margin_gain_eur=round(margin_gain, 2),
            total_annual_economic_benefit_eur=round(total_benefit, 2),
            oee_availability_gain_pct=round(oee_gain_pct, 2)
        )
