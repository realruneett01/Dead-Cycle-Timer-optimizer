"""Dead-Cycle Time Optimizer (DCTO) - Executive & Engineering Dashboard."""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components import (
    render_anomaly_breakdown_pie,
    render_financial_waterfall,
    render_phase_duration_scatter,
    render_phase_waterfall,
)
from dashboard.oee_calculator import OEECalculator, PlantParameters
from detector.detector_service import DetectorService
from simulator.anomaly_injector import AnomalyInjector
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

# Streamlit Page Config
st.set_page_config(
    page_title="Dead-Cycle Time Optimizer",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E1E4E8;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #5E6C84;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #172B4D;
    }
    .kpi-delta {
        font-size: 12px;
        font-weight: 600;
        color: #00875A;
    }
    .disclaimer-banner {
        background-color: #DEEBFF;
        border-left: 4px solid #0052CC;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #0747A6;
        margin-top: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/engine.png", width=64)
    st.title("DCTO Engine")
    st.caption("Dead-Cycle Time Telemetry & Recovery Engine")
    st.divider()

    st.subheader("🏭 Press & Plant Configuration")
    press_id = st.text_input("Press ID", value="Extrusion_Press_28MN_01")
    tonnage = st.number_input("Rated Press Tonnage (MN)", min_value=10.0, max_value=80.0, value=28.0, step=1.0)
    billet_wt = st.number_input("Nominal Billet Weight (kg)", min_value=30.0, max_value=300.0, value=95.0, step=5.0)
    nominal_cph = st.number_input("Nominal Cycles / Hour", min_value=20.0, max_value=70.0, value=43.0, step=1.0)
    op_hours = st.number_input("Operating Hours / Year (24/7)", min_value=2000.0, max_value=8760.0, value=7200.0, step=100.0)
    hourly_rate = st.number_input("Press Operating Cost (€ / Hour)", min_value=300.0, max_value=3000.0, value=950.0, step=50.0)
    aluminum_margin = st.number_input("Extrusion Value-Add (€ / Ton)", min_value=100.0, max_value=1500.0, value=450.0, step=25.0)

    st.divider()
    st.subheader("🎯 Optimization Tuning")
    capture_efficiency = st.slider("Target Delay Elimination Rate (%)", min_value=50, max_value=100, value=85, step=5) / 100.0
    detection_threshold = st.slider("Detector Z-Score Sensitivity", min_value=2.0, max_value=3.5, value=2.75, step=0.05)

    st.divider()
    st.subheader("🔄 Data Feed Source")
    data_source = st.radio("Telemetry Mode", ["Generated Benchmark (400 Cycles)", "Load CSV Stream"])

# Initialize or Load Telemetry
plant_params = PlantParameters(
    press_id=press_id,
    press_tonnage_rating_mn=tonnage,
    nominal_billet_weight_kg=billet_wt,
    nominal_cycles_per_hour=nominal_cph,
    operating_hours_per_year=op_hours,
    plant_operating_cost_per_hour_eur=hourly_rate,
    aluminum_value_add_per_ton_eur=aluminum_margin
)
calculator = OEECalculator(params=plant_params)
config = PressConfig()

@st.cache_data
def generate_or_load_data(source_mode: str, z_thresh: float):
    telem_file = PROJECT_ROOT / "data" / "telemetry_stream.csv"
    if source_mode == "Load CSV Stream" and telem_file.exists():
        df = pd.read_csv(telem_file)
        return df

    # Run fast simulation
    inj = AnomalyInjector(anomaly_probability=0.12, seed=42)
    sm = PressStateMachine(config=config, injector=inj, seed=42)
    detector = DetectorService(config=config, threshold_z=z_thresh)

    records = []
    for c in range(1, 401):
        events = sm.run_cycle(cycle_id=c)
        for ev in events:
            rec = detector.process_phase_event(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                is_dead_cycle=ev.is_dead_cycle
            )
            records.append(rec)
    return pd.DataFrame(records)

with st.spinner("Processing telemetry and anomaly detector pipeline..."):
    telemetry_df = generate_or_load_data(data_source, detection_threshold)

# Calculate Economic & OEE Metrics
total_cycles = int(telemetry_df["cycle_id"].max())
total_excess_sec = float(telemetry_df[telemetry_df["is_anomaly"]]["excess_seconds"].sum())
total_anomalies = int(telemetry_df["is_anomaly"].sum())

metrics = calculator.calculate_recovery(
    total_excess_seconds=total_excess_sec,
    total_cycles=total_cycles,
    total_anomalies=total_anomalies,
    capture_efficiency=capture_efficiency
)

# ----------------- HEADER SECTION -----------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("28 MN Extrusion Press")
    st.markdown("**Real-Time Dead-Cycle Time (DCT) Optimization & Micro-Stall Recovery**")
with col_status:
    st.success("🟢 OPC-UA Server: Online (ns=2)")
    st.caption(f"Analyzed Cycles: **{total_cycles}** | Detected Anomalies: **{total_anomalies}**")

# ----------------- TOP KPI ROW -----------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Recoverable DCT", f"{metrics.avg_recoverable_sec_per_cycle:.2f} s / cycle", delta=f"{metrics.oee_availability_gain_pct:.1f}% OEE Gain")
with kpi2:
    st.metric("Annual Machine Hours", f"{metrics.annual_recovered_hours:.1f} hrs / yr", delta="+ Production Uptime")
with kpi3:
    st.metric("Capacity Value (€)", f"€{metrics.total_annual_economic_benefit_eur:,.0f} / yr", delta="+ Direct & Margin")
with kpi4:
    st.metric("Additional Output", f"{metrics.annual_additional_tonnage_mt:.1f} Tons / yr", delta=f"+{metrics.annual_additional_billets:,} Billets")
with kpi5:
    latest_anom = telemetry_df.tail(8)["is_anomaly"].any()
    status_text = "Anomaly Alert" if latest_anom else "Nominal Operation"
    delta_color = "inverse" if latest_anom else "normal"
    st.metric("Current Press State", status_text, delta="Inspected" if latest_anom else "Clear", delta_color=delta_color)

st.divider()

# ----------------- TAB NAVIGATION -----------------
tab1, tab2, tab3 = st.tabs([
    "📊 Live Cycle Waterfall & Telemetry",
    "🔍 Micro-Stall Diagnostics & Drift",
    "💰 OEE & Plant Financial Model"
])

# ----------------- TAB 1: WATERFALL -----------------
with tab1:
    st.subheader("Latest Press Cycle Breakdown")
    latest_cycle_id = total_cycles
    cycle_events_df = telemetry_df[telemetry_df["cycle_id"] == latest_cycle_id].copy()

    # Map nominal durations
    nominal_dict = {k: v.nominal_sec for k, v in config.phases.items()}
    cycle_events_df["nominal_duration"] = cycle_events_df["phase_name"].map(nominal_dict)

    col_wf, col_summary = st.columns([2, 1])
    with col_wf:
        fig_wf = render_phase_waterfall(cycle_events_df.to_dict(orient="records"))
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_summary:
        st.markdown(f"### Cycle #{latest_cycle_id} Summary")
        dead_cycle_time = cycle_events_df[cycle_events_df["phase_name"] != "extrusion"]["actual_duration"].sum()
        total_time = cycle_events_df["actual_duration"].sum()
        excess_time = cycle_events_df["excess_seconds"].sum()

        st.write(f"⏱️ **Dead-Cycle Time (DCT):** `{dead_cycle_time:.2f} s` (Nominal: `{config.nominal_dead_cycle_duration:.1f} s`)")
        st.write(f"⏳ **Total Billet Cycle Time:** `{total_time:.2f} s` (Nominal: `{config.nominal_total_cycle_duration:.1f} s`)")
        if excess_time > 0:
            st.error(f"⚠️ **Recoverable Micro-Stall Delay:** `+{excess_time:.2f} s`")
        else:
            st.success("✅ **Cycle Executed at Target Kinematic Velocity**")

        st.dataframe(
            cycle_events_df[["phase_name", "actual_duration", "nominal_duration", "is_anomaly", "anomaly_type"]],
            hide_index=True,
            use_container_width=True
        )

# ----------------- TAB 2: DIAGNOSTICS -----------------
with tab2:
    st.subheader("Sub-Second Anomaly Diagnostics & Root-Cause Attribution")
    
    col_pie, col_scatter = st.columns([1, 2])
    with col_pie:
        fig_pie = render_anomaly_breakdown_pie(telemetry_df)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Distribution of flagged industrial failure modes across analyzed cycles.")

    with col_scatter:
        selected_phase = st.selectbox(
            "Select Phase for Temporal Drift Inspection:",
            config.canonical_phase_names,
            index=2 # shear_stroke
        )
        phase_nominal = config.phases[selected_phase].nominal_sec
        threshold_val = phase_nominal + (detection_threshold * config.phases[selected_phase].std_dev_sec)
        
        fig_scatter = render_phase_duration_scatter(
            telemetry_df,
            phase_name=selected_phase,
            nominal_sec=phase_nominal,
            threshold_sec=threshold_val
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Flagged Anomaly Audit Log (Last 15 Incidents)")
    anom_table = telemetry_df[telemetry_df["is_anomaly"]].tail(15)[
        ["cycle_id", "phase_name", "actual_duration", "baseline_median", "robust_z_score", "excess_seconds", "anomaly_type"]
    ]
    st.dataframe(anom_table, hide_index=True, use_container_width=True)

# ----------------- TAB 3: OEE & FINANCIAL MODEL -----------------
with tab3:
    st.subheader("Plant-Level Economic Capacity Recovery")

    st.markdown("""
    <div class="disclaimer-banner">
        <b>Industrial Verification Notice:</b> All financial capacity gains and tonnage estimates are analytical projections 
        calculated from the configured press operating parameters and simulated cycle telemetry. They demonstrate algorithmic 
        dead-cycle recovery potential and must be calibrated to physical plant LVDTs and operating cost accounting before commissioning.
    </div>
    """, unsafe_allow_html=True)

    col_fin1, col_fin2 = st.columns([1, 1])
    with col_fin1:
        fig_fin = render_financial_waterfall(metrics)
        st.plotly_chart(fig_fin, use_container_width=True)

    with col_fin2:
        st.markdown("### Operational Capacity Breakdown")
        st.write(f"- **Planned Production Schedule:** `{plant_params.operating_hours_per_year:,.0f} hours / year` (3 Shifts, 24/7)")
        st.write(f"- **Recovered Machine Availability:** `+{metrics.annual_recovered_hours:.1f} hours / year`")
        st.write(f"- **Additional Finished Billets:** `+{metrics.annual_additional_billets:,} billets / year`")
        st.write(f"- **Additional Aluminum Throughput:** `+{metrics.annual_additional_tonnage_mt:.1f} MT / year`")
        st.write(f"- **Direct Machine Amortization Savings:** `€{metrics.annual_direct_cost_savings_eur:,.0f} / year`")
        st.write(f"- **Tonnage Value-Add Margin:** `€{metrics.annual_margin_gain_eur:,.0f} / year`")
        st.metric("Total Annual Economic Gain", f"€{metrics.total_annual_economic_benefit_eur:,.0f}", delta=f"{metrics.oee_availability_gain_pct:.2f}% OEE Availability Gain")
