"""Reusable Plotly charts and visual components for the Extrusion Press Dashboard."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
from dashboard.oee_calculator import OEERecoveryMetrics

# Premium industrial color palette tailored for press automation
COLORS = {
    "primary": "#0052CC",       # Industrial Blue
    "secondary": "#172B4D",     # Deep Slate
    "accent": "#00B8D9",        # Hydraulic Cyan
    "warning": "#FFAB00",       # Interlock Amber (Valve Overlap)
    "danger": "#DE350B",        # Micro-Stall Alert Red
    "wear": "#FF5630",          # Creeping Wear Orange
    "success": "#36B37E",       # Nominal Green
    "background": "#F4F5F7",    # Clean Light Grey
    "card_bg": "#FFFFFF",
}

def render_phase_waterfall(events: List[Dict]) -> go.Figure:
    """Renders a horizontal phase timeline comparing nominal duration vs. actual excess delay."""
    df = pd.DataFrame(events)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No cycle events recorded yet")
        return fig

    fig = go.Figure()

    # Nominal base bar
    fig.add_trace(go.Bar(
        y=df["phase_name"],
        x=df["nominal_duration"],
        name="Nominal Execution",
        orientation="h",
        marker=dict(color=COLORS["primary"], opacity=0.85)
    ))

    # Excess delay bar (anomalies)
    if "excess_seconds" in df.columns:
        fig.add_trace(go.Bar(
            y=df["phase_name"],
            x=df["excess_seconds"],
            name="Recoverable Excess Delay",
            orientation="h",
            marker=dict(color=COLORS["danger"], opacity=0.90)
        ))

    fig.update_layout(
        barmode="stack",
        title="<b>Press Cycle Phase Breakdown (Nominal vs. Excess Delay)</b>",
        xaxis_title="Duration (Seconds)",
        yaxis_title="Phase Name",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_phase_duration_scatter(
    df: pd.DataFrame,
    phase_name: str,
    nominal_sec: float,
    threshold_sec: float
) -> go.Figure:
    """Renders a scatter timeline of durations for a specific phase, highlighting anomalies."""
    phase_df = df[df["phase_name"] == phase_name].copy()
    if phase_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No telemetry data for {phase_name}")
        return fig

    fig = go.Figure()

    # Baseline nominal reference line
    fig.add_hline(
        y=nominal_sec,
        line_dash="dot",
        line_color=COLORS["success"],
        annotation_text=f"Nominal ({nominal_sec:.2f}s)",
        annotation_position="bottom right"
    )

    # Upper tolerance / anomaly threshold line
    fig.add_hline(
        y=threshold_sec,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Anomaly Threshold ({threshold_sec:.2f}s)",
        annotation_position="top right"
    )

    # Normal points
    normal_pts = phase_df[~phase_df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal_pts["cycle_id"],
        y=normal_pts["actual_duration"],
        mode="markers",
        name="Normal Cycle",
        marker=dict(color=COLORS["primary"], size=7, opacity=0.7)
    ))

    # Anomalous points
    anom_pts = phase_df[phase_df["is_anomaly"]]
    if not anom_pts.empty:
        fig.add_trace(go.Scatter(
            x=anom_pts["cycle_id"],
            y=anom_pts["actual_duration"],
            mode="markers+text",
            name="Detected Anomaly",
            marker=dict(color=COLORS["danger"], size=10, symbol="x"),
            text=anom_pts["anomaly_type"],
            textposition="top center"
        ))

    fig.update_layout(
        title=f"<b>Cycle Duration History: {phase_name}</b>",
        xaxis_title="Cycle Number",
        yaxis_title="Duration (Seconds)",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_anomaly_breakdown_pie(df: pd.DataFrame) -> go.Figure:
    """Renders a donut chart of detected anomaly classifications."""
    anom_df = df[df["is_anomaly"] & (df["anomaly_type"] != "NONE")]
    if anom_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No anomalies detected yet")
        return fig

    counts = anom_df["anomaly_type"].value_counts().reset_index()
    counts.columns = ["anomaly_type", "count"]

    color_map = {
        "VALVE_OVERLAP": COLORS["warning"],
        "MICRO_STALL": COLORS["danger"],
        "CREEPING_WEAR": COLORS["wear"]
    }
    pie_colors = [color_map.get(t, COLORS["primary"]) for t in counts["anomaly_type"]]

    fig = go.Figure(data=[go.Pie(
        labels=counts["anomaly_type"],
        values=counts["count"],
        hole=0.45,
        marker=dict(colors=pie_colors)
    )])

    fig.update_layout(
        title="<b>Anomaly Root-Cause Distribution</b>",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def render_financial_waterfall(metrics: OEERecoveryMetrics) -> go.Figure:
    """Renders a waterfall chart showing capacity and financial value recovery."""
    fig = go.Figure(go.Waterfall(
        name="Value Recovery",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Direct Press Cost (€)", "Extruded Tonnage Margin (€)", "Total Annual Benefit (€)"],
        textposition="outside",
        text=[
            f"+€{metrics.annual_direct_cost_savings_eur:,.0f}",
            f"+€{metrics.annual_margin_gain_eur:,.0f}",
            f"€{metrics.total_annual_economic_benefit_eur:,.0f}"
        ],
        y=[
            metrics.annual_direct_cost_savings_eur,
            metrics.annual_margin_gain_eur,
            metrics.total_annual_economic_benefit_eur
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": COLORS["danger"]}},
        increasing={"marker": {"color": COLORS["success"]}},
        totals={"marker": {"color": COLORS["primary"]}}
    ))

    fig.update_layout(
        title="<b>Annual Economic Value Recovery Waterfall (€ / Year)</b>",
        showlegend=False,
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    return fig
