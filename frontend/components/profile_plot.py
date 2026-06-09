"""
Profile plot component — CTD depth profiles, T-S diagrams, and time-series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0e1a",
    font=dict(family="Inter", color="#f0f6ff", size=12),
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis=dict(gridcolor="rgba(59,130,246,0.1)", zerolinecolor="rgba(59,130,246,0.2)"),
    yaxis=dict(gridcolor="rgba(59,130,246,0.1)", zerolinecolor="rgba(59,130,246,0.2)"),
)


def ctd_profile_plot(df: pd.DataFrame, profile_ids: list[int] | None = None,
                     title: str = "CTD Profiles") -> go.Figure:
    """
    Vertical temperature and salinity profiles.
    df columns: pressure, temperature, salinity (+ optional profile_id)
    """
    if df.empty:
        return go.Figure().update_layout(**DARK_LAYOUT, title="No profile data")

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=["Temperature (°C)", "Salinity (PSU)"],
        horizontal_spacing=0.08,
    )

    colors = px.colors.qualitative.Vivid
    groups = df.groupby("profile_id") if "profile_id" in df.columns else [(0, df)]

    for i, (pid, grp) in enumerate(groups):
        grp = grp.sort_values("pressure")
        c   = colors[i % len(colors)]

        if "temperature" in grp.columns:
            fig.add_trace(go.Scatter(
                x=grp["temperature"], y=grp["pressure"],
                mode="lines",
                name=f"Profile {pid} — T",
                line=dict(color=c, width=2),
                hovertemplate="T: %{x:.2f}°C<br>P: %{y:.0f} dbar<extra></extra>",
            ), row=1, col=1)

        if "salinity" in grp.columns:
            fig.add_trace(go.Scatter(
                x=grp["salinity"], y=grp["pressure"],
                mode="lines",
                name=f"Profile {pid} — S",
                line=dict(color=c, width=2, dash="dot"),
                showlegend=False,
                hovertemplate="S: %{x:.3f} PSU<br>P: %{y:.0f} dbar<extra></extra>",
            ), row=1, col=2)

    fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=1)
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#94a3b8")),
        height=500,
        legend=dict(bgcolor="rgba(17,24,39,0.8)", borderwidth=1),
    )
    return fig


def ts_diagram(df: pd.DataFrame, title: str = "T-S Diagram") -> go.Figure:
    """
    Temperature-Salinity scatter coloured by pressure (water mass identification).
    """
    if df.empty or "temperature" not in df.columns or "salinity" not in df.columns:
        return go.Figure().update_layout(**DARK_LAYOUT, title="No T-S data")

    df = df.dropna(subset=["temperature", "salinity"])

    fig = px.scatter(
        df, x="salinity", y="temperature",
        color="pressure" if "pressure" in df.columns else None,
        color_continuous_scale="Thermal",
        labels={"salinity": "Salinity (PSU)", "temperature": "Temperature (°C)",
                "pressure": "Pressure (dbar)"},
        title=title,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(**DARK_LAYOUT, height=450)
    return fig


def depth_time_contour(df: pd.DataFrame,
                       var: str = "temperature",
                       title: str = "Depth-Time Section") -> go.Figure:
    """
    Hovmöller diagram: time on X, depth on Y, variable as colour.
    df columns: juld, pressure, <var>
    """
    if df.empty or var not in df.columns:
        return go.Figure().update_layout(**DARK_LAYOUT, title="No data for contour")

    df = df.dropna(subset=["pressure", var]).copy()
    df["juld"] = pd.to_datetime(df["juld"])

    pivot = df.pivot_table(index="pressure", columns="juld", values=var, aggfunc="mean")

    colorscale = "RdBu_r" if var == "temperature" else "Haline"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index,
        colorscale=colorscale,
        colorbar=dict(title=var.capitalize(), thickness=14, len=0.7),
        hovertemplate="Date: %{x}<br>Depth: %{y} dbar<br>Value: %{z:.3f}<extra></extra>",
    ))

    fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)")
    fig.update_xaxes(title_text="Date")
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#94a3b8")),
        height=450,
    )
    return fig


def multi_profile_comparison(profiles: dict[str, pd.DataFrame],
                              var: str = "temperature") -> go.Figure:
    """
    Overlay profiles from different regions/floats for comparison.
    profiles: {label -> DataFrame with pressure + var}
    """
    colors = px.colors.qualitative.Vivid
    fig = go.Figure()

    for i, (label, df) in enumerate(profiles.items()):
        if df.empty or var not in df.columns:
            continue
        df = df.sort_values("pressure")
        fig.add_trace(go.Scatter(
            x=df[var], y=df["pressure"],
            mode="lines",
            name=label,
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate=f"{label}<br>{var}: %{{x:.2f}}<br>Depth: %{{y:.0f}} dbar<extra></extra>",
        ))

    fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)")
    unit = "°C" if var == "temperature" else "PSU" if var == "salinity" else ""
    fig.update_xaxes(title_text=f"{var.capitalize()} ({unit})")
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"{var.capitalize()} Profile Comparison", font=dict(size=15, color="#94a3b8")),
        height=500,
        legend=dict(bgcolor="rgba(17,24,39,0.8)", borderwidth=1),
    )
    return fig
