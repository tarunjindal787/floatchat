"""
Map visualization component — Plotly Mapbox float trajectories and heatmaps.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAPBOX_STYLE = "carto-darkmatter"

OCEAN_REGIONS = {
    "Indian Ocean":   {"lat": -10, "lon": 75,  "zoom": 3},
    "Arabian Sea":    {"lat": 16,  "lon": 64,  "zoom": 4},
    "Bay of Bengal":  {"lat": 13,  "lon": 90,  "zoom": 4},
    "Southern Ocean": {"lat": -55, "lon": 90,  "zoom": 3},
    "Global":         {"lat": 0,   "lon": 0,   "zoom": 1},
}


def float_trajectory_map(df: pd.DataFrame, float_id: str = "") -> go.Figure:
    """
    Animated trajectory map for a single float.
    df must have: latitude, longitude, juld, cycle_number
    """
    if df.empty:
        return _empty_map()

    df = df.sort_values("juld").reset_index(drop=True)
    df["juld_str"] = df["juld"].astype(str).str[:10]

    fig = go.Figure()

    # Track line
    fig.add_trace(go.Scattermapbox(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="lines",
        line=dict(width=2, color="rgba(6,182,212,0.4)"),
        name="Track",
        hoverinfo="skip",
    ))

    # Profile points
    fig.add_trace(go.Scattermapbox(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(
            size=8,
            color=list(range(len(df))),
            colorscale="Viridis",
            colorbar=dict(title="Cycle", thickness=12, len=0.6),
            showscale=True,
        ),
        text=df.apply(
            lambda r: f"Float {float_id}<br>Cycle {r.get('cycle_number', '?')}<br>{r['juld_str']}",
            axis=1,
        ),
        hovertemplate="%{text}<br>Lat: %{lat:.3f}<br>Lon: %{lon:.3f}<extra></extra>",
        name=f"Float {float_id}",
    ))

    # Start / end markers
    fig.add_trace(go.Scattermapbox(
        lat=[df["latitude"].iloc[0],  df["latitude"].iloc[-1]],
        lon=[df["longitude"].iloc[0], df["longitude"].iloc[-1]],
        mode="markers+text",
        marker=dict(size=14, color=["#22c55e", "#ef4444"]),
        text=["Start", "End"],
        textposition="top right",
        hoverinfo="skip",
        name="Start/End",
    ))

    center = dict(lat=float(df["latitude"].mean()), lon=float(df["longitude"].mean()))
    _apply_layout(fig, center=center, zoom=4, title=f"Float {float_id} Trajectory")
    return fig


def profiles_heatmap(df: pd.DataFrame,
                     color_col: str = "mean_temp",
                     title: str = "ARGO Profile Locations") -> go.Figure:
    """
    Scatter map of profile locations coloured by a variable.
    df must have: latitude, longitude, float_id, and color_col.
    """
    if df.empty:
        return _empty_map()

    label_map = {
        "mean_temp":     "Temp (°C)",
        "mean_salinity": "Salinity (PSU)",
        "max_pressure":  "Max Depth (dbar)",
    }

    fig = px.scatter_mapbox(
        df.dropna(subset=["latitude", "longitude"]),
        lat="latitude",
        lon="longitude",
        color=color_col if color_col in df.columns else None,
        color_continuous_scale="Thermal",
        size_max=10,
        zoom=2,
        hover_data={c: True for c in df.columns if c in ["float_id", "juld", color_col]},
        labels={color_col: label_map.get(color_col, color_col)},
        title=title,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    _apply_layout(fig, center={"lat": 0, "lon": 70}, zoom=2)
    return fig


def multi_float_map(trajectories: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Overlay trajectories of multiple floats on one map.
    trajectories: {float_id -> DataFrame with lat/lon/juld}
    """
    colors = px.colors.qualitative.Vivid
    fig = go.Figure()

    for i, (fid, df) in enumerate(trajectories.items()):
        if df.empty:
            continue
        c = colors[i % len(colors)]
        fig.add_trace(go.Scattermapbox(
            lat=df["latitude"], lon=df["longitude"],
            mode="lines+markers",
            marker=dict(size=6, color=c),
            line=dict(width=2, color=c),
            name=f"Float {fid}",
            hovertemplate=f"Float {fid}<br>Lat: %{{lat:.3f}}<br>Lon: %{{lon:.3f}}<extra></extra>",
        ))

    _apply_layout(fig, center={"lat": 0, "lon": 75}, zoom=2,
                  title="Multi-Float Trajectories")
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_map() -> go.Figure:
    fig = go.Figure(go.Scattermapbox())
    _apply_layout(fig, center={"lat": 0, "lon": 75}, zoom=2, title="No data")
    return fig


def _apply_layout(fig: go.Figure, center: dict, zoom: int,
                  title: str = "") -> None:
    fig.update_layout(
        mapbox=dict(style=MAPBOX_STYLE, center=center, zoom=zoom),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#f0f6ff", size=12),
        title=dict(text=title, font=dict(size=15, color="#94a3b8")),
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="rgba(59,130,246,0.2)",
            borderwidth=1,
        ),
    )
