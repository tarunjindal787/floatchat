"""
BGC comparison charts — dissolved oxygen, chlorophyll, nitrate, etc.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0e1a",
    font=dict(family="Inter", color="#f0f6ff", size=12),
    margin=dict(l=60, r=20, t=50, b=50),
)

BGC_META = {
    "doxy":       {"label": "Dissolved Oxygen", "unit": "µmol/kg", "color": "#22d3ee"},
    "chla":       {"label": "Chlorophyll-a",    "unit": "mg/m³",   "color": "#4ade80"},
    "nitrate":    {"label": "Nitrate",           "unit": "µmol/kg", "color": "#fb923c"},
    "ph_in_situ": {"label": "pH (in situ)",      "unit": "",        "color": "#f472b6"},
    "bbp700":     {"label": "Backscatter 700nm", "unit": "m⁻¹",    "color": "#a78bfa"},
}


def bgc_profiles_panel(df: pd.DataFrame, title: str = "BGC Profiles") -> go.Figure:
    """
    Multi-panel vertical BGC profiles (one panel per variable).
    df columns: pressure + any subset of BGC_META keys.
    """
    avail = [v for v in BGC_META if v in df.columns and df[v].notna().any()]
    if not avail:
        return go.Figure().update_layout(**DARK, title="No BGC data available")

    n = len(avail)
    fig = make_subplots(rows=1, cols=n,
                        shared_yaxes=True,
                        subplot_titles=[BGC_META[v]["label"] for v in avail],
                        horizontal_spacing=0.06)

    for i, var in enumerate(avail, 1):
        meta = BGC_META[var]
        grp  = df.dropna(subset=[var, "pressure"]).sort_values("pressure")

        fig.add_trace(go.Scatter(
            x=grp[var], y=grp["pressure"],
            mode="lines+markers",
            marker=dict(size=4, color=meta["color"]),
            line=dict(color=meta["color"], width=2),
            name=meta["label"],
            hovertemplate=(
                f"{meta['label']}: %{{x:.3f}} {meta['unit']}"
                "<br>Depth: %{y:.0f} dbar<extra></extra>"
            ),
        ), row=1, col=i)

    fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=1)
    fig.update_layout(**DARK, height=500,
                      title=dict(text=title, font=dict(size=15, color="#94a3b8")),
                      showlegend=False)
    return fig


def bgc_timeseries(df: pd.DataFrame, var: str = "doxy",
                   depth_layer: str = "surface") -> go.Figure:
    """
    Time series of a BGC variable at a given depth layer.
    depth_layer: 'surface' (< 50 dbar), 'subsurface' (50–200), 'deep' (> 200)
    """
    meta = BGC_META.get(var, {"label": var, "unit": "", "color": "#60a5fa"})

    depth_filters = {
        "surface":    (0,   50),
        "subsurface": (50,  200),
        "deep":       (200, 9999),
    }
    dmin, dmax = depth_filters.get(depth_layer, (0, 50))
    layer_df = df[(df["pressure"] >= dmin) & (df["pressure"] < dmax)]

    if layer_df.empty or var not in layer_df.columns:
        return go.Figure().update_layout(**DARK, title=f"No {meta['label']} data")

    agg = layer_df.groupby("juld")[var].mean().reset_index()
    agg = agg.sort_values("juld")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["juld"], y=agg[var],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(meta['color'])}, 0.15)",
        line=dict(color=meta["color"], width=2.5),
        marker=dict(size=5),
        name=meta["label"],
        hovertemplate=f"{meta['label']}: %{{y:.3f}} {meta['unit']}<extra></extra>",
    ))

    fig.update_layout(
        **DARK,
        title=dict(
            text=f"{meta['label']} — {depth_layer.capitalize()} Layer Time Series",
            font=dict(size=15, color="#94a3b8"),
        ),
        xaxis_title="Date",
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        height=380,
    )
    return fig


def bgc_region_comparison(data: dict[str, pd.DataFrame], var: str = "doxy") -> go.Figure:
    """
    Box plot comparing a BGC variable across different regions/floats.
    data: {region_label -> DataFrame with var column}
    """
    meta = BGC_META.get(var, {"label": var, "unit": "", "color": "#60a5fa"})
    colors = px.colors.qualitative.Vivid

    fig = go.Figure()
    for i, (label, df) in enumerate(data.items()):
        if df.empty or var not in df.columns:
            continue
        fig.add_trace(go.Box(
            y=df[var].dropna(),
            name=label,
            boxmean=True,
            marker_color=colors[i % len(colors)],
            line_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        **DARK,
        title=dict(text=f"{meta['label']} Regional Comparison", font=dict(size=15, color="#94a3b8")),
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        height=420,
        showlegend=True,
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
