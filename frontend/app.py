"""
FloatChat — Main Streamlit Application
Run: streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from loguru import logger

from config.settings import get_settings
from backend.response_generator import ResponseGenerator, FloatChatResponse
from backend.sql_executor import get_float_trajectory, get_nearest_floats
from frontend.components.map_view import (
    float_trajectory_map, profiles_heatmap, multi_float_map,
)
from frontend.components.profile_plot import (
    ctd_profile_plot, ts_diagram, depth_time_contour,
)
from frontend.components.bgc_comparison import bgc_profiles_panel, bgc_timeseries
from frontend.components.data_export import export_buttons, show_data_table

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat — ARGO Ocean AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://argo.ucsd.edu/",
        "Report a bug": "https://github.com/your-org/floatchat/issues",
        "About":        "FloatChat — AI-powered ARGO oceanographic data interface.",
    },
)

# ── CSS injection ─────────────────────────────────────────────────────────────
css_path = ROOT / "frontend" / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "generator" not in st.session_state:
    st.session_state.generator = ResponseGenerator()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <span style="font-size:2.5rem">🌊</span>
            <h2 style="margin:0; background: linear-gradient(135deg,#3b82f6,#06b6d4);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       font-size:1.5rem; font-weight:700;">FloatChat</h2>
            <p style="color:#94a3b8; font-size:0.78rem; margin:4px 0 0;">
                AI-Powered ARGO Ocean Data
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Quick stats ──────────────────────────────────────────────────────
        st.markdown("### 📊 Database Stats")
        stats = _get_db_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Floats",    stats.get("floats",    "—"))
            st.metric("Profiles",  stats.get("profiles",  "—"))
        with col2:
            st.metric("Levels",    stats.get("levels",    "—"))
            st.metric("BGC Obs",   stats.get("bgc",       "—"))

        st.divider()

        # ── Quick access ─────────────────────────────────────────────────────
        st.markdown("### 🚀 Quick Queries")
        quick_queries = [
            "Show temperature profiles in the Arabian Sea last 3 months",
            "Compare salinity in Bay of Bengal vs Arabian Sea",
            "List 10 floats with most profiles in Indian Ocean",
            "What is the mean oxygen level at 500m depth?",
            "Show BGC parameters for the latest 5 profiles",
        ]
        for q in quick_queries:
            if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                st.session_state.quick_query = q

        st.divider()

        # ── Visualization settings ───────────────────────────────────────────
        st.markdown("### ⚙️ Visualization")
        viz_opts = {
            "map_type":    st.selectbox("Map type", ["heatmap", "trajectory", "multi-float"]),
            "color_var":   st.selectbox("Colour by", ["mean_temp", "mean_salinity", "max_pressure"]),
            "profile_var": st.selectbox("Profile variable", ["temperature", "salinity"]),
            "bgc_var":     st.selectbox("BGC variable", ["doxy", "chla", "nitrate", "bbp700"]),
            "depth_layer": st.selectbox("Depth layer", ["surface", "subsurface", "deep"]),
        }

        st.divider()
        st.markdown("### 🔧 Settings")
        s = get_settings()
        st.caption(f"**LLM:** {s.llm_provider} / {s.llm_model}")
        st.caption(f"**Embedding:** {s.embedding_provider}")
        st.caption(f"**Max rows:** {s.max_sql_rows:,}")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

    return viz_opts


# ── Chat panel ────────────────────────────────────────────────────────────────
def render_chat() -> str | None:
    """Render chat history and input box. Returns new user input if submitted."""
    st.markdown("### 💬 Ocean Data Assistant")

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.messages:
            _render_welcome()
        for msg in st.session_state.messages:
            _render_message(msg["role"], msg["content"])
        if st.session_state.get("is_thinking"):
            _render_typing()

    # Input
    with st.form(key="chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Ask anything about ARGO data…",
                placeholder="e.g. Show salinity profiles near the equator in March 2023",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    # Handle quick query
    if "quick_query" in st.session_state:
        q = st.session_state.pop("quick_query")
        return q

    return user_input if submitted and user_input.strip() else None


def _render_welcome():
    st.markdown("""
    <div style="text-align:center; padding:2rem 1rem; color:#94a3b8;">
        <div style="font-size:3rem; margin-bottom:1rem;">🌊</div>
        <h3 style="color:#f0f6ff; margin-bottom:0.5rem;">Welcome to FloatChat</h3>
        <p style="margin-bottom:1.5rem; font-size:0.9rem;">
            Your AI-powered interface for ARGO oceanographic data.
            Ask questions in plain English!
        </p>
        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center;">
    """ + "".join([
        f'<span style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3); '
        f'padding:4px 12px; border-radius:20px; font-size:0.78rem; color:#93c5fd;">{e}</span>'
        for e in [
            "🌡 Temperature profiles",
            "🧂 Salinity analysis",
            "🐠 BGC parameters",
            "🗺 Float trajectories",
            "📍 Nearest floats",
            "⬇ Data export",
        ]
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_message(role: str, content: str):
    icon = "👤" if role == "user" else "🤖"
    align = "row-reverse" if role == "user" else "row"
    bg = ("linear-gradient(135deg,rgba(59,130,246,0.12),rgba(139,92,246,0.08))"
          if role == "user" else "rgba(26,34,54,0.9)")
    border = ("rgba(59,130,246,0.25)" if role == "user" else "rgba(59,130,246,0.1)")

    st.markdown(f"""
    <div style="display:flex; gap:10px; flex-direction:{align}; margin-bottom:10px;
                padding:12px 14px; border-radius:12px; background:{bg};
                border:1px solid {border}; animation:fadeIn 0.3s ease;">
        <div style="font-size:1.3rem; flex-shrink:0;">{icon}</div>
        <div style="flex:1; line-height:1.7; font-size:0.92rem; word-break:break-word;">
            {content.replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_typing():
    st.markdown("""
    <div style="display:flex; gap:10px; align-items:center; padding:12px 14px;
                background:rgba(26,34,54,0.9); border:1px solid rgba(59,130,246,0.1);
                border-radius:12px; margin-bottom:10px;">
        <span style="font-size:1.3rem;">🤖</span>
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
        <span style="color:#94a3b8; font-size:0.85rem;">Thinking…</span>
    </div>
    """, unsafe_allow_html=True)


# ── Visualization panel ───────────────────────────────────────────────────────
def render_visualizations(resp: FloatChatResponse | None, viz_opts: dict):
    if resp is None:
        _render_placeholder_dashboard()
        return

    df = resp.data
    chart_type = resp.chart_type

    tab_map, tab_profile, tab_bgc, tab_data = st.tabs([
        "🗺 Map", "📈 Profiles", "🧪 BGC", "📋 Data"
    ])

    with tab_map:
        if chart_type == "trajectory" and df is not None and not df.empty:
            fid = resp.chart_config.get("float_id", "")
            st.plotly_chart(float_trajectory_map(df, fid), use_container_width=True)
        elif df is not None and not df.empty and "latitude" in df.columns:
            st.plotly_chart(
                profiles_heatmap(df, color_col=viz_opts["color_var"]),
                use_container_width=True,
            )
        else:
            st.info("No geospatial data available for this query.")

    with tab_profile:
        if df is not None and not df.empty:
            if "pressure" in df.columns and (
                "temperature" in df.columns or "salinity" in df.columns
            ):
                t1, t2, t3 = st.tabs(["CTD Profile", "T-S Diagram", "Depth-Time Section"])
                with t1:
                    st.plotly_chart(ctd_profile_plot(df), use_container_width=True)
                with t2:
                    st.plotly_chart(ts_diagram(df), use_container_width=True)
                with t3:
                    var = viz_opts["profile_var"]
                    st.plotly_chart(depth_time_contour(df, var=var), use_container_width=True)
            else:
                st.info("No CTD profile data in this result set.")
        else:
            st.info("Submit a query to see profile plots.")

    with tab_bgc:
        if df is not None and not df.empty:
            bgc_cols = [c for c in ["doxy", "chla", "nitrate", "ph_in_situ", "bbp700"]
                        if c in df.columns]
            if bgc_cols:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(bgc_profiles_panel(df), use_container_width=True)
                with c2:
                    st.plotly_chart(
                        bgc_timeseries(df, var=viz_opts["bgc_var"],
                                       depth_layer=viz_opts["depth_layer"]),
                        use_container_width=True,
                    )
            else:
                st.info("No BGC variables in this result. Try asking about oxygen or chlorophyll.")
        else:
            st.info("Submit a BGC query to see these charts.")

    with tab_data:
        if resp.sql_used:
            with st.expander("🔍 SQL Used", expanded=False):
                st.code(resp.sql_used, language="sql")
        show_data_table(df)
        if df is not None and not df.empty:
            st.markdown("**Export Data:**")
            export_buttons(df, key_prefix=f"exp_{len(st.session_state.messages)}")

        if resp.sources:
            with st.expander(f"📚 RAG Sources ({len(resp.sources)})", expanded=False):
                for src in resp.sources:
                    st.caption(f"Float {src.get('float_id')} | {src.get('ocean_basin')} | {src.get('juld', '')[:10]}")


def _render_placeholder_dashboard():
    """Shown before first query."""
    st.markdown("""
    <div style="padding:2rem; text-align:center; color:#94a3b8;">
        <div style="font-size:2.5rem; margin-bottom:1rem;">🗺</div>
        <p style="font-size:0.95rem;">
            Visualizations will appear here after you submit a query.
        </p>
        <p style="font-size:0.8rem; color:#475569;">
            Supported: float trajectories · CTD profiles · T-S diagrams ·
            Hovmöller sections · BGC time series
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── DB stats ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _get_db_stats() -> dict:
    try:
        from ingestion.db_writer import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            floats   = conn.execute(text("SELECT COUNT(*) FROM floats")).scalar()
            profiles = conn.execute(text("SELECT COUNT(*) FROM profiles")).scalar()
            levels   = conn.execute(text("SELECT COUNT(*) FROM measurements")).scalar()
            bgc      = conn.execute(text("SELECT COUNT(*) FROM bgc_data")).scalar()
        return {
            "floats":   f"{floats:,}",
            "profiles": f"{profiles:,}",
            "levels":   f"{levels:,}",
            "bgc":      f"{bgc:,}",
        }
    except Exception:
        return {"floats": "N/A", "profiles": "N/A", "levels": "N/A", "bgc": "N/A"}


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    viz_opts = render_sidebar()

    # Header
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
        <h1 style="margin:0; font-size:1.8rem; font-weight:700;
                   background:linear-gradient(135deg,#3b82f6,#06b6d4,#14b8a6);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🌊 FloatChat
        </h1>
        <p style="color:#94a3b8; margin:4px 0 0; font-size:0.9rem;">
            AI-Powered Conversational Interface for ARGO Ocean Data Discovery
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main columns
    left, right = st.columns([2, 3], gap="large")

    with left:
        user_input = render_chat()

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.is_thinking = True
            st.rerun()

    # Process if last message is from user (no assistant reply yet)
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "user":
        query = msgs[-1]["content"]
        history = [{"role": m["role"], "content": m["content"]}
                   for m in msgs[:-1]]
        with st.spinner(""):
            resp = st.session_state.generator.generate(query, history)
        st.session_state.messages.append({"role": "assistant", "content": resp.text})
        st.session_state.last_response = resp
        st.session_state.is_thinking = False
        st.rerun()

    with right:
        st.markdown("### 📊 Visualizations")
        render_visualizations(st.session_state.last_response, viz_opts)


if __name__ == "__main__":
    main()
