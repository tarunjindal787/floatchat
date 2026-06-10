"""
FloatChat — Main Streamlit Application
Run: streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from loguru import logger

from config.settings import get_settings
from frontend.demo_data import (
    is_db_available, make_demo_profiles, make_demo_ctd,
    make_demo_bgc, make_demo_trajectory,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat — ARGO Ocean AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://argo.ucsd.edu/",
        "Report a bug": "https://github.com/tarunjindal787/floatchat/issues",
        "About":        "FloatChat — AI-powered ARGO oceanographic data interface.",
    },
)

# ── CSS ───────────────────────────────────────────────────────────────────────
css_path = ROOT / "frontend" / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── DB availability check (cached 60s) ────────────────────────────────────────
@st.cache_data(ttl=60)
def _db_ok() -> bool:
    return is_db_available()

DB_AVAILABLE = _db_ok()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_response" not in st.session_state: st.session_state.last_response = None
if "demo_profiles" not in st.session_state: st.session_state.demo_profiles = make_demo_profiles(200)

# Lazy-load heavy backend only when DB is available
if DB_AVAILABLE:
    try:
        from backend.response_generator import ResponseGenerator
        if "generator" not in st.session_state:
            st.session_state.generator = ResponseGenerator()
    except Exception as e:
        logger.warning(f"Backend init failed: {e}")
        DB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <span style="font-size:2.5rem">🌊</span>
            <h2 style="margin:0;background:linear-gradient(135deg,#3b82f6,#06b6d4);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       font-size:1.5rem;font-weight:700;">FloatChat</h2>
            <p style="color:#94a3b8;font-size:0.78rem;margin:4px 0 0;">
                AI-Powered ARGO Ocean Data
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not DB_AVAILABLE:
            st.warning("🟡 **Demo Mode** — showing synthetic Indian Ocean data. Connect a PostgreSQL DB to use real Argo data.", icon="ℹ️")

        st.divider()

        # Stats
        st.markdown("### 📊 Database Stats")
        stats = _get_db_stats()
        c1, c2 = st.columns(2)
        c1.metric("Floats",   stats["floats"])
        c2.metric("Profiles", stats["profiles"])
        c1.metric("Levels",   stats["levels"])
        c2.metric("BGC Obs",  stats["bgc"])

        st.divider()

        # Quick queries
        st.markdown("### 🚀 Quick Queries")
        quick = [
            "Show temperature profiles in the Arabian Sea",
            "Compare salinity in Bay of Bengal vs Arabian Sea",
            "List floats with most profiles in Indian Ocean",
            "Show oxygen levels at 500m depth",
            "Show chlorophyll-a trends last 6 months",
        ]
        for q in quick:
            if st.button(q, key=f"q_{hash(q)}", use_container_width=True):
                st.session_state.quick_query = q

        st.divider()

        # Viz settings
        st.markdown("### ⚙️ Visualization")
        opts = {
            "color_var":   st.selectbox("Colour by",        ["mean_temp","mean_salinity","max_pressure"]),
            "profile_var": st.selectbox("Profile variable", ["temperature","salinity"]),
            "bgc_var":     st.selectbox("BGC variable",     ["doxy","chla","nitrate","bbp700"]),
            "depth_layer": st.selectbox("Depth layer",      ["surface","subsurface","deep"]),
        }

        st.divider()
        s = get_settings()
        st.caption(f"**LLM:** {s.llm_provider} / {s.llm_model}")
        st.caption(f"**Mode:** {'Live DB' if DB_AVAILABLE else 'Demo'}")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

    return opts


# ─────────────────────────────────────────────────────────────────────────────
# CHAT PANEL
# ─────────────────────────────────────────────────────────────────────────────
def render_chat() -> str | None:
    st.markdown("### 💬 Ocean Data Assistant")

    with st.container(height=420):
        if not st.session_state.messages:
            _welcome()
        for m in st.session_state.messages:
            _bubble(m["role"], m["content"])
        if st.session_state.get("thinking"):
            _typing()

    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        with ci:
            user_input = st.text_input(
                "Ask anything…",
                placeholder="e.g. Show salinity profiles near the equator in March 2023",
                label_visibility="collapsed",
            )
        with cb:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    if "quick_query" in st.session_state:
        return st.session_state.pop("quick_query")
    return user_input if submitted and user_input.strip() else None


def _welcome():
    badges = ["🌡 Temperature","🧂 Salinity","🐠 BGC","🗺 Trajectories","📍 Nearest floats","⬇ Export"]
    badge_html = "".join(
        f'<span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);'
        f'padding:4px 12px;border-radius:20px;font-size:0.78rem;color:#93c5fd;">{b}</span>'
        for b in badges
    )
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 1rem;color:#94a3b8;">
        <div style="font-size:3rem;margin-bottom:1rem;">🌊</div>
        <h3 style="color:#f0f6ff;margin-bottom:0.5rem;">Welcome to FloatChat</h3>
        <p style="font-size:0.9rem;margin-bottom:1.5rem;">
            Your AI-powered interface for ARGO oceanographic data.<br>
            Ask questions in plain English!
        </p>
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def _bubble(role: str, content: str):
    icon  = "👤" if role == "user" else "🤖"
    align = "row-reverse" if role == "user" else "row"
    bg    = ("linear-gradient(135deg,rgba(59,130,246,0.12),rgba(139,92,246,0.08))"
             if role == "user" else "rgba(26,34,54,0.9)")
    border= "rgba(59,130,246,0.25)" if role == "user" else "rgba(59,130,246,0.1)"
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-direction:{align};margin-bottom:10px;
                padding:12px 14px;border-radius:12px;background:{bg};
                border:1px solid {border};animation:fadeIn 0.3s ease;">
        <div style="font-size:1.3rem;flex-shrink:0;">{icon}</div>
        <div style="flex:1;line-height:1.7;font-size:0.92rem;word-break:break-word;">
            {content.replace(chr(10),"<br>")}
        </div>
    </div>""", unsafe_allow_html=True)


def _typing():
    st.markdown("""
    <div style="display:flex;gap:10px;align-items:center;padding:12px 14px;
                background:rgba(26,34,54,0.9);border:1px solid rgba(59,130,246,0.1);
                border-radius:12px;margin-bottom:10px;">
        <span style="font-size:1.3rem;">🤖</span>
        <div class="typing-indicator"><span></span><span></span><span></span></div>
        <span style="color:#94a3b8;font-size:0.85rem;">Thinking…</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def handle_query(query: str) -> dict:
    """Generate response — uses real backend if DB available, else demo."""
    if DB_AVAILABLE:
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]
        resp = st.session_state.generator.generate(query, history)
        return {
            "text":       resp.text,
            "data":       resp.data,
            "chart_type": resp.chart_type,
            "sql":        resp.sql_used,
            "sources":    resp.sources,
        }
    else:
        return _demo_response(query)


def _demo_response(query: str) -> dict:
    """Rule-based demo responses with synthetic data."""
    q = query.lower()
    profiles = st.session_state.demo_profiles

    if any(w in q for w in ["trajectory", "track", "path", "drift"]):
        fid  = "6900001"
        data = make_demo_trajectory(fid)
        return {"text": f"🗺 **Demo trajectory** for float **{fid}** — showing {len(data)} synthetic positions across the Indian Ocean (2023). Real data requires a connected PostgreSQL database.",
                "data": data, "chart_type": "trajectory", "sql": None, "sources": []}

    if any(w in q for w in ["bgc", "oxygen", "chlorophyll", "doxy", "chla", "nitrate"]):
        data = make_demo_bgc(1)
        return {"text": "🧪 **Demo BGC profile** — showing synthetic dissolved oxygen, chlorophyll-a, and nitrate profiles for an Indian Ocean float. Real BGC data requires a connected database.",
                "data": data, "chart_type": "bgc", "sql": None, "sources": []}

    if any(w in q for w in ["temperature", "salinity", "ctd", "profile", "depth"]):
        data = make_demo_ctd(1)
        return {"text": "📈 **Demo CTD profile** — showing synthetic temperature (28→4°C) and salinity (35→36.5 PSU) profiles typical of the Arabian Sea. Connect a real database to query actual Argo profiles.",
                "data": data, "chart_type": "profile", "sql": None, "sources": []}

    if any(w in q for w in ["nearest", "closest", "near", "around"]):
        sample = profiles.sample(min(10, len(profiles))).reset_index(drop=True)
        return {"text": f"📍 **Demo mode** — showing {len(sample)} random Indian Ocean float positions. Provide a lat/lon and connect a real DB for actual nearest-float queries.",
                "data": sample, "chart_type": "heatmap", "sql": None, "sources": []}

    # Default — show map of all demo profiles
    return {
        "text": (
            f"🌊 **FloatChat Demo Mode**\n\n"
            f"I'm running with **{len(profiles)} synthetic Indian Ocean profiles** "
            f"across 20 demo floats (2023).\n\n"
            f"To use real ARGO data, connect a **PostgreSQL database** and ingest NetCDF files "
            f"using the ingestion pipeline.\n\n"
            f"Try asking: *'Show temperature profiles'*, *'Show BGC data'*, or *'Show float trajectory'*"
        ),
        "data": profiles, "chart_type": "heatmap", "sql": None, "sources": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
def render_visualizations(resp: dict | None, opts: dict):
    if resp is None:
        _viz_placeholder()
        return

    from frontend.components.map_view    import float_trajectory_map, profiles_heatmap
    from frontend.components.profile_plot import ctd_profile_plot, ts_diagram, depth_time_contour
    from frontend.components.bgc_comparison import bgc_profiles_panel, bgc_timeseries
    from frontend.components.data_export import export_buttons, show_data_table

    df         = resp.get("data")
    chart_type = resp.get("chart_type")

    tab_map, tab_profile, tab_bgc, tab_data = st.tabs(["🗺 Map","📈 Profiles","🧪 BGC","📋 Data"])

    with tab_map:
        if chart_type == "trajectory" and df is not None and not df.empty:
            fid = df["float_id"].iloc[0] if "float_id" in df.columns else "?"
            st.plotly_chart(float_trajectory_map(df, fid), use_container_width=True)
        elif df is not None and not df.empty and "latitude" in df.columns:
            st.plotly_chart(profiles_heatmap(df, color_col=opts["color_var"]), use_container_width=True)
        else:
            st.info("No geospatial data for this query.")

    with tab_profile:
        if df is not None and not df.empty and "pressure" in df.columns:
            t1, t2, t3 = st.tabs(["CTD Profile","T-S Diagram","Depth-Time"])
            with t1: st.plotly_chart(ctd_profile_plot(df), use_container_width=True)
            with t2: st.plotly_chart(ts_diagram(df), use_container_width=True)
            with t3: st.plotly_chart(depth_time_contour(df, var=opts["profile_var"]), use_container_width=True)
        else:
            st.info("No CTD profile data in this result.")

    with tab_bgc:
        if df is not None and not df.empty:
            bgc_cols = [c for c in ["doxy","chla","nitrate","ph_in_situ","bbp700"] if c in df.columns]
            if bgc_cols:
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(bgc_profiles_panel(df), use_container_width=True)
                with c2: st.plotly_chart(bgc_timeseries(df, var=opts["bgc_var"], depth_layer=opts["depth_layer"]), use_container_width=True)
            else:
                st.info("No BGC variables in this result. Try: *'Show oxygen or chlorophyll data'*")

    with tab_data:
        if resp.get("sql"):
            with st.expander("🔍 SQL Used", expanded=False):
                st.code(resp["sql"], language="sql")
        show_data_table(df)
        if df is not None and not df.empty:
            st.markdown("**Export:**")
            export_buttons(df, key_prefix=f"exp_{len(st.session_state.messages)}")
        if resp.get("sources"):
            with st.expander(f"📚 RAG Sources ({len(resp['sources'])})", expanded=False):
                for s in resp["sources"]:
                    st.caption(f"Float {s.get('float_id')} | {s.get('ocean_basin')} | {str(s.get('juld',''))[:10]}")


def _viz_placeholder():
    st.markdown("""
    <div style="padding:3rem;text-align:center;color:#94a3b8;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">🗺</div>
        <p style="font-size:0.95rem;">Visualizations appear here after your first query.</p>
        <p style="font-size:0.8rem;color:#475569;">
            Supports: float trajectories · CTD profiles · T-S diagrams · Hovmöller sections · BGC time series
        </p>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB STATS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _get_db_stats() -> dict:
    if not DB_AVAILABLE:
        demo = st.session_state.get("demo_profiles", make_demo_profiles())
        return {"floats": f"{demo['float_id'].nunique()} (demo)",
                "profiles": f"{len(demo)} (demo)", "levels": "~120K (demo)", "bgc": "~48K (demo)"}
    try:
        from ingestion.db_writer import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            return {
                "floats":   f"{conn.execute(text('SELECT COUNT(*) FROM floats')).scalar():,}",
                "profiles": f"{conn.execute(text('SELECT COUNT(*) FROM profiles')).scalar():,}",
                "levels":   f"{conn.execute(text('SELECT COUNT(*) FROM measurements')).scalar():,}",
                "bgc":      f"{conn.execute(text('SELECT COUNT(*) FROM bgc_data')).scalar():,}",
            }
    except Exception:
        return {"floats":"N/A","profiles":"N/A","levels":"N/A","bgc":"N/A"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    opts = render_sidebar()

    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem;">
        <h1 style="margin:0;font-size:1.8rem;font-weight:700;
                   background:linear-gradient(135deg,#3b82f6,#06b6d4,#14b8a6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🌊 FloatChat
        </h1>
        <p style="color:#94a3b8;margin:4px 0 0;font-size:0.9rem;">
            AI-Powered Conversational Interface for ARGO Ocean Data Discovery & Visualization
        </p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([2, 3], gap="large")

    with left:
        user_input = render_chat()
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.thinking = True
            st.rerun()

    # Process pending user message
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "user":
        with st.spinner(""):
            resp = handle_query(msgs[-1]["content"])
        st.session_state.messages.append({"role": "assistant", "content": resp["text"]})
        st.session_state.last_response = resp
        st.session_state.thinking = False
        st.rerun()

    with right:
        st.markdown("### 📊 Visualizations")
        render_visualizations(st.session_state.last_response, opts)


if __name__ == "__main__":
    main()
