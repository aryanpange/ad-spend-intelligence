import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AdIntel — Spend Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES — DARK PREMIUM THEME
# ─────────────────────────────────────────────────────────────────────────────

# Fonts injected separately — combining <link> + <style> in one st.markdown()
# causes Streamlit to render the CSS as visible text on screen.
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800'
    '&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600'
    '&family=DM+Mono:wght@400;500'
    '&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200'
    '&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""<style>

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* Make Material Symbols render as icons everywhere Streamlit uses them */
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: #070B13 !important;
    color: #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────────── */
/* Use visibility:hidden (not display:none) on the top bar so the DOM
   node remains alive — Streamlit 1.5x attaches the sidebar toggle button
   as a sibling/child of the toolbar. display:none removes it from layout
   entirely and the reopen button disappears with it. */
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
.stDeployButton { display: none !important; }
div[data-testid="stSidebarNav"] { display: none !important; }

/* Make the top toolbar transparent but keep it in the DOM */
[data-testid="stToolbar"] {
    background: transparent !important;
    /* Do NOT use display:none or visibility:hidden on the toolbar itself —
       the collapsed-sidebar button lives here and will vanish with it. */
}

/* ── Sidebar collapsed-control (reopen arrow) ─────────────────────────────── */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] > *,
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg path {
    display: revert !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
[data-testid="collapsedControl"] {
    z-index: 999999 !important;
    background: #09101E !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 3px 0 16px rgba(0,0,0,0.5) !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(59,130,246,0.14) !important;
    border-color: rgba(59,130,246,0.45) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg path {
    fill: #60A5FA !important;
    stroke: none !important;
}
/* Hide the raw Material Icons ligature text ("keyboard_double_arrow_left")
   that appears when the Material Icons font is not loaded */
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] .material-icons,
[data-testid="collapsedControl"] .material-symbols-rounded {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}

/* ── Main container ───────────────────────────────────────────────────────── */
[data-testid="block-container"] {
    padding: 1.8rem 2.8rem 3rem !important;
    max-width: 1680px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #09101E !important;
    border-right: 1px solid rgba(59,130,246,0.1) !important;
    /* Do NOT set min-width/max-width — those prevent Streamlit collapsing
       the sidebar to 0 and the reopen toggle never appears. */
    width: 255px !important;
}
/* Only apply the fixed width when the sidebar is actually expanded */
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 255px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #64748B !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Radio nav */
.stRadio > div {
    gap: 2px !important;
    padding: 0 10px !important;
}
.stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    color: #475569 !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
}
.stRadio > div > label:hover {
    background: rgba(59,130,246,0.07) !important;
    color: #7DD3FC !important;
}
.stRadio > div > label[data-checked="true"],
.stRadio > div > label[aria-checked="true"] {
    background: rgba(59,130,246,0.1) !important;
    color: #93C5FD !important;
    border-left: 2px solid #3B82F6 !important;
    border-radius: 0 8px 8px 0 !important;
    padding-left: 12px !important;
}
/* Hide the radio dot */
.stRadio > div > label > div:first-child { display: none !important; }

/* ── Main-area widgets ────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox [data-baseweb="select"] span,
.stMultiSelect [data-baseweb="select"] span {
    color: #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Multiselect selected tags (pills) ────────────────────────────────────── */
/* The coloured pill that appears when an item is selected */
[data-baseweb="tag"] {
    background: rgba(59,130,246,0.15) !important;
    border: 1px solid rgba(59,130,246,0.28) !important;
    border-radius: 6px !important;
}
/* Tag label text — make it readable on the dark pill */
[data-baseweb="tag"] span[title],
[data-baseweb="tag"] span {
    color: #93C5FD !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}
/* Tag close (×) button */
[data-baseweb="tag"] button,
[data-baseweb="tag"] [role="button"] {
    color: #60A5FA !important;
    opacity: 0.7 !important;
}
[data-baseweb="tag"] button:hover,
[data-baseweb="tag"] [role="button"]:hover {
    opacity: 1 !important;
}

/* ── Selectbox / dropdown option list ─────────────────────────────────────── */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"] li,
[data-baseweb="option"] {
    background: transparent !important;
    color: #94A3B8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="option"]:hover {
    background: rgba(59,130,246,0.1) !important;
    color: #93C5FD !important;
}
/* Highlighted / currently selected option in dropdown list */
[aria-selected="true"][data-baseweb="option"],
[data-baseweb="option"][aria-selected="true"] {
    background: rgba(59,130,246,0.15) !important;
    color: #60A5FA !important;
}
/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #3B82F6 !important;
    border-color: #3B82F6 !important;
}

/* Date input */
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stDateInput"] input {
    background: #111827 !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
}

/* ── DataFrames ───────────────────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stDataFrame > div {
    background: #0D1520 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

/* ── Plotly charts ────────────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border-radius: 14px !important;
}
[data-testid="stPlotlyChart"] > div {
    border-radius: 14px !important;
}

/* ── Download button ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    color: #93C5FD !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stDownloadButton > button:hover {
    background: rgba(59,130,246,0.16) !important;
    border-color: rgba(59,130,246,0.4) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* ── Alert / info boxes ───────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(59,130,246,0.07) !important;
    border: 1px solid rgba(59,130,246,0.18) !important;
    border-radius: 10px !important;
    color: #93C5FD !important;
}

/* ── HR ───────────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 1.5rem 0 !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   COMPONENT CLASSES
   ════════════════════════════════════════════════════════════════════════════ */

/* ── Page header ──────────────────────────────────────────────────────────── */
.page-header {
    padding: 0 0 28px 0;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.page-eyebrow {
    font-size: 10.5px;
    font-weight: 600;
    color: #3B82F6;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 10px;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(110deg, #F1F5F9 0%, #93C5FD 70%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 14px;
    color: #475569;
    margin: 10px 0 0 0;
    line-height: 1.65;
    max-width: 680px;
}

/* ── Section header ───────────────────────────────────────────────────────── */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin: 2.2rem 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 14px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%);
}

/* ── KPI Cards ────────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(150deg, #111827 0%, #0D1520 100%);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 16px;
    padding: 22px 22px 18px;
    position: relative;
    overflow: hidden;
    min-height: 105px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(59,130,246,0.2); }
.kpi-top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-orb {
    position: absolute;
    top: -25px; right: -25px;
    width: 85px; height: 85px;
    border-radius: 50%;
    opacity: 0.1;
    pointer-events: none;
}
.kpi-label {
    font-size: 10.5px;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 10px;
    position: relative;
}
.kpi-value {
    font-size: 27px;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'Syne', sans-serif;
    line-height: 1.1;
    position: relative;
    letter-spacing: -0.5px;
}
.kpi-delta-pos {
    font-size: 11.5px;
    color: #34D399;
    font-weight: 500;
    margin-top: 5px;
    position: relative;
}
.kpi-delta-neg {
    font-size: 11.5px;
    color: #F87171;
    font-weight: 500;
    margin-top: 5px;
    position: relative;
}
.kpi-delta-neu {
    font-size: 11.5px;
    color: #475569;
    font-weight: 500;
    margin-top: 5px;
    position: relative;
}

/* ── Insight / callout box ────────────────────────────────────────────────── */
.insight-box {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 12px;
    padding: 16px 22px;
    font-size: 13.5px;
    color: #94A3B8;
    line-height: 1.75;
    margin: 18px 0 8px;
}
.insight-box b { color: #60A5FA; }

/* ── Alert feed cards ─────────────────────────────────────────────────────── */
.alert-critical {
    background: rgba(127,29,29,0.18);
    border: 1px solid rgba(239,68,68,0.22);
    border-left: 3px solid #EF4444;
    border-radius: 10px;
    padding: 11px 14px;
    margin: 5px 0;
    font-size: 12.5px;
    color: #E2E8F0;
    line-height: 1.55;
}
.alert-high {
    background: rgba(120,53,15,0.18);
    border: 1px solid rgba(251,146,60,0.22);
    border-left: 3px solid #FB923C;
    border-radius: 10px;
    padding: 11px 14px;
    margin: 5px 0;
    font-size: 12.5px;
    color: #E2E8F0;
    line-height: 1.55;
}
.alert-medium {
    background: rgba(92,60,0,0.18);
    border: 1px solid rgba(251,191,36,0.18);
    border-left: 3px solid #FBBF24;
    border-radius: 10px;
    padding: 11px 14px;
    margin: 5px 0;
    font-size: 12.5px;
    color: #E2E8F0;
    line-height: 1.55;
}
.alert-low {
    background: rgba(6,78,59,0.15);
    border: 1px solid rgba(52,211,153,0.18);
    border-left: 3px solid #34D399;
    border-radius: 10px;
    padding: 11px 14px;
    margin: 5px 0;
    font-size: 12.5px;
    color: #E2E8F0;
    line-height: 1.55;
}
.sev-badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    vertical-align: middle;
    margin-right: 4px;
}
.sev-critical { background: rgba(239,68,68,0.2);  color: #FCA5A5; }
.sev-high     { background: rgba(251,146,60,0.2); color: #FED7AA; }
.sev-medium   { background: rgba(251,191,36,0.2); color: #FEF08A; }
.sev-low      { background: rgba(52,211,153,0.2); color: #A7F3D0; }
.ch-tag {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 10px;
    background: rgba(255,255,255,0.06);
    color: #64748B;
    font-family: 'DM Mono', monospace;
    vertical-align: middle;
}

/* ── Attribution model card ───────────────────────────────────────────────── */
.model-card {
    background: #0D1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 18px;
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.6;
    margin-top: 10px;
}
.model-card b { color: #60A5FA; }

/* ── Stat row for sidebar bottom ──────────────────────────────────────────── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
}
.stat-row .stat-label { font-size: 12px; color: #334155; }
.stat-row .stat-val   { font-size: 12px; font-family: 'DM Mono', monospace; color: #475569; }
.stat-row .stat-val-hi { font-size: 13px; font-family: 'DM Mono', monospace; color: #60A5FA; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ── Remove browser-native title tooltips from sidebar toggle ─────────────────
# CSS cannot suppress native OS tooltips from title= attributes.
# We use a MutationObserver to strip the title attribute whenever it appears.
components.html("""
<script>
(function() {
    function stripTitles() {
        // Target the collapsedControl button and its children
        var targets = document.querySelectorAll(
            '[data-testid="collapsedControl"], ' +
            '[data-testid="collapsedControl"] *, ' +
            '[data-testid="stSidebarCollapseButton"], ' +
            '[data-testid="stSidebarCollapseButton"] *'
        );
        targets.forEach(function(el) {
            if (el.hasAttribute('title')) el.removeAttribute('title');
            if (el.hasAttribute('aria-label')) {
                // Keep aria-label for accessibility but make it empty
                // so screen readers still work but browser tooltip won't show
            }
        });
    }

    // Run immediately in case elements already exist
    stripTitles();

    // Watch for DOM changes (Streamlit re-renders components)
    var obs = new MutationObserver(function(mutations) {
        stripTitles();
    });
    obs.observe(document.documentElement, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['title']
    });
})();
</script>
""", height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Vibrant palette tuned for dark backgrounds
CHANNEL_COLORS = {
    'Google Search':        '#60A5FA',
    'Meta Ads':             '#818CF8',
    'Programmatic Display': '#FB923C',
    'YouTube':              '#F87171',
    'Affiliate':            '#34D399',
}

CHANNELS = list(CHANNEL_COLORS.keys())

SEVERITY_COLORS = {
    'Critical': '#EF4444',
    'High':     '#FB923C',
    'Medium':   '#FBBF24',
    'Low':      '#34D399',
    'Normal':   '#334155',
}

SEVERITY_ORDER = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Normal': 4}

# KPI card glow colours per metric (cycles through)
_KPI_GLOWS = ['#3B82F6', '#34D399', '#A78BFA', '#FB923C',
              '#60A5FA', '#F59E0B', '#34D399', '#818CF8']


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _page_header(eyebrow: str, title: str, subtitle: str = "") -> str:
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="page-header">
        <div class="page-eyebrow">{eyebrow}</div>
        <h1 class="page-title">{title}</h1>
        {sub}
    </div>"""


def _section(text: str) -> str:
    return f'<div class="section-header">{text}</div>'


def _kpi(label: str, value: str, delta: str = None,
         positive: bool = True, glow: str = "#3B82F6") -> str:
    if delta:
        cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
        arr = "▲" if positive else "▼"
        d_html = f'<div class="{cls}">{arr} {delta}</div>'
    else:
        d_html = ""
    # top gradient bar colours derived from glow colour
    return f"""
    <div class="kpi-card">
        <div class="kpi-top-bar" style="background:linear-gradient(90deg,{glow}88,{glow}22);"></div>
        <div class="kpi-orb" style="background:{glow};"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {d_html}
    </div>"""

def _dark_layout(**overrides) -> dict:
    """Base dark chart layout — merge overrides on top.
    Note: xaxis/yaxis are intentionally excluded so callers can pass
    them directly to update_layout() without 'multiple values' errors."""
    base = dict(
        plot_bgcolor='rgba(13,21,32,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#64748B', family='DM Sans', size=12),
        hoverlabel=dict(
            bgcolor='#1E293B',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#E2E8F0', size=12, family='DM Sans'),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8', size=11),
            orientation='h',
            y=1.1,
        ),
        hovermode='x unified',
        margin=dict(l=4, r=4, t=30, b=4),
    )
    base.update(overrides)
    return base


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a 6-digit hex color to rgba() string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_campaign_data():
    path = os.path.join(os.path.dirname(__file__), 'data', 'campaign_data.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    df['month']     = df['date'].dt.to_period('M').astype(str)
    df['month_num'] = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['week']      = df['date'].dt.isocalendar().week.astype(int)
    return df


@st.cache_data
def load_ground_truth():
    path = os.path.join(os.path.dirname(__file__), 'data', 'ground_truth_anomalies.csv')
    return pd.read_csv(path, parse_dates=['date'])


@st.cache_data
def load_attribution_results():
    path = os.path.join(
        os.path.dirname(__file__),
        'outputs', 'attribution_data', 'attribution_results.csv'
    )
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_alert_report():
    path = os.path.join(
        os.path.dirname(__file__),
        'outputs', 'anomaly_data', 'alert_report.csv'
    )
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def build_anomaly_features(df):
    df_sorted = df.sort_values(['channel', 'date']).copy()
    eps = 1e-6

    def add_features(group):
        group = group.copy()
        for window, label in [(7, '7d'), (14, '14d')]:
            group[f'roll_mean_spend_{label}'] = group['spend'].rolling(window, min_periods=3).mean()
            group[f'roll_std_spend_{label}']  = group['spend'].rolling(window, min_periods=3).std()
            group[f'roll_mean_roas_{label}']  = group['roas'].rolling(window, min_periods=3).mean()
            group[f'roll_mean_cpc_{label}']   = group['cpc'].rolling(window, min_periods=3).mean()
            group[f'roll_mean_ctr_{label}']   = group['ctr'].rolling(window, min_periods=3).mean()
            group[f'roll_mean_cvr_{label}']   = group['cvr'].rolling(window, min_periods=3).mean()
        group['spend_ratio_7d']  = group['spend'] / (group['roll_mean_spend_7d']  + eps)
        group['spend_ratio_14d'] = group['spend'] / (group['roll_mean_spend_14d'] + eps)
        group['roas_ratio_7d']   = group['roas']  / (group['roll_mean_roas_7d']   + eps)
        group['cpc_ratio_7d']    = group['cpc']   / (group['roll_mean_cpc_7d']    + eps)
        group['ctr_ratio_7d']    = group['ctr']   / (group['roll_mean_ctr_7d']    + eps)
        group['cvr_ratio_7d']    = group['cvr']   / (group['roll_mean_cvr_7d']    + eps)
        group['spend_zscore']    = ((group['spend'] - group['roll_mean_spend_7d']) /
                                    (group['roll_std_spend_7d'] + eps))
        group['spend_roas_div']  = group['spend_ratio_7d'] - group['roas_ratio_7d']
        group['dow_num']         = group['date'].dt.dayofweek
        group['month_num_feat']  = group['date'].dt.month
        return group

    # Explicit loop — pandas 3.0 removes the groupby key from .apply() results
    groups = []
    for _, grp in df_sorted.groupby('channel'):
        groups.append(add_features(grp.copy()))
    df_feat = pd.concat(groups)

    ratio_cols = [c for c in df_feat.columns if 'ratio' in c or 'zscore' in c or 'div' in c]
    roll_cols  = [c for c in df_feat.columns if 'roll_' in c]
    df_feat[ratio_cols] = df_feat[ratio_cols].fillna(1.0)
    df_feat[roll_cols]  = df_feat[roll_cols].bfill().fillna(0)
    return df_feat


@st.cache_data
def run_isolation_forest(df_feat, contamination=0.03):
    FEATURES = [
        'spend', 'roas', 'cpc', 'ctr', 'cvr',
        'spend_ratio_7d', 'spend_ratio_14d',
        'roas_ratio_7d', 'cpc_ratio_7d',
        'ctr_ratio_7d', 'cvr_ratio_7d',
        'spend_zscore', 'spend_roas_div',
        'dow_num', 'month_num_feat',
    ]
    all_preds = []
    for channel in CHANNELS:
        ch = df_feat[df_feat['channel'] == channel].copy()
        if len(ch) < 5:   # skip channels absent from filtered data
            continue
        scaler = StandardScaler()
        X      = scaler.fit_transform(ch[FEATURES].fillna(0))
        model  = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=42, n_jobs=-1
        )
        model.fit(X)
        ch['if_prediction']    = model.predict(X)
        ch['if_anomaly_score'] = model.decision_function(X)
        ch['if_is_anomaly']    = (ch['if_prediction'] == -1).astype(int)
        all_preds.append(ch)

    return pd.concat(all_preds, ignore_index=True).sort_values(['date', 'channel'])


def classify_anomaly_label(row):
    if row['if_is_anomaly'] == 0:
        return 'Normal', 'Normal', 'No action required', 0

    eps         = 1e-6
    spend_ratio = row.get('spend_ratio_7d', 1.0)
    roas_ratio  = row.get('roas_ratio_7d',  1.0)
    cpc_ratio   = row.get('cpc_ratio_7d',   1.0)
    ctr_ratio   = row.get('ctr_ratio_7d',   1.0)
    cvr_ratio   = row.get('cvr_ratio_7d',   1.0)
    spend       = row['spend']
    roll_spend  = row.get('roll_mean_spend_7d', spend)

    if spend_ratio > 2.0:
        if roas_ratio < 0.6:
            return ('Budget Spike + ROI Collapse', 'Critical',
                    'Pause campaign immediately. Spend 2x+ normal, ROAS collapsed. '
                    'Check bidding strategy and audience targeting.',
                    round(max(spend - roll_spend, 0), 2))
        return ('Budget Spike', 'High',
                'Review budget caps. Spend 2x+ normal. '
                'Check for accidental cap removal or bidding error.',
                round(max(spend - roll_spend, 0), 2))

    if spend_ratio < 0.25:
        if spend < 100:
            return ('Zero Traffic / Account Issue', 'Critical',
                    'Check platform account immediately. Campaign may be paused, '
                    'billing failed, or account flagged.',
                    round(max(roll_spend - spend, 0), 2))
        return ('Budget Crash', 'High',
                'Spend >75% below normal. Check for manual pause, '
                'budget exhaustion, or billing issue.',
                round(max(roll_spend - spend, 0), 2))

    if cpc_ratio > 1.8 and 0.6 < spend_ratio < 2.0:
        impact = (row['cpc'] - row.get('roll_mean_cpc_7d', row['cpc'])) * row['clicks']
        return ('CPC Spike — Bid Strategy Issue', 'Medium',
                'CPC 80%+ above normal. Check competitor activity, '
                'broad match triggers, or automated bid changes.',
                round(max(impact, 0), 2))

    if cvr_ratio < 0.35 and row['clicks'] > 20:
        impact = (row.get('roll_mean_cvr_7d', row['cvr']) - row['cvr']) * row['clicks'] * 1200
        return ('CVR Collapse — Funnel Break', 'High',
                'CVR 65%+ below normal with normal click volume. '
                'Check landing page, checkout flow, offer changes.',
                round(max(impact, 0), 2))

    if roas_ratio < 0.45:
        impact = (row.get('roll_mean_roas_7d', row['roas']) - row['roas']) * spend
        return ('ROAS Deterioration — ROI Risk', 'Medium',
                'ROAS 55%+ below normal. Review audience targeting, '
                'creative fatigue, and bid strategy.',
                round(max(impact, 0), 2))

    if ctr_ratio < 0.40 and row['impressions'] > 1000:
        return ('CTR Drop — Creative Fatigue', 'Low',
                'CTR 60%+ below normal with sufficient impressions. '
                'Refresh ad creatives and review frequency caps.',
                round(spend * 0.15, 2))

    return ('Unusual Pattern — Monitor', 'Low',
            'Multiple metrics show mild deviation. Monitor for 3 days.',
            round(abs(spend - roll_spend) * 0.5, 2))


@st.cache_data
def build_full_anomaly_df(df_feat, contamination):
    df_pred = run_isolation_forest(df_feat, contamination)
    results = df_pred.apply(
        lambda r: pd.Series(
            classify_anomaly_label(r),
            index=['anomaly_label', 'severity', 'recommended_action', 'impact_inr']
        ),
        axis=1,
    )
    return pd.concat([df_pred, results], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION MODELS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_journeys(n=8000):
    np.random.seed(42)
    FIRST_W = {'YouTube': 0.28, 'Programmatic Display': 0.25,
               'Meta Ads': 0.22, 'Google Search': 0.15, 'Affiliate': 0.10}
    LAST_W  = {'Google Search': 0.35, 'Affiliate': 0.28, 'Meta Ads': 0.18,
               'YouTube': 0.12, 'Programmatic Display': 0.07}
    MID_W   = {'Meta Ads': 0.30, 'YouTube': 0.25, 'Google Search': 0.22,
               'Programmatic Display': 0.15, 'Affiliate': 0.08}
    records = []
    for _ in range(n):
        n_t  = np.random.choice([1,2,3,4,5], p=[0.22,0.30,0.28,0.14,0.06])
        tp   = [np.random.choice(list(FIRST_W.keys()), p=list(FIRST_W.values()))]
        for __ in range(n_t - 2):
            tp.append(np.random.choice(list(MID_W.keys()), p=list(MID_W.values())))
        if n_t > 1:
            tp.append(np.random.choice(list(LAST_W.keys()), p=list(LAST_W.values())))
        cvr  = min(0.28 + (n_t-1)*0.04 + (0.12 if tp[-1] in ['Google Search','Affiliate'] else 0), 0.72)
        conv = np.random.random() < cvr
        rev  = np.random.uniform(800,2500) * (1+(n_t-1)*0.08) if conv else 0.0
        records.append({'touchpoints': tp, 'n_touches': n_t,
                        'converted': conv, 'revenue': round(rev,2)})
    return pd.DataFrame(records)


def compute_attribution(journeys_df, model='Linear'):
    results = {ch: 0.0 for ch in CHANNELS}
    conv    = journeys_df[journeys_df['converted'] == True]
    for _, row in conv.iterrows():
        tp  = row['touchpoints']
        rev = row['revenue']
        n   = len(tp)
        if model == 'First Touch':
            results[tp[0]] += rev
        elif model == 'Last Touch':
            results[tp[-1]] += rev
        elif model == 'Linear':
            share = rev / n
            for ch in tp: results[ch] += share
        elif model == 'Time Decay':
            w     = [2**i for i in range(n)]
            total = sum(w)
            for i, ch in enumerate(tp): results[ch] += rev * w[i] / total
        elif model == 'Position Based':
            if n == 1:
                results[tp[0]] += rev
            elif n == 2:
                results[tp[0]] += rev * 0.5; results[tp[-1]] += rev * 0.5
            else:
                results[tp[0]]  += rev * 0.40; results[tp[-1]] += rev * 0.40
                mid = (rev * 0.20) / (n-2)
                for ch in tp[1:-1]: results[ch] += mid
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:

        # ── Logo ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:24px 20px 20px;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,#3B82F6,#8B5CF6);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">⚡</div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:#F1F5F9;letter-spacing:-0.3px;line-height:1;">AdIntel</div>
                    <div style="font-size:9.5px;color:#1E3A5F;font-family:'DM Mono',monospace;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px;">Spend Intelligence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Nav label ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:12px 20px 6px;">
            <div style="font-size:9.5px;color:#1E3A5F;font-family:'DM Mono',monospace;letter-spacing:0.15em;text-transform:uppercase;">Navigation</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["📊  Executive Overview",
             "🔗  Attribution Analysis",
             "🚨  Anomaly Monitor",
             "💰  Budget Pacing"],
            label_visibility="collapsed",
        )

        # ── Filters ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:20px 20px 6px;margin-top:8px;border-top:1px solid rgba(255,255,255,0.04);">
            <div style="font-size:9.5px;color:#1E3A5F;font-family:'DM Mono',monospace;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px;">Filters</div>
        </div>
        """, unsafe_allow_html=True)

        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        start_date, end_date = (date_range if len(date_range) == 2
                                else (min_date, max_date))

        selected_channels = st.multiselect(
            "Channels",
            options=CHANNELS,
            default=CHANNELS,
        )
        if not selected_channels:
            selected_channels = CHANNELS

        # ── Bottom dataset stats ──────────────────────────────────────────────
        total_spend = df['spend'].sum()
        st.markdown(f"""
        <div style="margin-top:24px;padding:16px 10px 8px;border-top:1px solid rgba(255,255,255,0.04);">
            <div style="font-size:9.5px;color:#1E3A5F;font-family:'DM Mono',monospace;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px;">Dataset</div>
            <div class="stat-row"><span class="stat-label">Rows</span><span class="stat-val">2,740</span></div>
            <div class="stat-row"><span class="stat-label">Channels</span><span class="stat-val">5</span></div>
            <div class="stat-row"><span class="stat-label">Period</span><span class="stat-val">18 months</span></div>
            <div class="stat-row" style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.04);">
                <span class="stat-label">Total Spend</span>
                <span class="stat-val-hi">₹{total_spend/10_000_000:.2f}Cr</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Strip emoji prefix before returning
    page_clean = page.split("  ", 1)[-1].strip()
    return page_clean, start_date, end_date, selected_channels


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def page_executive_overview(df):
    st.markdown(_page_header(
        "Performance Dashboard",
        "Executive Overview",
        "Top-level campaign performance across all channels and the full selected date range."
    ), unsafe_allow_html=True)

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    total_spend       = df['spend'].sum()
    total_revenue     = df['revenue'].sum()
    total_conversions = df['conversions'].sum()
    blended_roas      = total_revenue / total_spend if total_spend > 0 else 0
    total_clicks      = df['clicks'].sum()
    avg_cpc           = df[df['cpc'] > 0]['cpc'].mean()
    avg_cvr           = df[df['cvr'] > 0]['cvr'].mean()
    total_impressions = df['impressions'].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi("Total Spend",       f"₹{total_spend/100_000:.1f}L",  glow='#3B82F6'), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi("Total Revenue",     f"₹{total_revenue/100_000:.1f}L", glow='#34D399'), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi("Blended ROAS",      f"{blended_roas:.2f}×",           glow='#A78BFA'), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("Conversions",       f"{total_conversions:,}",          glow='#FB923C'), unsafe_allow_html=True)

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(_kpi("Total Clicks",      f"{total_clicks:,}",              glow='#60A5FA'), unsafe_allow_html=True)
    with c6:
        st.markdown(_kpi("Avg CPC",           f"₹{avg_cpc:.0f}",               glow='#F59E0B'), unsafe_allow_html=True)
    with c7:
        st.markdown(_kpi("Avg CVR",           f"{avg_cvr*100:.2f}%",            glow='#34D399'), unsafe_allow_html=True)
    with c8:
        st.markdown(_kpi("Impressions",       f"{total_impressions/1_000_000:.1f}M", glow='#818CF8'), unsafe_allow_html=True)

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    # ── Revenue & Spend trend + spend mix ────────────────────────────────────
    st.markdown(_section("Revenue & Spend Trend"), unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        monthly = (df.groupby('month')
                     .agg(spend=('spend','sum'), revenue=('revenue','sum'))
                     .reset_index().sort_values('month'))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=monthly['month'], y=monthly['spend']/100_000,
            name='Spend (₹L)',
            marker_color=_hex_to_rgba('#3B82F6', 0.45),
            marker_line_color=_hex_to_rgba('#3B82F6', 0.6),
            marker_line_width=0.5,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['revenue']/100_000,
            name='Revenue (₹L)',
            line=dict(color='#34D399', width=2.5),
            mode='lines+markers',
            marker=dict(size=5, color='#34D399', line=dict(color='#070B13', width=2)),
        ), secondary_y=True)
        fig.update_layout(**_dark_layout(height=320))
        fig.update_yaxes(title_text="Spend (₹L)", secondary_y=False,
                 tickprefix='₹', ticksuffix='L',
                 gridcolor='rgba(255,255,255,0.04)',
                 tickfont=dict(color='#475569', size=11),
                 title=dict(font=dict(color='#475569', size=11)))
        fig.update_yaxes(title_text="Revenue (₹L)", secondary_y=True,
                 showgrid=False,
                 tickprefix='₹', ticksuffix='L',
                 tickfont=dict(color='#475569', size=11),
                 title=dict(font=dict(color='#475569', size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        ch_spend = (df.groupby('channel')['spend'].sum()
                      .reset_index().sort_values('spend', ascending=False))
        fig2 = go.Figure(go.Pie(
            labels=ch_spend['channel'],
            values=ch_spend['spend'],
            hole=0.62,
            marker=dict(
                colors=[CHANNEL_COLORS[c] for c in ch_spend['channel']],
                line=dict(color='#070B13', width=2.5),
            ),
            textinfo='label+percent',
            textfont=dict(size=11, color='#94A3B8'),
            hovertemplate='%{label}<br>₹%{value:,.0f}<br>%{percent}<extra></extra>',
        ))
        fig2.update_layout(
            **_dark_layout(height=320, hovermode='closest'),
            showlegend=False,
        )
        # Donut centre annotation
        fig2.add_annotation(
            text="Spend<br>Mix", x=0.5, y=0.5,
            font=dict(size=13, color='#475569', family='DM Mono'),
            showarrow=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Channel scorecard ─────────────────────────────────────────────────────
    st.markdown(_section("Channel Performance Scorecard"), unsafe_allow_html=True)

    scorecard = df.groupby('channel').agg(
        Spend        = ('spend',       'sum'),
        Revenue      = ('revenue',     'sum'),
        Conversions  = ('conversions', 'sum'),
        Avg_CPC      = ('cpc',         'mean'),
        Avg_CTR      = ('ctr',         'mean'),
        Avg_CVR      = ('cvr',         'mean'),
    ).round(3).reset_index()
    scorecard['ROAS']          = (scorecard['Revenue'] / scorecard['Spend']).round(2)
    scorecard['Cost_Per_Conv'] = (scorecard['Spend']   / scorecard['Conversions']).round(0)
    scorecard['Spend_%']       = (scorecard['Spend']   / scorecard['Spend'].sum() * 100).round(1)
    scorecard['Rev_%']         = (scorecard['Revenue'] / scorecard['Revenue'].sum() * 100).round(1)

    disp = scorecard[['channel','Spend','Revenue','ROAS','Avg_CPC',
                       'Avg_CTR','Avg_CVR','Cost_Per_Conv','Spend_%','Rev_%']]\
            .sort_values('ROAS', ascending=False).copy()
    disp['Spend']        = disp['Spend'].apply(lambda x: f"₹{x:,.0f}")
    disp['Revenue']      = disp['Revenue'].apply(lambda x: f"₹{x:,.0f}")
    disp['Avg_CPC']      = disp['Avg_CPC'].apply(lambda x: f"₹{x:.0f}")
    disp['Avg_CTR']      = disp['Avg_CTR'].apply(lambda x: f"{x:.2%}")
    disp['Avg_CVR']      = disp['Avg_CVR'].apply(lambda x: f"{x:.2%}")
    disp['Cost_Per_Conv']= disp['Cost_Per_Conv'].apply(lambda x: f"₹{x:,.0f}")
    disp['Spend_%']      = disp['Spend_%'].apply(lambda x: f"{x:.1f}%")
    disp['Rev_%']        = disp['Rev_%'].apply(lambda x: f"{x:.1f}%")
    disp.columns = ['Channel','Spend','Revenue','ROAS','Avg CPC',
                    'Avg CTR','Avg CVR','Cost / Conv','Spend %','Rev %']
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── ROAS trend ────────────────────────────────────────────────────────────
    st.markdown(_section("Monthly ROAS Trend by Channel"), unsafe_allow_html=True)
    monthly_ch = (df.groupby(['month','channel'])['roas']
                    .mean().reset_index().sort_values('month'))
    fig3 = px.line(
        monthly_ch, x='month', y='roas', color='channel',
        color_discrete_map=CHANNEL_COLORS, markers=True,
        labels={'roas': 'Average ROAS', 'month': 'Month', 'channel': 'Channel'},
    )
    fig3.add_hline(y=1.0, line_dash='dash',
                   line_color='rgba(239,68,68,0.45)', line_width=1.5,
                   annotation_text='Break-even',
                   annotation_font=dict(color='#EF4444', size=10),
                   annotation_position='top right')
    fig3.update_traces(line=dict(width=2),
                       marker=dict(size=5, line=dict(color='#070B13', width=1.5)))
    fig3.update_layout(**_dark_layout(height=340))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Key insight:</b> November shows the highest blended ROAS across all channels
        (Diwali festive effect). Brands that maintain budget discipline through Oct–Nov
        earn disproportionate returns. January and February are the weakest months —
        reallocation away from these months toward October improves overall efficiency.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ATTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def page_attribution_analysis():
    st.markdown(_page_header(
        "Multi-touch Attribution",
        "Attribution Analysis",
        "Compare 5 attribution models and quantify the misattribution gap "
        "between platform self-reporting and a fair data-driven model."
    ), unsafe_allow_html=True)

    journeys_df = generate_journeys(8000)

    MODELS = ['First Touch', 'Last Touch', 'Linear', 'Time Decay', 'Position Based']
    MODEL_DESC = {
        'First Touch':    '100% credit to the first channel. Values awareness channels that begin journeys.',
        'Last Touch':     '100% credit to the final channel. Default for most ad platforms — creates the most bias.',
        'Linear':         'Equal credit to all touchpoints. Fairest baseline — no channel is favoured.',
        'Time Decay':     'Recent touches get exponentially more credit. Rewards channels that close the sale.',
        'Position Based': '40% first / 40% last / 20% shared. Recognises both discovery and closing channels.',
    }

    col1, col2 = st.columns([2, 3])
    with col1:
        model_a = st.selectbox("Primary model", MODELS, index=2)
        st.markdown(f'<div class="model-card"><b>{model_a}</b><br>{MODEL_DESC[model_a]}</div>',
                    unsafe_allow_html=True)
    with col2:
        compare_models = st.multiselect(
            "Compare against",
            [m for m in MODELS if m != model_a],
            default=['Last Touch'],
        )

    st.markdown(_section("Attributed Revenue by Model & Channel"), unsafe_allow_html=True)

    models_to_show = [model_a] + compare_models
    attribution_results = {m: compute_attribution(journeys_df, m) for m in models_to_show}
    attr_df = pd.DataFrame(attribution_results, index=CHANNELS)

    model_palette = ['#60A5FA', '#F87171', '#34D399', '#FBBF24', '#A78BFA']
    fig = go.Figure()
    for i, model in enumerate(models_to_show):
        fig.add_trace(go.Bar(
            name=model,
            x=CHANNELS,
            y=(attr_df[model] / 100_000).round(2),
            marker_color=model_palette[i % len(model_palette)],
            opacity=0.85,
            hovertemplate=f'<b>%{{x}}</b><br>{model}: ₹%{{y:.2f}}L<extra></extra>',
        ))
    fig.update_layout(
        **_dark_layout(height=380),
        barmode='group',
        yaxis=dict(title='Attributed Revenue (₹L)', tickprefix='₹', ticksuffix='L',
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
        xaxis=dict(title='Channel', tickangle=0, tickfont=dict(color='#64748B', size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Misattribution gap ────────────────────────────────────────────────────
    st.markdown(_section("Misattribution Gap — Last Touch vs Linear"), unsafe_allow_html=True)
    st.caption("Red = channel overvalued by Last Touch.  Green = undervalued and starved of deserved budget.")

    lt_rev     = pd.Series(compute_attribution(journeys_df, 'Last Touch'))
    lin_rev    = pd.Series(compute_attribution(journeys_df, 'Linear'))
    gap_series = ((lt_rev - lin_rev) / 100_000).round(2)
    bar_colors = ['#F87171' if v > 0 else '#34D399' for v in gap_series.values]

    fig2 = go.Figure(go.Bar(
        x=gap_series.values, y=gap_series.index,
        orientation='h',
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.85,
        hovertemplate='<b>%{y}</b><br>Gap: ₹%{x:.2f}L<extra></extra>',
        text=[f'₹{v:+.2f}L' for v in gap_series.values],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11),
    ))
    fig2.add_vline(x=0, line_color='rgba(255,255,255,0.15)', line_width=1.5)
    fig2.update_layout(
        **_dark_layout(height=280, hovermode='closest', margin=dict(l=4, r=80, t=20, b=4)),
        xaxis=dict(title='Gap in ₹ Lakhs (Last Touch minus Linear)',
                   tickprefix='₹', ticksuffix='L', tickangle=0,
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Gap table ─────────────────────────────────────────────────────────────
    gap_table = pd.DataFrame({
        'Channel':         gap_series.index,
        'Last Touch (₹L)': (lt_rev  / 100_000).round(2).values,
        'Linear (₹L)':     (lin_rev / 100_000).round(2).values,
        'Gap (₹L)':        gap_series.values,
        'Gap %':           ((lt_rev - lin_rev) / (lin_rev + 1e-6) * 100).round(1).values,
        'Verdict': [
            'Overvalued'    if v > 1  else
            'Undervalued'   if v < -1 else
            'Fairly valued'
            for v in gap_series.values
        ],
    })
    st.dataframe(gap_table, use_container_width=True, hide_index=True)

    # ── Journey analytics ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(_section("Customer Journey Analytics"), unsafe_allow_html=True)

    j_analysis = journeys_df.groupby('n_touches').agg(
        Journeys    = ('converted', 'count'),
        Converted   = ('converted', 'sum'),
        Avg_Revenue = ('revenue',   'mean'),
    ).reset_index()
    j_analysis['CVR %']   = (j_analysis['Converted'] / j_analysis['Journeys'] * 100).round(1)
    j_analysis['Avg Rev'] = j_analysis['Avg_Revenue'].round(0)

    c1, c2 = st.columns(2)
    with c1:
        fig3 = px.bar(j_analysis, x='n_touches', y='CVR %',
                      color='CVR %', color_continuous_scale='Blues',
                      text='CVR %',
                      labels={'n_touches': 'Touchpoints', 'CVR %': 'Conversion Rate (%)'})
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                           marker_line_width=0)
        fig3.update_layout(**_dark_layout(height=280, hovermode='closest'),
                           showlegend=False,
                           title=dict(text='CVR by Journey Length', font=dict(color='#94A3B8', size=13)))
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = px.bar(j_analysis, x='n_touches', y='Avg Rev',
                      color='Avg Rev', color_continuous_scale='Greens',
                      text='Avg Rev',
                      labels={'n_touches': 'Touchpoints', 'Avg Rev': 'Avg Revenue (₹)'})
        fig4.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside',
                           marker_line_width=0)
        fig4.update_layout(**_dark_layout(height=280, hovermode='closest'),
                           showlegend=False,
                           title=dict(text='Avg Order Value by Journey Length', font=dict(color='#94A3B8', size=13)))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Key insight:</b> Customers with more touchpoints convert at a higher rate AND
        spend more per order. This directly justifies retargeting spend — cutting mid-funnel
        channels to save cost would collapse both CVR and average order value.
    </div>""", unsafe_allow_html=True)

    # ── Budget reallocation ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(_section("Budget Reallocation Simulator"), unsafe_allow_html=True)
    st.caption("Adjust the slider to see projected ROAS improvement from reallocating budget away from overvalued channels.")

    realloc_pct = st.slider(
        "% of budget to reallocate from overvalued → undervalued channels",
        min_value=0, max_value=30, value=10, step=1, format="%d%%",
    )
    overvalued   = gap_table[gap_table['Verdict'] == 'Overvalued']['Channel'].tolist()
    undervalued  = gap_table[gap_table['Verdict'] == 'Undervalued']['Channel'].tolist()
    base_roas    = 2.08
    proj_uplift  = realloc_pct * 0.014
    proj_roas    = base_roas * (1 + proj_uplift)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overvalued channels",  ', '.join(overvalued)  if overvalued  else 'None')
    c2.metric("Undervalued channels", ', '.join(undervalued) if undervalued else 'None')
    c3.metric("Projected ROAS", f"{proj_roas:.2f}×",
              delta=f"+{proj_uplift*100:.1f}% vs current {base_roas:.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ANOMALY MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def page_anomaly_monitor(df):
    st.markdown(_page_header(
        "ML-Powered Detection",
        "Anomaly Monitor",
        "Real-time Isolation Forest anomaly detection across all channels. "
        "Every alert is classified, severity-scored, and action-tagged."
    ), unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 1])
    with col_l:
        contamination = st.slider(
            "Model sensitivity (contamination rate)",
            min_value=0.01, max_value=0.08,
            value=0.03, step=0.005, format="%.3f",
            help="Higher = more alerts flagged. Lower = only the most extreme anomalies.",
        )
    with col_r:
        severity_filter = st.multiselect(
            "Severity filter",
            ['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium'],
        )

    with st.spinner("Running Isolation Forest across all channels…"):
        df_feat = build_anomaly_features(df)
        df_pred = build_full_anomaly_df(df_feat, contamination)

    flagged = df_pred[df_pred['if_is_anomaly'] == 1].copy()
    flagged = flagged[flagged['severity'].isin(severity_filter)]
    flagged = flagged.sort_values('severity', key=lambda s: s.map(SEVERITY_ORDER))

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_alerts      = len(flagged)
    critical_count    = (flagged['severity'] == 'Critical').sum()
    high_count        = (flagged['severity'] == 'High').sum()
    total_impact      = flagged['impact_inr'].sum()
    channels_affected = flagged['channel'].nunique() if len(flagged) > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_kpi("Total Alerts",     str(total_alerts),        glow='#3B82F6'), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi("Critical",         str(critical_count),      glow='#EF4444'), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi("High Severity",    str(high_count),          glow='#FB923C'), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("Channels Hit",     str(channels_affected),   glow='#FBBF24'), unsafe_allow_html=True)
    with c5:
        st.markdown(_kpi("Spend at Risk",    f"₹{total_impact/100_000:.1f}L", glow='#F87171'), unsafe_allow_html=True)

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Alert feed + Timeline ─────────────────────────────────────────────────
    col_feed, col_chart = st.columns([2, 3])

    with col_feed:
        st.markdown(_section("Live Alert Feed (Top 25)"), unsafe_allow_html=True)

        if len(flagged) == 0:
            st.info("No anomalies detected at the current sensitivity level.")
        else:
            feed_df = flagged.nlargest(25, 'impact_inr').sort_values(
                'severity', key=lambda s: s.map(SEVERITY_ORDER)
            )
            for _, row in feed_df.iterrows():
                sev      = row['severity']
                css_key  = sev.lower()
                date_str = pd.Timestamp(row['date']).strftime('%d %b %Y')
                badge_cls = f"sev-{css_key}"
                st.markdown(f"""
                <div class="alert-{css_key}">
                    <div style="margin-bottom:4px;">
                        <span class="sev-badge {badge_cls}">{sev}</span>
                        <span class="ch-tag">{row['channel']}</span>
                        <span style="float:right;font-size:11px;color:#475569;font-family:'DM Mono',monospace;">{date_str}</span>
                    </div>
                    <div style="font-weight:600;font-size:12.5px;color:#E2E8F0;margin-bottom:3px;">{row['anomaly_label']}</div>
                    <div style="font-size:12px;color:#94A3B8;">
                        Spend: <b style="color:#CBD5E1;">₹{row['spend']:,.0f}</b>
                        &nbsp;·&nbsp; Ratio: <b style="color:#CBD5E1;">{row.get('spend_ratio_7d',1.0):.2f}×</b>
                        &nbsp;·&nbsp; Impact: <b style="color:#F87171;">₹{row['impact_inr']:,.0f}</b>
                    </div>
                    <div style="font-size:11.5px;color:#475569;margin-top:4px;">{row['recommended_action']}</div>
                </div>""", unsafe_allow_html=True)

    with col_chart:
        ch_sel = st.selectbox("Select channel for timeline", CHANNELS, index=0,
                              key='anomaly_timeline_channel')
        st.markdown(_section(f"Spend Timeline — {ch_sel}"), unsafe_allow_html=True)

        ch_data = df_pred[df_pred['channel'] == ch_sel].sort_values('date')

        if len(ch_data) == 0:
            st.warning(f"No data for {ch_sel} in the selected date range / filters.")
        else:
            anom = ch_data[ch_data['if_is_anomaly'] == 1]
            fig  = go.Figure()

            # Rolling average
            fig.add_trace(go.Scatter(
                x=ch_data['date'], y=ch_data['roll_mean_spend_7d'],
                name='7-day avg',
                line=dict(color='rgba(148,163,184,0.4)', width=1.5, dash='dot'),
                hovertemplate='%{x}<br>7-day avg: ₹%{y:,.0f}<extra></extra>',
            ))

            # Actual spend fill
            fill_rgba = _hex_to_rgba(CHANNEL_COLORS[ch_sel], 0.1)
            fig.add_trace(go.Scatter(
                x=ch_data['date'], y=ch_data['spend'],
                name='Daily spend',
                line=dict(color=CHANNEL_COLORS[ch_sel], width=1.5),
                opacity=0.75,
                fill='tozeroy',
                fillcolor=fill_rgba,
                hovertemplate='%{x}<br>Spend: ₹%{y:,.0f}<extra></extra>',
            ))

            # Severity markers
            for sev, symbol, size, col in [
                ('Critical', 'star',        18, '#EF4444'),
                ('High',     'triangle-up', 12, '#FB923C'),
                ('Medium',   'square',       9, '#FBBF24'),
                ('Low',      'circle',       7, '#34D399'),
            ]:
                rows = anom[anom['severity'] == sev]
                if len(rows) > 0:
                    fig.add_trace(go.Scatter(
                        x=rows['date'], y=rows['spend'],
                        mode='markers',
                        name=f'{sev} ({len(rows)})',
                        marker=dict(symbol=symbol, size=size, color=col,
                                    line=dict(color='#070B13', width=1.5)),
                        customdata=np.stack([
                            rows['anomaly_label'],
                            rows['impact_inr'],
                            rows['recommended_action'],
                        ], axis=-1),
                        hovertemplate=(
                            '<b>%{x}</b><br>Spend: ₹%{y:,.0f}<br>'
                            'Type: %{customdata[0]}<br>'
                            'Impact: ₹%{customdata[1]:,.0f}<br>'
                            'Action: %{customdata[2]}<extra></extra>'
                        ),
                    ))

            fig.update_layout(
                **_dark_layout(height=340, hovermode='closest'),
                xaxis=dict(title='Date', tickangle=30,
                           gridcolor='rgba(255,255,255,0.04)',
                           tickfont=dict(color='#475569', size=11)),
                yaxis=dict(title='Spend (₹)', tickprefix='₹',
                           gridcolor='rgba(255,255,255,0.04)',
                           tickfont=dict(color='#475569', size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Severity donut
            st.markdown(_section("Severity Distribution"), unsafe_allow_html=True)
            sev_counts = (flagged['severity']
                          .value_counts()
                          .reindex(['Critical','High','Medium','Low'])
                          .fillna(0))
            fig3 = go.Figure(go.Pie(
                labels=sev_counts.index,
                values=sev_counts.values,
                hole=0.62,
                marker=dict(
                    colors=[SEVERITY_COLORS[s] for s in sev_counts.index],
                    line=dict(color='#070B13', width=2),
                ),
                textinfo='label+value',
                textfont=dict(size=11, color='#94A3B8'),
                hovertemplate='%{label}: %{value} alerts<extra></extra>',
            ))
            fig3.update_layout(
                **_dark_layout(height=220, hovermode='closest'),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Financial impact heatmap ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(_section("Financial Impact Heatmap — Channel × Month"), unsafe_allow_html=True)

    if len(flagged) > 0:
        flagged['month_str'] = pd.to_datetime(flagged['date']).dt.to_period('M').astype(str)
        impact_pivot = flagged.pivot_table(
            values='impact_inr', index='channel',
            columns='month_str', aggfunc='sum', fill_value=0,
        ) / 100_000

        fig4 = px.imshow(
            impact_pivot.round(1),
            color_continuous_scale='OrRd',
            aspect='auto',
            labels=dict(x='Month', y='Channel', color='Impact (₹L)'),
            text_auto='.1f',
        )
        fig4.update_layout(
            **_dark_layout(height=260, hovermode='closest'),
            xaxis=dict(tickangle=45, tickfont=dict(size=9, color='#475569')),
            yaxis=dict(tickfont=dict(size=11, color='#64748B')),
            coloraxis_colorbar=dict(
                tickfont=dict(color='#64748B', size=10),
                title=dict(text='₹L', font=dict(color='#475569')),
            ),
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No flagged anomalies to display in the heatmap.")

    # ── Full alert table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(_section("Full Anomaly Report — Download Ready"), unsafe_allow_html=True)

    if len(flagged) > 0:
        export_cols = ['date','channel','severity','anomaly_label',
                       'spend','roas','cpc','impact_inr','recommended_action']
        export_df = flagged[export_cols].copy()
        export_df['date']       = pd.to_datetime(export_df['date']).dt.strftime('%Y-%m-%d')
        export_df['spend']      = export_df['spend'].round(0)
        export_df['roas']       = export_df['roas'].round(2)
        export_df['cpc']        = export_df['cpc'].round(0)
        export_df['impact_inr'] = export_df['impact_inr'].round(0)
        export_df.columns = ['Date','Channel','Severity','Anomaly Type',
                              'Spend (₹)','ROAS','CPC (₹)',
                              'Impact (₹)','Recommended Action']
        st.dataframe(export_df, use_container_width=True, hide_index=True)
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇  Download alert report as CSV",
            data=csv_bytes, file_name="anomaly_alert_report.csv", mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — BUDGET PACING
# ─────────────────────────────────────────────────────────────────────────────

def page_budget_pacing(df):
    st.markdown(_page_header(
        "Spend vs Target",
        "Budget Pacing",
        "Monthly spend vs target — identify over and underspend early "
        "so you can correct course before the month closes."
    ), unsafe_allow_html=True)

    # ── Build pacing table ────────────────────────────────────────────────────
    monthly_actual = (df.groupby('month')['spend'].sum()
                        .reset_index().sort_values('month'))
    monthly_actual.columns = ['month', 'actual_spend']

    festive  = ['2023-10','2023-11','2023-12','2024-10','2024-11']
    slow     = ['2023-01','2023-02','2024-01','2024-02']
    base_bud = monthly_actual['actual_spend'].median() * 0.98

    monthly_actual['budget_target'] = monthly_actual['month'].apply(
        lambda m: base_bud * 1.55 if m in festive
                  else base_bud * 0.80 if m in slow
                  else base_bud
    )
    monthly_actual['pacing_pct'] = (
        monthly_actual['actual_spend'] / monthly_actual['budget_target'] * 100
    ).round(1)
    monthly_actual['over_under'] = (
        monthly_actual['actual_spend'] - monthly_actual['budget_target']
    ).round(0)
    monthly_actual['status'] = monthly_actual['pacing_pct'].apply(
        lambda x: 'Over budget'   if x > 107
        else      ('Under budget' if x < 90 else 'On track')
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    over_months  = (monthly_actual['status'] == 'Over budget').sum()
    under_months = (monthly_actual['status'] == 'Under budget').sum()
    on_track     = (monthly_actual['status'] == 'On track').sum()
    total_over   =  monthly_actual[monthly_actual['over_under'] > 0]['over_under'].sum()
    total_under  = -monthly_actual[monthly_actual['over_under'] < 0]['over_under'].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_kpi("On-Track Months",    str(on_track),                  glow='#34D399'), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi("Over Budget",        str(over_months),               glow='#EF4444'), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi("Under Budget",       str(under_months),              glow='#FBBF24'), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("Total Overspend",    f"₹{total_over/100_000:.1f}L",  glow='#F87171'), unsafe_allow_html=True)
    with c5:
        st.markdown(_kpi("Total Underspend",   f"₹{total_under/100_000:.1f}L", glow='#60A5FA'), unsafe_allow_html=True)

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Spend vs target chart ─────────────────────────────────────────────────
    st.markdown(_section("Monthly Actual Spend vs Budget Target"), unsafe_allow_html=True)

    STATUS_CLR = {'Over budget': '#F87171', 'Under budget': '#34D399', 'On track': '#60A5FA'}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_actual['month'], y=monthly_actual['budget_target']/100_000,
        name='Budget target',
        marker_color='rgba(255,255,255,0.07)',
        marker_line_color='rgba(255,255,255,0.12)', marker_line_width=1,
        hovertemplate='%{x}<br>Target: ₹%{y:.2f}L<extra></extra>',
    ))
    for status, color in STATUS_CLR.items():
        rows = monthly_actual[monthly_actual['status'] == status]
        if len(rows) > 0:
            fig.add_trace(go.Bar(
                x=rows['month'], y=rows['actual_spend']/100_000,
                name=f'Actual — {status}',
                marker_color=_hex_to_rgba(color, 0.7),
                marker_line_color=color, marker_line_width=0.5,
                hovertemplate=f'%{{x}}<br>Actual: ₹%{{y:.2f}}L<br>{status}<extra></extra>',
            ))
    fig.update_layout(
        **_dark_layout(height=360),
        barmode='overlay',
        xaxis=dict(title='Month', tickangle=45, tickfont=dict(color='#475569', size=10)),
        yaxis=dict(title='Spend (₹ Lakhs)', tickprefix='₹', ticksuffix='L',
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Pacing % bar ──────────────────────────────────────────────────────────
    st.markdown(_section("Pacing % by Month"), unsafe_allow_html=True)
    st.caption("100% = exactly on target.  Green = underspend.  Red = overspend.")

    pacing_bar_colors = [
        '#F87171' if p > 107 else '#34D399' if p < 90 else '#60A5FA'
        for p in monthly_actual['pacing_pct']
    ]
    fig2 = go.Figure(go.Bar(
        x=monthly_actual['month'], y=monthly_actual['pacing_pct'],
        marker_color=[_hex_to_rgba(c, 0.75) for c in pacing_bar_colors],
        marker_line_color=pacing_bar_colors, marker_line_width=0.8,
        opacity=0.9,
        text=monthly_actual['pacing_pct'].apply(lambda x: f'{x:.0f}%'),
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11),
        hovertemplate='%{x}<br>Pacing: %{y:.1f}%<extra></extra>',
    ))
    fig2.add_hline(y=100, line_dash='dash', line_color='rgba(255,255,255,0.2)', line_width=1.5,
                   annotation_text='100% target',
                   annotation_font=dict(color='#64748B', size=10),
                   annotation_position='top right')
    fig2.add_hline(y=107, line_dash='dot', line_color='rgba(248,113,113,0.4)', line_width=1,
                   annotation_text='Over-budget',
                   annotation_font=dict(color='#F87171', size=10),
                   annotation_position='top right')
    fig2.add_hline(y=90, line_dash='dot', line_color='rgba(52,211,153,0.4)', line_width=1,
                   annotation_text='Under-budget',
                   annotation_font=dict(color='#34D399', size=10),
                   annotation_position='bottom right')
    fig2.update_layout(
        **_dark_layout(height=320, margin=dict(l=4, r=80, t=30, b=4)),
        yaxis=dict(title='Pacing %', range=[50, 155], ticksuffix='%',
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
        xaxis=dict(tickangle=45, tickfont=dict(size=10, color='#475569')),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Over/Under waterfall ──────────────────────────────────────────────────
    st.markdown(_section("Monthly Over / Underspend vs Target"), unsafe_allow_html=True)
    over_under_colors = [
        _hex_to_rgba('#F87171', 0.7) if v > 0 else _hex_to_rgba('#34D399', 0.7)
        for v in monthly_actual['over_under']
    ]
    fig3 = go.Figure(go.Bar(
        x=monthly_actual['month'], y=monthly_actual['over_under']/100_000,
        marker_color=over_under_colors,
        marker_line_color=['#F87171' if v > 0 else '#34D399' for v in monthly_actual['over_under']],
        marker_line_width=0.6,
        opacity=0.9,
        text=monthly_actual['over_under'].apply(lambda x: f'₹{x/100_000:+.2f}L'),
        textposition='outside',
        textfont=dict(color='#94A3B8', size=10),
        hovertemplate='%{x}<br>Over/Under: ₹%{y:+.2f}L<extra></extra>',
    ))
    fig3.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1.5)
    fig3.update_layout(
        **_dark_layout(height=300, margin=dict(l=4, r=4, t=30, b=4)),
        yaxis=dict(title='Over/Under spend (₹ Lakhs)', tickprefix='₹', ticksuffix='L',
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
        xaxis=dict(tickangle=45, tickfont=dict(size=10, color='#475569')),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Channel-level pacing drill-down ───────────────────────────────────────
    st.markdown("---")
    st.markdown(_section("Channel-Level Pacing Breakdown"), unsafe_allow_html=True)

    selected_month = st.selectbox(
        "Select a month to drill down",
        sorted(monthly_actual['month'].unique(), reverse=True),
    )
    month_df  = df[df['month'] == selected_month]
    ch_actual = month_df.groupby('channel')['spend'].sum().reset_index()
    ch_actual.columns = ['channel', 'actual']

    channel_budgets = {
        'Google Search': 35, 'Meta Ads': 27,
        'Programmatic Display': 16, 'YouTube': 13, 'Affiliate': 9,
    }
    total_budget_month = monthly_actual.loc[
        monthly_actual['month'] == selected_month, 'budget_target'
    ].values[0]

    ch_actual['target'] = ch_actual['channel'].map(
        {ch: total_budget_month * pct / 100 for ch, pct in channel_budgets.items()}
    )
    ch_actual['pacing'] = (ch_actual['actual'] / ch_actual['target'] * 100).round(1)
    ch_actual['status'] = ch_actual['pacing'].apply(
        lambda x: 'Over' if x > 107 else ('Under' if x < 90 else 'On track')
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name='Budget target',
        x=ch_actual['channel'], y=ch_actual['target']/100_000,
        marker_color='rgba(255,255,255,0.07)',
        marker_line_color='rgba(255,255,255,0.12)', marker_line_width=1,
    ))
    fig4.add_trace(go.Bar(
        name='Actual spend',
        x=ch_actual['channel'], y=ch_actual['actual']/100_000,
        marker_color=[_hex_to_rgba(CHANNEL_COLORS[ch], 0.7) for ch in ch_actual['channel']],
        marker_line_color=[CHANNEL_COLORS[ch] for ch in ch_actual['channel']],
        marker_line_width=0.8,
        text=ch_actual['pacing'].apply(lambda x: f'{x:.0f}%'),
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11),
    ))
    fig4.update_layout(
        **_dark_layout(height=320),
        barmode='group',
        title=dict(text=f'Channel pacing — {selected_month}',
                   font=dict(color='#64748B', size=13, family='DM Mono')),
        yaxis=dict(title='Spend (₹ Lakhs)', tickprefix='₹', ticksuffix='L',
                   gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569', size=11)),
        xaxis=dict(tickangle=0, tickfont=dict(color='#64748B', size=12)),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Pacing table ──────────────────────────────────────────────────────────
    st.markdown(_section("Full Pacing Table"), unsafe_allow_html=True)
    pacing_display = monthly_actual[
        ['month','actual_spend','budget_target','pacing_pct','over_under','status']
    ].copy()
    pacing_display['actual_spend']  = pacing_display['actual_spend'].apply(lambda x: f"₹{x:,.0f}")
    pacing_display['budget_target'] = pacing_display['budget_target'].apply(lambda x: f"₹{x:,.0f}")
    pacing_display['pacing_pct']    = pacing_display['pacing_pct'].apply(lambda x: f"{x:.1f}%")
    pacing_display['over_under']    = pacing_display['over_under'].apply(lambda x: f"₹{x:+,.0f}")
    pacing_display.columns = ['Month','Actual Spend','Budget Target',
                               'Pacing %','Over / Under','Status']
    st.dataframe(pacing_display, use_container_width=True, hide_index=True)

    csv_p = pacing_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇  Download pacing table as CSV",
        data=csv_p, file_name="budget_pacing.csv", mime="text/csv",
    )

    st.markdown("""
    <div class="insight-box">
        <b>Key insight:</b> Consistent overspend in festive months combined with
        underspend in Jan–Feb suggests media plan targets are not adjusted to match
        actual market demand curves. Recommendation: set dynamic monthly targets
        that flex with seasonality rather than flat monthly caps.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_campaign_data()
    page, start_date, end_date, selected_channels = render_sidebar(df)

    mask = (
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date) &
        (df['channel'].isin(selected_channels))
    )
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        st.warning("No data matches the selected filters. Adjust the date range or channel selection.")
        return

    if page == "Executive Overview":
        page_executive_overview(df_filtered)
    elif page == "Attribution Analysis":
        page_attribution_analysis()
    elif page == "Anomaly Monitor":
        page_anomaly_monitor(df_filtered)
    elif page == "Budget Pacing":
        page_budget_pacing(df_filtered)


if __name__ == '__main__':
    main()