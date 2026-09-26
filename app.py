"""
MineWatch V2 - Interactive Mining Production, Safety & Environmental Analytics
================================================================================
A Streamlit EDA dashboard built on a synthetic, project-generated mining
dataset. This is an ACADEMIC PROTOTYPE, not an official government
compliance / safety system. See the "About Project" page for full
positioning and limitations.

Run:
    python -m streamlit run app.py
"""

# ==================================================
# 1. IMPORTS
# ==================================================
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ==================================================
# 2. CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="MineWatch V2 | Mining Analytics",
    page_icon="\u26CF\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "minewatch_data.csv"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

VIOLATION_CATEGORIES_CANON = ["Safety", "Environmental", "Operational",
                               "Equipment", "Documentation", "Emergency Preparedness"]
MINE_TYPE_CANON = ["Open Cast", "Underground", "Mixed"]

# --------------------------------------------------
# 2a. SEMANTIC COLOR SYSTEM
# --------------------------------------------------
# Brand / primary
PRIMARY = "#F2A93B"        # amber / gold - branding, primary actions, production highlights
# Operations (production, productivity, energy)
OPERATIONS = "#4EA1D8"     # steel blue
# Environment (waste, environmental indicators, sustainability)
ENVIRONMENT = "#45C2B1"    # teal / green
# Safety & alerts (incidents, violations, critical alerts, negative change)
SAFETY = "#E56B5D"         # red / coral
# Positive / completed
POSITIVE = "#62B58A"       # muted green
# Secondary analytics (forecasting, AI insights)
SECONDARY = "#9A86D8"      # purple
# Cost (economics)
COST = "#C98A5B"           # muted orange
# Neutral / reference
NEUTRAL = "#77838F"

# Backgrounds
BG = "#0B0F14"
SIDEBAR_BG = "#10161D"
CARD_BG = "#151C24"
CARD_BG_2 = "#1A222C"
BORDER = "#28323D"

# Text
TEXT = "#F2F4F5"
TEXT_SECONDARY = "#A8B2BD"
TEXT_MUTED = "#77838F"

# Backwards-compatible aliases used deeper in the file
ACCENT = PRIMARY
ACCENT_2 = OPERATIONS
PANEL = CARD_BG
PANEL_BORDER = BORDER
MUTED = TEXT_SECONDARY
GOOD = POSITIVE
WARN = SAFETY

PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORWAY = [PRIMARY, OPERATIONS, ENVIRONMENT, SAFETY, POSITIVE, SECONDARY, COST, NEUTRAL]

# Reusable semantic Plotly color map — same metric, same color, everywhere.
PLOT_COLORS = {
    "production": PRIMARY,
    "productivity": OPERATIONS,
    "energy": OPERATIONS,
    "environment": ENVIRONMENT,
    "waste": ENVIRONMENT,
    "safety": SAFETY,
    "violations": SAFETY,
    "completed": POSITIVE,
    "pending": PRIMARY,
    "revenue": OPERATIONS,
    "cost": COST,
    "profit": POSITIVE,
    "forecast": SECONDARY,
    "historical": OPERATIONS,
    "reserve_initial": NEUTRAL,
    "reserve_remaining": OPERATIONS,
    "neutral": NEUTRAL,
}

# Risk palette: low -> muted green, medium -> amber, high -> coral (restrained, not neon)
RISK_COLORSCALE = [
    [0.0, "#2E6B54"],
    [0.35, POSITIVE],
    [0.6, PRIMARY],
    [1.0, SAFETY],
]

# ==================================================
# 3. CUSTOM CSS
# ==================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"]  {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

.block-container {{
    max-width: 1400px;
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
}}

section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem;
}}

h1, h2, h3, h4 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    color: {TEXT};
}}

/* ---------- Page header ---------- */
.mw-eyebrow {{
    color: {PRIMARY};
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}}
.mw-page-title {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.15rem;
    line-height: 1.2;
}}
.mw-page-sub {{
    color: {TEXT_SECONDARY};
    font-size: 0.92rem;
    margin-bottom: 1.1rem;
}}
.mw-header-block {{
    margin-bottom: 1.3rem;
}}

/* ---------- KPI cards ---------- */
.mw-kpi-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 3px solid var(--mw-accent, {PRIMARY});
    border-radius: 11px;
    padding: 14px 18px;
    height: 100%;
    min-height: 104px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.mw-kpi-card:hover {{
    border-color: var(--mw-accent, {PRIMARY});
    transform: translateY(-1px);
}}
.mw-kpi-label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.mw-kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: {TEXT};
    line-height: 1.15;
}}
.mw-kpi-sub {{
    font-size: 0.76rem;
    color: {TEXT_SECONDARY};
}}
.mw-kpi-change {{
    font-size: 0.76rem;
    font-weight: 600;
}}
.mw-change-up-good {{ color: {POSITIVE}; }}
.mw-change-up-bad {{ color: {SAFETY}; }}
.mw-change-down-good {{ color: {POSITIVE}; }}
.mw-change-down-bad {{ color: {SAFETY}; }}
.mw-change-neutral {{ color: {TEXT_MUTED}; }}

/* ---------- Generic panel ---------- */
.mw-panel {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 11px;
    padding: 16px 20px;
    margin-bottom: 14px;
}}
.mw-panel-2 {{
    background-color: {CARD_BG_2};
    border: 1px solid {BORDER};
    border-radius: 11px;
    padding: 16px 20px;
    margin-bottom: 14px;
}}

/* ---------- Findings ---------- */
.mw-finding-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 3px solid var(--mw-accent, {PRIMARY});
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
}}
.mw-finding-cat {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--mw-accent, {PRIMARY});
    margin-bottom: 3px;
}}
.mw-finding-text {{
    font-size: 0.9rem;
    color: {TEXT};
}}

/* ---------- What changed ---------- */
.mw-wc-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 4px;
    border-bottom: 1px solid {BORDER};
}}
.mw-wc-row:last-child {{ border-bottom: none; }}
.mw-wc-label {{ color: {TEXT_SECONDARY}; font-size: 0.86rem; }}
.mw-wc-value {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.9rem; }}

/* ---------- Badges ---------- */
.mw-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    background-color: rgba(154, 134, 216, 0.14);
    color: {SECONDARY};
    border: 1px solid rgba(154, 134, 216, 0.35);
}}
.mw-badge-blue {{
    background-color: rgba(78, 161, 216, 0.14);
    color: {OPERATIONS};
    border: 1px solid rgba(78, 161, 216, 0.35);
}}

/* ---------- Sidebar nav ---------- */
.mw-sidebar-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.02em;
    margin-bottom: 0;
}}
.mw-sidebar-caption {{
    color: {TEXT_MUTED};
    font-size: 0.75rem;
    margin-top: -4px;
    margin-bottom: 0.6rem;
}}
.mw-nav-section {{
    color: {TEXT_MUTED};
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.75rem 0 0.15rem 0;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 2px 0;
}}
section[data-testid="stSidebar"] input[type="radio"] {{
    accent-color: {PRIMARY};
}}

/* ---------- Info boxes ---------- */
.mw-info-box {{
    background-color: rgba(154, 134, 216, 0.08);
    border: 1px solid rgba(154, 134, 216, 0.30);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: {TEXT_SECONDARY};
    margin-top: 8px;
}}
.mw-disclaimer {{
    font-size: 0.76rem;
    color: {TEXT_MUTED};
    font-style: italic;
    margin-top: 6px;
}}

/* ---------- Empty state ---------- */
.mw-empty-state {{
    background-color: {CARD_BG};
    border: 1px dashed {BORDER};
    border-radius: 12px;
    padding: 40px 20px;
    text-align: center;
    color: {TEXT_SECONDARY};
    margin: 12px 0;
}}
.mw-empty-icon {{
    font-size: 1.8rem;
    margin-bottom: 8px;
    opacity: 0.6;
}}
.mw-empty-title {{
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    margin-bottom: 6px;
}}

/* ---------- Mine profile ---------- */
.mw-profile-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px 24px;
    margin-top: 8px;
}}
.mw-profile-item-label {{
    font-size: 0.68rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.mw-profile-item-value {{
    font-size: 0.95rem;
    color: {TEXT};
    font-weight: 600;
}}

hr {{ border-color: {BORDER}; }}

[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
}}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. DATA LOADING  (unchanged logic)
# ==================================================
@st.cache_data(show_spinner="Loading MineWatch dataset...")
def load_raw_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner="Cleaning dataset...")
def clean_data(raw: pd.DataFrame):
    df = raw.copy()
    stats = {}
    stats["raw_rows"] = len(df)

    # ---- duplicate removal ----
    stats["duplicates_found"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    stats["rows_after_dedup"] = len(df)

    # ---- category normalization ----
    def norm_mine_type(v):
        if pd.isna(v):
            return v
        v2 = str(v).strip().lower().replace("-", " ")
        mapping = {"open cast": "Open Cast", "underground": "Underground", "mixed": "Mixed"}
        return mapping.get(v2, str(v).strip())

    def norm_title(v):
        if pd.isna(v):
            return v
        return str(v).strip().title()

    df["Mine_Type"] = df["Mine_Type"].apply(norm_mine_type)
    df["Violation_Category"] = df["Violation_Category"].apply(norm_title)
    df["Incident_Category"] = df["Incident_Category"].apply(
        lambda v: str(v).strip() if pd.notna(v) else v
    )

    # ---- date handling ----
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    # ---- missing value treatment ----
    stats["missing_before"] = int(df[["Working_Hours", "Energy_Consumption",
                                       "Commodity_Price", "Environmental_Impact"]].isna().sum().sum())
    for col in ["Working_Hours", "Energy_Consumption", "Commodity_Price", "Environmental_Impact"]:
        med = df.groupby("Mine_ID")[col].transform("median")
        df[col] = df[col].fillna(med)
        df[col] = df[col].fillna(df[col].median())
    stats["missing_after"] = int(df[["Working_Hours", "Energy_Consumption",
                                      "Commodity_Price", "Environmental_Impact"]].isna().sum().sum())

    # ---- derived metrics ----
    df["Energy_Intensity"] = (df["Energy_Consumption"] / df["Production"]).replace([np.inf, -np.inf], np.nan)
    df["Waste_Intensity"] = (df["Waste_Generation"] / df["Production"]).replace([np.inf, -np.inf], np.nan)
    df["Incident_Rate_per_1000h"] = (df["Safety_Incidents"] / df["Working_Hours"] * 1000).replace(
        [np.inf, -np.inf], np.nan)

    return df, stats


raw_df = load_raw_data(DATA_FILE)

if raw_df is None:
    st.error(
        f"Dataset file `{DATA_FILE}` was not found. Please run `python generate_dataset.py` "
        "first to create the dataset, then restart the app."
    )
    st.stop()

df, clean_stats = clean_data(raw_df)

MIN_DATE = df["Date"].min()
MAX_DATE = df["Date"].max()


# ==================================================
# 5. HELPER FUNCTIONS
# ==================================================
def pct_change(new, old):
    if old is None or old == 0 or pd.isna(old) or pd.isna(new):
        return None
    return (new - old) / abs(old) * 100.0


def page_header(eyebrow, title, sub=""):
    """Consistent SMALL LABEL / big title / one-line description header."""
    st.markdown(f"""
    <div class="mw-header-block">
        <div class="mw-eyebrow">{eyebrow}</div>
        <div class="mw-page-title">{title}</div>
        {f'<div class="mw-page-sub">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def _change_class(change_pct, higher_is_good=True):
    if change_pct is None:
        return "mw-change-neutral", ""
    up = change_pct >= 0
    good = (up and higher_is_good) or (not up and not higher_is_good)
    arrow = "\u2191" if up else "\u2193"
    cls = "mw-change-up-good" if (up and good) else \
          "mw-change-up-bad" if (up and not good) else \
          "mw-change-down-good" if (not up and good) else "mw-change-down-bad"
    return cls, f"{arrow} {abs(change_pct):.1f}%"


def kpi_card(label, value, sub="", accent=None, change_pct=None, higher_is_good=True, change_suffix=""):
    """Semantic KPI card: uppercase label, big value, subtitle, optional colored change indicator,
    subtle colored left border matching the metric's semantic accent."""
    accent = accent or PRIMARY
    change_html = ""
    if change_pct is not None:
        cls, txt = _change_class(change_pct, higher_is_good)
        if txt:
            change_html = f'<div class="mw-kpi-change {cls}">{txt}{change_suffix}</div>'
    st.markdown(f"""
    <div class="mw-kpi-card" style="--mw-accent: {accent};">
        <div class="mw-kpi-label">{label}</div>
        <div class="mw-kpi-value">{value}</div>
        {f'<div class="mw-kpi-sub">{sub}</div>' if sub else ''}
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def style_fig(fig, title=None, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="IBM Plex Sans", size=12),
        colorway=CHART_COLORWAY,
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=CARD_BG_2, font=dict(color=TEXT, size=12), bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, showline=False)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, showline=False)
    return fig


def empty_state():
    st.markdown("""
    <div class="mw-empty-state">
        <div class="mw-empty-icon">&#9723;</div>
        <div class="mw-empty-title">No data for current filters</div>
        The selected filters do not return any records.<br>
        Try removing one or more filters from the sidebar.
    </div>
    """, unsafe_allow_html=True)


def safe_agg(frame, col, how="sum"):
    if frame.empty:
        return None
    if how == "sum":
        return frame[col].sum()
    if how == "mean":
        return frame[col].mean()
    if how == "median":
        return frame[col].median()
    return None


def format_num(x, decimals=0, suffix=""):
    if x is None or pd.isna(x):
        return "-"
    if abs(x) >= 1_00_00_000:  # 1 crore
        return f"{x/1_00_00_000:.2f} Cr{suffix}"
    if abs(x) >= 1_00_000:
        return f"{x/1_00_000:.2f} L{suffix}"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K{suffix}"
    return f"{x:,.{decimals}f}{suffix}"


def finding_card(category, text, accent=None):
    accent = accent or PRIMARY
    st.markdown(f"""
    <div class="mw-finding-card" style="--mw-accent: {accent};">
        <div class="mw-finding-cat">{category}</div>
        <div class="mw-finding-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def classify_finding(text):
    """Pick a semantic category + accent for a rule-based finding sentence,
    purely for display — does not alter the underlying analytical text."""
    t = text.lower()
    if "safety incident" in t:
        return "SAFETY", SAFETY
    if "violation" in t:
        return "VIOLATIONS", SAFETY
    if "pending corrective" in t:
        return "CORRECTIVE ACTIONS", PRIMARY
    if "environmental" in t:
        return "ENVIRONMENT", ENVIRONMENT
    if "waste" in t:
        return "ENVIRONMENT", ENVIRONMENT
    if "energy" in t:
        return "ENERGY", OPERATIONS
    if "profit" in t:
        return "ECONOMICS", POSITIVE
    if "productivity" in t:
        return "PRODUCTIVITY", OPERATIONS
    if "production" in t:
        return "PRODUCTION", PRIMARY
    return "OBSERVATION", NEUTRAL


def what_changed_panel(wc_df, prev_y, last_y):
    """Compact WHAT CHANGED? comparison panel with metric-aware coloring."""
    # metric -> higher_is_good
    directionality = {
        "Production": True,
        "Productivity (avg)": True,
        "Safety Incidents": False,
        "Violations": False,
        "Profit": True,
        "Energy Intensity": False,
        "Waste Intensity": False,
    }
    rows_html = []
    for _, row in wc_df.iterrows():
        metric = row["Metric"]
        change = row["Change %"]
        higher_good = directionality.get(metric, True)
        cls, txt = _change_class(change, higher_good)
        if not txt:
            txt = "-"
            cls = "mw-change-neutral"
        rows_html.append(
            f'<div class="mw-wc-row"><span class="mw-wc-label">{metric}</span>'
            f'<span class="mw-wc-value {cls}">{txt}</span></div>'
        )
    st.markdown(f"""
    <div class="mw-panel">
        <div class="mw-eyebrow" style="margin-bottom:6px;">WHAT CHANGED? &middot; {prev_y} vs {last_y}</div>
        {''.join(rows_html)}
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 6. SIDEBAR - FILTERS & NAVIGATION
# ==================================================
st.sidebar.markdown('<div class="mw-sidebar-title">\u26CF\uFE0F MINEWATCH</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="mw-sidebar-caption">Mining Analytics Platform</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

NAV_OPTIONS = [
    "Overview",
    "— Operations —",
    "Production",
    "Reserves",
    "Productivity",
    "— Efficiency —",
    "Energy",
    "Waste",
    "— Safety & Risk —",
    "Safety Trends",
    "Violations",
    "Risk Heatmap",
    "— Analysis —",
    "Environment",
    "Economics",
    "Regional Performance",
    "Forecasting",
    "Mine Explorer",
    "— Tools —",
    "AI Insights",
    "Data Explorer",
    "About Project",
]
SELECTABLE = [o for o in NAV_OPTIONS if not o.startswith("—")]


def _format_nav(x):
    if x.startswith("—"):
        return x.strip("— ").upper()
    return f"  {x}"


st.sidebar.markdown('<div class="mw-nav-section">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate",
    options=NAV_OPTIONS,
    format_func=_format_nav,
    label_visibility="collapsed",
)
if page.startswith("—"):
    # a section header was somehow selected (shouldn't normally happen); default to Overview
    page = "Overview"

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="mw-nav-section">Global Filters</div>', unsafe_allow_html=True)

if st.sidebar.button("\u21BB Reset Filters", use_container_width=True):
    for k in list(st.session_state.keys()):
        if k.startswith("f_"):
            del st.session_state[k]
    st.rerun()

f_mine = st.sidebar.multiselect("Mine", sorted(df["Mine_ID"].unique()), key="f_mine")
f_region = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), key="f_region")
f_state = st.sidebar.multiselect("State", sorted(df["State"].unique()), key="f_state")
f_type = st.sidebar.multiselect("Mine Type", MINE_TYPE_CANON, key="f_type")
f_commodity = st.sidebar.multiselect("Commodity", sorted(df["Commodity"].unique()), key="f_commodity")
f_year = st.sidebar.multiselect("Year", sorted(df["Year"].unique()), key="f_year")
f_month = st.sidebar.multiselect("Month", list(range(1, 13)), key="f_month")
f_violation_cat = st.sidebar.multiselect(
    "Violation Category", VIOLATION_CATEGORIES_CANON, key="f_violation_cat")
f_incident_cat = st.sidebar.multiselect(
    "Incident Category", sorted(df["Incident_Category"].dropna().unique()), key="f_incident_cat")
f_severity = st.sidebar.multiselect(
    "Severity", ["Minor", "Moderate", "Serious", "Fatal"], key="f_severity")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Academic EDA prototype using synthetic data. Not an official government "
    "compliance or safety system. See *About Project* for details."
)


def apply_filters(frame):
    f = frame.copy()
    if f_mine:
        f = f[f["Mine_ID"].isin(f_mine)]
    if f_region:
        f = f[f["Region"].isin(f_region)]
    if f_state:
        f = f[f["State"].isin(f_state)]
    if f_type:
        f = f[f["Mine_Type"].isin(f_type)]
    if f_commodity:
        f = f[f["Commodity"].isin(f_commodity)]
    if f_year:
        f = f[f["Year"].isin(f_year)]
    if f_month:
        f = f[f["Month"].isin(f_month)]
    if f_violation_cat:
        f = f[f["Violation_Category"].isin(f_violation_cat)]
    if f_incident_cat:
        f = f[f["Incident_Category"].isin(f_incident_cat)]
    if f_severity:
        f = f[f["Severity"].isin(f_severity)]
    return f


fdf = apply_filters(df)


# ==================================================
# 7. FINDINGS ENGINE  (unchanged logic)
# ==================================================
def generate_findings(frame):
    """Rule-based findings computed purely from the (filtered) data."""
    findings = []
    if frame.empty or frame["Year"].nunique() < 2:
        return findings

    yearly = frame.groupby("Year").agg(
        Production=("Production", "sum"),
        Productivity=("Productivity_Index", "mean"),
        Safety_Incidents=("Safety_Incidents", "sum"),
        Violations=("Violations", "sum"),
        Pending_Actions=("Pending_Actions", "sum"),
        Environmental_Impact=("Environmental_Impact", "mean"),
        Energy_Intensity=("Energy_Intensity", "mean"),
        Waste_Intensity=("Waste_Intensity", "mean"),
        Profit=("Profit", "sum"),
    ).sort_index()

    if len(yearly) < 2:
        return findings

    first_year, last_year = yearly.index.min(), yearly.index.max()
    first, last = yearly.loc[first_year], yearly.loc[last_year]

    def add(metric_label, first_v, last_v, unit="%", lower_is_notable=False):
        change = pct_change(last_v, first_v)
        if change is None:
            return
        direction = "increased" if change >= 0 else "decreased"
        findings.append(
            f"{metric_label} {direction} by {abs(change):.1f}% between {first_year} and {last_year}."
        )

    add("Production", first["Production"], last["Production"])
    add("Average productivity", first["Productivity"], last["Productivity"])
    add("Safety incidents", first["Safety_Incidents"], last["Safety_Incidents"])
    add("Violations", first["Violations"], last["Violations"])
    add("Pending corrective actions", first["Pending_Actions"], last["Pending_Actions"])
    add("Average environmental impact", first["Environmental_Impact"], last["Environmental_Impact"])
    add("Energy intensity", first["Energy_Intensity"], last["Energy_Intensity"])
    add("Waste intensity", first["Waste_Intensity"], last["Waste_Intensity"])
    add("Total profit", first["Profit"], last["Profit"])

    # Most frequent violation category
    if frame["Violation_Category"].notna().any():
        top_cat = frame["Violation_Category"].value_counts().idxmax()
        top_n = frame["Violation_Category"].value_counts().max()
        findings.append(f"'{top_cat}' is the most frequently recorded violation category ({int(top_n)} records).")

    return findings


def what_changed(frame, period_col_years):
    """Compare the most recent year in the filtered data vs the prior year."""
    years = sorted(frame["Year"].unique())
    if len(years) < 2:
        return None
    last_y, prev_y = years[-1], years[-2]
    cur = frame[frame["Year"] == last_y]
    prev = frame[frame["Year"] == prev_y]
    if cur.empty or prev.empty:
        return None

    metrics = {
        "Production": ("Production", "sum"),
        "Productivity (avg)": ("Productivity_Index", "mean"),
        "Safety Incidents": ("Safety_Incidents", "sum"),
        "Violations": ("Violations", "sum"),
        "Profit": ("Profit", "sum"),
        "Energy Intensity": ("Energy_Intensity", "mean"),
        "Waste Intensity": ("Waste_Intensity", "mean"),
    }
    out = []
    for label, (col, how) in metrics.items():
        cur_v = safe_agg(cur, col, how)
        prev_v = safe_agg(prev, col, how)
        change = pct_change(cur_v, prev_v)
        out.append({"Metric": label, f"{prev_y}": prev_v, f"{last_y}": cur_v, "Change %": change})
    return pd.DataFrame(out), prev_y, last_y


def _wc_lookup(wc_df, metric):
    v = wc_df.loc[wc_df.Metric == metric, "Change %"]
    return float(v.iloc[0]) if len(v) and v.iloc[0] is not None else None


# ==================================================
# 8. OVERVIEW
# ==================================================
if page == "Overview":
    page_header(
        "MineWatch V2",
        "Mining Production, Safety & Environmental Analytics",
        "Explore production, safety, environmental and economic trends across the portfolio. "
        "Academic EDA prototype - synthetic data - not an official compliance system."
    )

    if fdf.empty:
        empty_state()
    else:
        wc_res = what_changed(fdf, "Year")
        wc_df_ov = wc_res[0] if wc_res else None

        c1, c2, c3, c4 = st.columns(4)

        with c1:
             kpi_card(
                         "Total Mines",
            f"{fdf['Mine_ID'].nunique()}",
                "in current selection",
             accent=NEUTRAL
                )

        with c2:
            kpi_card(
                    "Total Production",
                 format_num(fdf["Production"].sum()),
             "tonnes",
                accent=PLOT_COLORS["production"],
             change_pct=_wc_lookup(wc_df_ov, "Production") if wc_df_ov is not None else None,
             higher_is_good=True,
                change_suffix=" vs prev. year"
                )

        with c3:
                    kpi_card(
                    "Total Safety Incidents",
                      f"{int(fdf['Safety_Incidents'].sum()):,}",
                  "",
                         accent=PLOT_COLORS["safety"],
                     change_pct=_wc_lookup(wc_df_ov, "Safety Incidents") if wc_df_ov is not None else None,
                     higher_is_good=False,
                    change_suffix=" vs prev. year"
                     )

        with c4:
                kpi_card(
                        "Total Violations",
                f"{int(fdf['Violations'].sum()):,}",
                "",
                accent=PLOT_COLORS["violations"],
                change_pct=_wc_lookup(wc_df_ov, "Violations") if wc_df_ov is not None else None,
                 higher_is_good=False,
                change_suffix=" vs prev. year"
            )
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            kpi_card("Total Revenue", format_num(fdf["Revenue"].sum()), "indexed units",
                      accent=PLOT_COLORS["revenue"])
        with c6:
            kpi_card("Total Profit", format_num(fdf["Profit"].sum()), "indexed units",
                      accent=PLOT_COLORS["profit"],
                      change_pct=_wc_lookup(wc_df_ov, "Profit") if wc_df_ov is not None else None,
                      higher_is_good=True, change_suffix=" vs prev. year")
        with c7:
            kpi_card("Avg Productivity", f"{fdf['Productivity_Index'].mean():.2f}", "tonnes/hour",
                      accent=PLOT_COLORS["productivity"])
        with c8:
            kpi_card("Pending Actions", f"{int(fdf['Pending_Actions'].sum()):,}", "",
                      accent=PLOT_COLORS["pending"])

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            trend = fdf.groupby("Date", as_index=False)["Production"].sum()
            fig = px.line(trend, x="Date", y="Production")
            fig.update_traces(line_color=PLOT_COLORS["production"])
            st.plotly_chart(style_fig(fig, "Production Trend"), use_container_width=True)
        with col2:
            trend2 = fdf.groupby("Date", as_index=False)["Safety_Incidents"].sum()
            fig = px.line(trend2, x="Date", y="Safety_Incidents")
            fig.update_traces(line_color=PLOT_COLORS["safety"])
            st.plotly_chart(style_fig(fig, "Safety Incident Trend"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.scatter(fdf, x="Production", y="Environmental_Impact",
                              color="Region", opacity=0.7)
            st.plotly_chart(style_fig(fig, "Production vs Environmental Impact"), use_container_width=True)
        with col4:
            econ = fdf.groupby("Date", as_index=False)[["Revenue", "Operating_Cost"]].sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=econ["Date"], y=econ["Revenue"], name="Revenue",
                                      line=dict(color=PLOT_COLORS["revenue"])))
            fig.add_trace(go.Scatter(x=econ["Date"], y=econ["Operating_Cost"], name="Operating Cost",
                                      line=dict(color=PLOT_COLORS["cost"])))
            st.plotly_chart(style_fig(fig, "Cost vs Revenue"), use_container_width=True)

        region_perf = fdf.groupby("Region", as_index=False).agg(
            Production=("Production", "sum"), Productivity=("Productivity_Index", "mean"))
        fig = px.bar(region_perf.sort_values("Production", ascending=False),
                     x="Region", y="Production", color_discrete_sequence=[PLOT_COLORS["production"]])
        st.plotly_chart(style_fig(fig, "Regional Performance"), use_container_width=True)

        st.markdown("### Key Findings")
        findings = generate_findings(fdf)
        if findings:
            for fnd in findings:
                cat, accent = classify_finding(fnd)
                finding_card(cat, fnd, accent=accent)
        else:
            st.caption("Select a broader range of years to generate trend-based findings.")

        if wc_res:
            st.markdown("<br>", unsafe_allow_html=True)
            what_changed_panel(*wc_res)


# ==================================================
# 9. OPERATIONS - PRODUCTION
# ==================================================
elif page == "Production":
    page_header("OPERATIONS", "Mining Production",
                "Explore production trends, mine output and regional share over time.")
    if fdf.empty:
        empty_state()
    else:
        agg_level = st.radio("Aggregation", ["Monthly", "Yearly"], horizontal=True)
        group_col = "Date" if agg_level == "Monthly" else "Year"
        trend = fdf.groupby(group_col, as_index=False)["Production"].sum()
        fig = px.line(trend, x=group_col, y="Production", markers=(agg_level == "Yearly"))
        fig.update_traces(line_color=PLOT_COLORS["production"])
        st.plotly_chart(style_fig(fig, f"{agg_level} Production Trend", height=460), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            by_mine = fdf.groupby("Mine_ID", as_index=False)["Production"].sum().sort_values(
                "Production", ascending=False).head(15)
            fig = px.bar(by_mine, x="Mine_ID", y="Production",
                         color_discrete_sequence=[PLOT_COLORS["production"]])
            st.plotly_chart(style_fig(fig, "Top 15 Mines by Production"), use_container_width=True)
        with col2:
            by_region = fdf.groupby("Region", as_index=False)["Production"].sum()
            fig = px.pie(by_region, names="Region", values="Production", hole=0.55)
            st.plotly_chart(style_fig(fig, "Production Share by Region"), use_container_width=True)


# ==================================================
# RESERVES
# ==================================================
elif page == "Reserves":
    page_header("OPERATIONS", "Reserve Depletion",
                "Analytical reserve-depletion model based on the project dataset - not an official reserve estimate.")
    if fdf.empty:
        empty_state()
    else:
        trend = fdf.groupby("Date", as_index=False)["Remaining_Reserve"].sum()
        fig = px.area(trend, x="Date", y="Remaining_Reserve",
                       color_discrete_sequence=[PLOT_COLORS["reserve_remaining"]])
        st.plotly_chart(style_fig(fig, "Remaining Reserve Over Time", height=460), use_container_width=True)

        latest_year = fdf["Year"].max()
        snap = fdf[fdf["Year"] == latest_year].groupby("Mine_ID", as_index=False).agg(
            Initial_Reserve=("Initial_Reserve", "first"),
            Remaining_Reserve=("Remaining_Reserve", "last"))
        snap = snap.sort_values("Remaining_Reserve", ascending=False).head(20)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=snap["Mine_ID"], y=snap["Initial_Reserve"], name="Initial Reserve",
                              marker_color=PLOT_COLORS["reserve_initial"]))
        fig.add_trace(go.Bar(x=snap["Mine_ID"], y=snap["Remaining_Reserve"], name="Remaining Reserve",
                              marker_color=PLOT_COLORS["reserve_remaining"]))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(style_fig(fig, f"Initial vs Remaining Reserve ({latest_year})"), use_container_width=True)


# ==================================================
# PRODUCTIVITY
# ==================================================
elif page == "Productivity":
    page_header("OPERATIONS", "Productivity Index",
                "Production relative to working-hour exposure, across mines and regions.")
    if fdf.empty:
        empty_state()
    else:
        trend = fdf.groupby("Date", as_index=False)["Productivity_Index"].mean()
        fig = px.line(trend, x="Date", y="Productivity_Index")
        fig.update_traces(line_color=PLOT_COLORS["productivity"])
        st.plotly_chart(style_fig(fig, "Productivity Trend", height=440), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            by_mine = fdf.groupby("Mine_ID", as_index=False)["Productivity_Index"].mean().sort_values(
                "Productivity_Index", ascending=False).head(15)
            fig = px.bar(by_mine, x="Mine_ID", y="Productivity_Index",
                         color_discrete_sequence=[PLOT_COLORS["productivity"]])
            st.plotly_chart(style_fig(fig, "Top 15 Mines by Productivity"), use_container_width=True)
        with col2:
            by_region = fdf.groupby("Region", as_index=False)["Productivity_Index"].mean()
            fig = px.bar(by_region.sort_values("Productivity_Index", ascending=False),
                         x="Region", y="Productivity_Index",
                         color_discrete_sequence=[PLOT_COLORS["productivity"]])
            st.plotly_chart(style_fig(fig, "Productivity by Region"), use_container_width=True)

        st.markdown("#### Commodity Price vs Production")
        st.caption("The chart shows association, not causation.")
        fig = px.scatter(fdf, x="Commodity_Price", y="Production", color="Commodity", opacity=0.7)
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True)


# ==================================================
# EFFICIENCY - ENERGY
# ==================================================
elif page == "Energy":
    page_header("EFFICIENCY", "Energy Consumption",
                "How energy demand changes with production and mine type.")
    if fdf.empty:
        empty_state()
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(fdf, x="Production", y="Energy_Consumption", color="Mine_Type", opacity=0.7)
            st.plotly_chart(style_fig(fig, "Production vs Energy Consumption"), use_container_width=True)
        with col2:
            trend = fdf.groupby("Date", as_index=False)["Energy_Intensity"].mean()
            fig = px.line(trend, x="Date", y="Energy_Intensity")
            fig.update_traces(line_color=PLOT_COLORS["energy"])
            st.plotly_chart(style_fig(fig, "Energy Intensity Trend"), use_container_width=True)

        by_type = fdf.groupby("Mine_Type", as_index=False)["Energy_Consumption"].sum()
        fig = px.bar(by_type, x="Mine_Type", y="Energy_Consumption",
                     color_discrete_sequence=[PLOT_COLORS["energy"]])
        st.plotly_chart(style_fig(fig, "Total Energy Consumption by Mine Type"), use_container_width=True)


# ==================================================
# EFFICIENCY - WASTE
# ==================================================
elif page == "Waste":
    page_header("EFFICIENCY", "Waste Generation",
                "Waste generation relative to production, across regions.")
    if fdf.empty:
        empty_state()
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(fdf, x="Production", y="Waste_Generation", color="Region", opacity=0.7)
            st.plotly_chart(style_fig(fig, "Production vs Waste Generation"), use_container_width=True)
        with col2:
            trend = fdf.groupby("Date", as_index=False)["Waste_Intensity"].mean()
            fig = px.line(trend, x="Date", y="Waste_Intensity")
            fig.update_traces(line_color=PLOT_COLORS["waste"])
            st.plotly_chart(style_fig(fig, "Waste Intensity Trend"), use_container_width=True)


# ==================================================
# SAFETY & RISK - SAFETY TRENDS
# ==================================================
elif page == "Safety Trends":
    page_header("SAFETY & RISK", "Safety Trends",
                "Historical safety analysis - not an accident prediction system.")
    if fdf.empty:
        empty_state()
    else:
        trend = fdf.groupby("Date", as_index=False)["Safety_Incidents"].sum()
        fig = px.line(trend, x="Date", y="Safety_Incidents")
        fig.update_traces(line_color=PLOT_COLORS["safety"])
        st.plotly_chart(style_fig(fig, "Safety Incidents Over Time", height=440), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            by_cat = fdf[fdf["Incident_Category"].notna()]["Incident_Category"].value_counts().reset_index()
            by_cat.columns = ["Incident_Category", "Count"]
            fig = px.bar(by_cat, x="Incident_Category", y="Count",
                         color_discrete_sequence=[PLOT_COLORS["safety"]])
            st.plotly_chart(style_fig(fig, "Incidents by Category"), use_container_width=True)
        with col2:
            by_sev = fdf[fdf["Severity"].notna()]["Severity"].value_counts().reset_index()
            by_sev.columns = ["Severity", "Count"]
            fig = px.pie(by_sev, names="Severity", values="Count", hole=0.55,
                         color_discrete_sequence=[POSITIVE, PRIMARY, "#C97A4A", SAFETY])
            st.plotly_chart(style_fig(fig, "Incidents by Severity"), use_container_width=True)

        by_mine = fdf.groupby("Mine_ID", as_index=False).agg(
            Safety_Incidents=("Safety_Incidents", "sum"),
            Working_Hours=("Working_Hours", "sum"))
        by_mine["Incident_Rate_per_1000h"] = (by_mine["Safety_Incidents"] / by_mine["Working_Hours"] * 1000)
        by_mine = by_mine.sort_values("Incident_Rate_per_1000h", ascending=False).head(15)
        fig = px.bar(by_mine, x="Mine_ID", y="Incident_Rate_per_1000h",
                     color_discrete_sequence=[PLOT_COLORS["safety"]])
        st.plotly_chart(style_fig(fig, "Incident Rate per 1,000 Working Hours (Top 15 Mines)"),
                         use_container_width=True)


# ==================================================
# SAFETY & RISK - VIOLATIONS
# ==================================================
elif page == "Violations":
    page_header("SAFETY & RISK", "Violations",
                "Violation categories, corrective actions and their resolution status.")
    if fdf.empty:
        empty_state()
    else:
        col1, col2 = st.columns(2)
        with col1:
            by_cat = fdf[fdf["Violation_Category"].notna()]["Violation_Category"].value_counts().reset_index()
            by_cat.columns = ["Violation_Category", "Count"]
            fig = px.bar(by_cat.sort_values("Count", ascending=False), x="Violation_Category", y="Count",
                         color_discrete_sequence=[PLOT_COLORS["violations"]])
            st.plotly_chart(style_fig(fig, "Violations by Category"), use_container_width=True)
        with col2:
            trend = fdf.groupby("Year", as_index=False)["Violations"].sum()
            fig = px.bar(trend, x="Year", y="Violations",
                         color_discrete_sequence=[PLOT_COLORS["violations"]])
            st.plotly_chart(style_fig(fig, "Violations by Year"), use_container_width=True)

        st.markdown("#### Violations vs Corrective Actions")
        va = fdf.groupby("Mine_ID", as_index=False).agg(
            Violations=("Violations", "sum"), Corrective_Actions=("Corrective_Actions", "sum"))
        fig = px.scatter(va, x="Violations", y="Corrective_Actions", hover_name="Mine_ID",
                          color_discrete_sequence=[PLOT_COLORS["completed"]])
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("#### Corrective Action Status")
        status_counts = fdf["Action_Status"].value_counts().reset_index()
        status_counts.columns = ["Action_Status", "Count"]
        fig = px.pie(status_counts, names="Action_Status", values="Count", hole=0.55,
                     color_discrete_sequence=[POSITIVE, PRIMARY, SAFETY, NEUTRAL])
        st.plotly_chart(style_fig(fig), use_container_width=True)


# ==================================================
# SAFETY & RISK - RISK HEATMAP
# ==================================================
elif page == "Risk Heatmap":
    page_header("SAFETY & RISK", "Year x Risk Factor Heatmap",
                "Where project-defined risk indicators run higher, by year, mine type and region.")
    st.caption(
        "Cell values are the average of the project-defined Risk_Factor "
        "(a 0-10 composite of safety incidents, violations and randomness) for each "
        "year / mine-type combination. This identifies periods where risk indicators "
        "run higher, and is not an official risk certification."
    )
    if fdf.empty:
        empty_state()
    else:
        pivot = fdf.pivot_table(index="Year", columns="Mine_Type", values="Risk_Factor", aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale=RISK_COLORSCALE, aspect="auto",
                         labels=dict(color="Avg Risk Factor"))
        st.plotly_chart(style_fig(fig, "Average Risk Factor by Year and Mine Type", height=460),
                         use_container_width=True)

        pivot2 = fdf.pivot_table(index="Year", columns="Region", values="Risk_Factor", aggfunc="mean")
        fig = px.imshow(pivot2, color_continuous_scale=RISK_COLORSCALE, aspect="auto",
                         labels=dict(color="Avg Risk Factor"))
        st.plotly_chart(style_fig(fig, "Average Risk Factor by Year and Region", height=460),
                         use_container_width=True)

        # ---- MineWatch project-defined monitoring indicator (display-only composite) ----
        st.markdown("#### MineWatch Project-Defined Monitoring Indicator")
        total_viol = fdf["Violations"].sum()
        total_safety = fdf["Safety_Incidents"].sum()
        total_pending = fdf["Pending_Actions"].sum()
        total_env_breach = fdf["Environmental_Breach"].sum() if "Environmental_Breach" in fdf.columns else 0
        parts = {"Violations": total_viol, "Safety": total_safety,
                 "Pending": total_pending, "Environment": total_env_breach}
        total_parts = sum(parts.values())
        if total_parts > 0:
            weights = {k: v / total_parts for k, v in parts.items()}
            # simple bounded 0-100 composite: higher raw activity -> lower score (illustrative only)
            score = max(0, min(100, 100 - (fdf["Risk_Factor"].mean() * 10)))
            comp_rows = "".join(
                f'<div class="mw-wc-row"><span class="mw-wc-label">{k}</span>'
                f'<span class="mw-wc-value" style="color:{TEXT_SECONDARY};">{w*100:.0f}%</span></div>'
                for k, w in weights.items()
            )
            st.markdown(f"""
            <div class="mw-panel">
                <div class="mw-eyebrow">MINEWATCH PROJECT-DEFINED</div>
                <div class="mw-page-title" style="font-size:1.4rem;">Monitoring Indicator</div>
                <div class="mw-kpi-value" style="font-size:2.1rem; margin: 6px 0;">{score:.0f} / 100</div>
                {comp_rows}
                <div class="mw-disclaimer">Project-defined analytical indicator. Not an official government
                compliance or safety score.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Not enough data in the current selection to compute the monitoring indicator.")


# ==================================================
# ENVIRONMENT
# ==================================================
elif page == "Environment":
    page_header("ANALYSIS", "Environmental Impact",
                "Environmental impact and breaches relative to production and mine type.")
    if fdf.empty:
        empty_state()
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(fdf, x="Production", y="Environmental_Impact",
                              color="Mine_Type", opacity=0.7,
                              color_discrete_sequence=[ENVIRONMENT, OPERATIONS, SECONDARY])
            st.plotly_chart(style_fig(fig, "Production vs Environmental Impact"), use_container_width=True)
        with col2:
            trend = fdf.groupby("Date", as_index=False)["Environmental_Impact"].mean()
            fig = px.line(trend, x="Date", y="Environmental_Impact")
            fig.update_traces(line_color=PLOT_COLORS["environment"])
            st.plotly_chart(style_fig(fig, "Environmental Impact Trend"), use_container_width=True)

        breach = fdf.groupby("Year", as_index=False)["Environmental_Breach"].sum()
        fig = px.bar(breach, x="Year", y="Environmental_Breach",
                     color_discrete_sequence=[SAFETY])
        st.plotly_chart(style_fig(fig, "Environmental Breaches by Year"), use_container_width=True)


# ==================================================
# ECONOMICS
# ==================================================
elif page == "Economics":
    page_header("ANALYSIS", "Economics",
                "Operational economics: revenue, cost, profit and margin.")
    if fdf.empty:
        empty_state()
    else:
        econ = fdf.groupby("Date", as_index=False)[["Revenue", "Operating_Cost", "Profit"]].sum()
        fig = go.Figure()
        for col, color in zip(["Revenue", "Operating_Cost", "Profit"],
                               [PLOT_COLORS["revenue"], PLOT_COLORS["cost"], PLOT_COLORS["profit"]]):
            width = 3 if col == "Profit" else 2
            fig.add_trace(go.Scatter(x=econ["Date"], y=econ[col], name=col,
                                      line=dict(color=color, width=width)))
        st.plotly_chart(style_fig(fig, "Revenue, Cost & Profit Over Time", height=460), use_container_width=True)

        margin = fdf.groupby("Date", as_index=False)["Profit_Margin"].mean()
        fig = px.line(margin, x="Date", y="Profit_Margin")
        fig.update_traces(line_color=PLOT_COLORS["profit"])
        st.plotly_chart(style_fig(fig, "Average Profit Margin (%) Over Time"), use_container_width=True)

        by_commodity = fdf.groupby("Commodity", as_index=False).agg(
            Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
        fig = px.bar(by_commodity, x="Commodity", y=["Revenue", "Profit"], barmode="group",
                     color_discrete_map={"Revenue": PLOT_COLORS["revenue"], "Profit": PLOT_COLORS["profit"]})
        st.plotly_chart(style_fig(fig, "Revenue & Profit by Commodity"), use_container_width=True)


# ==================================================
# REGIONAL PERFORMANCE
# ==================================================
elif page == "Regional Performance":
    page_header("ANALYSIS", "Regional Performance",
                "Compare mining activity and productivity across regions and states.")
    if fdf.empty:
        empty_state()
    else:
        region = fdf.groupby(["Region", "State"], as_index=False).agg(
            Production=("Production", "sum"),
            Productivity=("Productivity_Index", "mean"),
            Mines=("Mine_ID", "nunique"))
        fig = px.scatter(region, x="Production", y="Productivity", size="Production",
                          color="Region", hover_name="State", size_max=45)
        st.plotly_chart(style_fig(fig, "Production vs Productivity by State (bubble size = production)",
                                   height=480), use_container_width=True)

        table = region.sort_values("Production", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)


# ==================================================
# FORECASTING
# ==================================================
elif page == "Forecasting":
    page_header("ANALYSIS", "Experimental Forecasting",
                "A simple linear projection of production, for illustration only.")
    st.markdown("""
    <div class="mw-info-box">
    Experimental forecasting based on historical project data using a simple linear model.
    This does NOT predict actual future mine production with certainty - it is a
    classroom-level illustration of a forecasting approach.
    </div>
    """, unsafe_allow_html=True)
    if fdf.empty:
        empty_state()
    else:
        monthly = fdf.groupby("Date", as_index=False)["Production"].sum().sort_values("Date")
        monthly["t"] = np.arange(len(monthly))

        if len(monthly) < 6:
            st.info("Select a wider date range (at least 6 months) to generate a forecast.")
        else:
            horizon = st.slider("Months to forecast", 3, 24, 12)

            X = monthly[["t"]].values
            y = monthly["Production"].values

            if SKLEARN_AVAILABLE:
                model = LinearRegression()
                model.fit(X, y)
                future_t = np.arange(len(monthly), len(monthly) + horizon).reshape(-1, 1)
                preds = model.predict(future_t)
            else:
                coeffs = np.polyfit(monthly["t"], y, 1)
                future_t = np.arange(len(monthly), len(monthly) + horizon)
                preds = np.polyval(coeffs, future_t)

            last_date = monthly["Date"].max()
            future_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Production"],
                                      name="Historical (Actual)",
                                      line=dict(color=PLOT_COLORS["historical"])))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Experimental Forecast",
                                      line=dict(color=PLOT_COLORS["forecast"], dash="dash")))
            st.plotly_chart(style_fig(fig, "Actual vs Forecast Production", height=460),
                             use_container_width=True)
            st.markdown("""
            <div class="mw-disclaimer">
            Model: simple linear trend fit on monthly aggregated production for the current
            filter selection. No seasonality, external factors, or mine-level detail is modeled.
            Experimental historical projection - this forecast is not a guarantee of future production.
            </div>
            """, unsafe_allow_html=True)


# ==================================================
# MINE EXPLORER
# ==================================================
elif page == "Mine Explorer":
    page_header("ANALYSIS", "Mine Explorer",
                "A detailed profile of production, safety, environmental and economic performance for one mine.")
    mine_options = sorted(df["Mine_ID"].unique())
    selected_mine = st.selectbox("Select a mine", mine_options)

    mdf = df[df["Mine_ID"] == selected_mine]
    if mdf.empty:
        empty_state()
    else:
        profile = mdf.iloc[0]
        st.markdown(f"""
        <div class="mw-panel">
            <div class="mw-eyebrow">MINE PROFILE</div>
            <div class="mw-page-title" style="font-size:1.3rem;">{profile['Mine_Name']}
                <span class="mw-badge mw-badge-blue">{selected_mine}</span>
            </div>
            <div class="mw-profile-grid">
                <div><div class="mw-profile-item-label">Region</div><div class="mw-profile-item-value">{profile['Region']}</div></div>
                <div><div class="mw-profile-item-label">State</div><div class="mw-profile-item-value">{profile['State']}</div></div>
                <div><div class="mw-profile-item-label">Mine Type</div><div class="mw-profile-item-value">{profile['Mine_Type']}</div></div>
                <div><div class="mw-profile-item-label">Commodity</div><div class="mw-profile-item-value">{profile['Commodity']}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Production", format_num(mdf["Production"].sum()), "tonnes",
                      accent=PLOT_COLORS["production"])
        with c2:
            kpi_card("Productivity", f"{mdf['Productivity_Index'].mean():.2f}", "tonnes/hour",
                      accent=PLOT_COLORS["productivity"])
        with c3:
            kpi_card("Safety Incidents", f"{int(mdf['Safety_Incidents'].sum())}", "",
                      accent=PLOT_COLORS["safety"])
        with c4:
            kpi_card("Violations", f"{int(mdf['Violations'].sum())}", "",
                      accent=PLOT_COLORS["violations"])

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            kpi_card("Environmental Impact", f"{mdf['Environmental_Impact'].mean():.2f}", "avg",
                      accent=PLOT_COLORS["environment"])
        with c6:
            kpi_card("Energy Intensity", f"{mdf['Energy_Intensity'].mean():.3f}", "avg",
                      accent=PLOT_COLORS["energy"])
        with c7:
            kpi_card("Waste Intensity", f"{mdf['Waste_Intensity'].mean():.3f}", "avg",
                      accent=PLOT_COLORS["waste"])
        with c8:
            kpi_card("Pending Actions", f"{int(mdf['Pending_Actions'].sum())}", "",
                      accent=PLOT_COLORS["pending"])

        c9, c10, c11 = st.columns(3)
        with c9:
            kpi_card("Total Revenue", format_num(mdf["Revenue"].sum()), "", accent=PLOT_COLORS["revenue"])
        with c10:
            kpi_card("Total Operating Cost", format_num(mdf["Operating_Cost"].sum()), "", accent=PLOT_COLORS["cost"])
        with c11:
            kpi_card("Total Profit", format_num(mdf["Profit"].sum()), "", accent=PLOT_COLORS["profit"])

        st.markdown("#### Performance Over Time")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Production", "Safety", "Environmental", "Energy & Waste", "Profitability"])
        with tab1:
            fig = px.line(mdf.sort_values("Date"), x="Date", y="Production")
            fig.update_traces(line_color=PLOT_COLORS["production"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with tab2:
            fig = px.line(mdf.sort_values("Date"), x="Date", y="Safety_Incidents")
            fig.update_traces(line_color=PLOT_COLORS["safety"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with tab3:
            fig = px.line(mdf.sort_values("Date"), x="Date", y="Environmental_Impact")
            fig.update_traces(line_color=PLOT_COLORS["environment"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with tab4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mdf.sort_values("Date")["Date"],
                                      y=mdf.sort_values("Date")["Energy_Intensity"], name="Energy Intensity",
                                      line=dict(color=PLOT_COLORS["energy"])))
            fig.add_trace(go.Scatter(x=mdf.sort_values("Date")["Date"],
                                      y=mdf.sort_values("Date")["Waste_Intensity"], name="Waste Intensity",
                                      line=dict(color=PLOT_COLORS["waste"], dash="dot")))
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with tab5:
            fig = px.line(mdf.sort_values("Date"), x="Date", y="Profit")
            fig.update_traces(line_color=PLOT_COLORS["profit"])
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("#### Mine Insight")
        if st.button("Generate Mine Insight"):
            yearly = mdf.groupby("Year").agg(
                Production=("Production", "sum"),
                Safety_Incidents=("Safety_Incidents", "sum"),
                Violations=("Violations", "sum"),
            ).sort_index()
            if len(yearly) >= 2:
                first, last = yearly.iloc[0], yearly.iloc[-1]
                prod_change = pct_change(last["Production"], first["Production"])
                inc_change = pct_change(last["Safety_Incidents"], first["Safety_Incidents"])
                viol_change = pct_change(last["Violations"], first["Violations"])
                finding_card("PRODUCTION",
                              f"Production changed by {prod_change:+.1f}% from {yearly.index.min()} to {yearly.index.max()}.",
                              accent=PLOT_COLORS["production"])
                finding_card("SAFETY",
                              f"Safety incidents changed by {inc_change:+.1f}% over the same period.",
                              accent=PLOT_COLORS["safety"])
                finding_card("VIOLATIONS",
                              f"Violations changed by {viol_change:+.1f}% over the same period.",
                              accent=PLOT_COLORS["violations"])
            else:
                st.caption("Not enough years of data in the current selection to compute a trend insight.")


# ==================================================
# AI INSIGHTS
# ==================================================
elif page == "AI Insights":
    page_header("TOOLS", "AI Insights",
                "AI-generated analytical observation, with a rule-based fallback when no API key is configured.")

    if fdf.empty:
        empty_state()
    else:
        result = what_changed(fdf, "Year")
        if result is None:
            st.info("Select a range covering at least two years to generate AI insights.")
        else:
            wc_df, prev_y, last_y = result
            top_violation = None
            if fdf["Violation_Category"].notna().any():
                top_violation = fdf["Violation_Category"].value_counts().idxmax()

            summary = {
                "period": f"{prev_y} vs {last_y}",
                "mines_in_selection": int(fdf["Mine_ID"].nunique()),
                "production_change_pct": round(float(wc_df.loc[wc_df.Metric == "Production", "Change %"].iloc[0] or 0), 2),
                "productivity_change_pct": round(float(wc_df.loc[wc_df.Metric == "Productivity (avg)", "Change %"].iloc[0] or 0), 2),
                "incident_change_pct": round(float(wc_df.loc[wc_df.Metric == "Safety Incidents", "Change %"].iloc[0] or 0), 2),
                "violation_change_pct": round(float(wc_df.loc[wc_df.Metric == "Violations", "Change %"].iloc[0] or 0), 2),
                "top_violation_category": top_violation,
                "pending_actions": int(fdf["Pending_Actions"].sum()),
                "energy_intensity_change_pct": round(float(wc_df.loc[wc_df.Metric == "Energy Intensity", "Change %"].iloc[0] or 0), 2),
                "waste_intensity_change_pct": round(float(wc_df.loc[wc_df.Metric == "Waste Intensity", "Change %"].iloc[0] or 0), 2),
            }

            with st.expander("Structured summary sent for analysis"):
                st.json(summary)

            def rule_based_insight(s):
                lines = []
                lines.append(
                    ("KEY OBSERVATION",
                     f"Production changed {s['production_change_pct']:+.1f}% "
                     f"and productivity changed {s['productivity_change_pct']:+.1f}% between "
                     f"{prev_y} and {last_y}.")
                )
                lines.append(
                    ("IMPORTANT TREND",
                     f"Safety incidents changed {s['incident_change_pct']:+.1f}% "
                     f"while violations changed {s['violation_change_pct']:+.1f}% over the same period.")
                )
                if s["top_violation_category"]:
                    lines.append(
                        ("RECURRING ISSUE",
                         f"'{s['top_violation_category']}' is the most frequently recorded violation "
                         f"category in the current selection.")
                    )
                lines.append(
                    ("AREA REQUIRING MONITORING",
                     f"{s['pending_actions']} corrective actions remain pending across the selected mines.")
                )
                return lines

            ai_used = False
            insight_lines = None

            if GEMINI_API_KEY:
                try:
                    import requests
                    prompt = (
                        "You are an analytics assistant for a student mining-EDA dashboard called "
                        "MineWatch. Using ONLY the structured JSON summary below (do not invent "
                        "numbers or causes), produce exactly four short bullet points: "
                        "1) Key observation 2) Important trend 3) Recurring issue "
                        "4) Area requiring monitoring. Do not claim official compliance status or "
                        "predict accidents.\n\nJSON:\n" + json.dumps(summary)
                    )
                    url = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                    )
                    resp = requests.post(
                        url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        raw_lines = [ln.strip("-• ").strip() for ln in text.split("\n") if ln.strip()]
                        labels = ["KEY OBSERVATION", "IMPORTANT TREND", "RECURRING ISSUE", "AREA REQUIRING MONITORING"]
                        insight_lines = list(zip(labels, raw_lines[:4])) if raw_lines else None
                        ai_used = insight_lines is not None
                except Exception:
                    ai_used = False

            st.markdown("""
            <div class="mw-eyebrow" style="color:%s;">AI-GENERATED ANALYTICAL OBSERVATION</div>
            """ % SECONDARY, unsafe_allow_html=True)

            if ai_used and insight_lines:
                st.markdown('<span class="mw-badge">AI-generated</span>', unsafe_allow_html=True)
                for cat, ln in insight_lines:
                    finding_card(cat, ln, accent=SECONDARY)
            else:
                if GEMINI_API_KEY:
                    st.caption("AI request unavailable or failed - showing rule-based analytical observations.")
                else:
                    st.caption("AI API not configured. Showing rule-based analytical observations.")
                for cat, ln in rule_based_insight(summary):
                    finding_card(cat, ln, accent=SECONDARY)

            st.markdown("""
            <div class="mw-disclaimer">This AI/rule-based layer is analytical and observational only -
            it does not control the dashboard, alter underlying data, or issue compliance determinations.</div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### What Changed?")
        result = what_changed(fdf, "Year")
        if result:
            what_changed_panel(*result)


# ==================================================
# DATA EXPLORER
# ==================================================
elif page == "Data Explorer":
    page_header("TOOLS", "Data Explorer",
                "Inspect the cleaning pipeline, filtered dataset and descriptive statistics.")

    st.markdown("#### Data Cleaning Methodology")
    st.markdown(f"""
    <div class="mw-panel">
    Raw records ({clean_stats['raw_rows']:,}) &rarr;
    Duplicate removal (&minus;{clean_stats['duplicates_found']}) &rarr;
    Category normalization &rarr;
    Missing-value treatment (median imputation by mine; {clean_stats['missing_before']} &rarr; {clean_stats['missing_after']} missing) &rarr;
    Derived metrics (Energy Intensity, Waste Intensity, Incident Rate) &rarr;
    EDA dataset ({clean_stats['rows_after_dedup']:,} rows)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Filtered Dataset")
    if fdf.empty:
        empty_state()
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Rows", f"{len(fdf):,}", "filtered", accent=NEUTRAL)
        with c2:
            kpi_card("Columns", f"{fdf.shape[1]}", "", accent=NEUTRAL)
        with c3:
            kpi_card("Missing Values", f"{int(fdf.isna().sum().sum()):,}", "", accent=NEUTRAL)
        with c4:
            kpi_card("Duplicates Removed", f"{clean_stats['duplicates_found']:,}", "at load time", accent=NEUTRAL)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_raw, tab_filtered, tab_stats = st.tabs(["Raw (sample)", "Filtered Data", "Descriptive Statistics"])
        with tab_raw:
            st.dataframe(raw_df.head(200), use_container_width=True, height=320)
        with tab_filtered:
            st.dataframe(fdf, use_container_width=True, height=380)
        with tab_stats:
            st.dataframe(fdf.describe().T, use_container_width=True)

        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("\u2B07 Download Filtered CSV", csv, "minewatch_filtered.csv", "text/csv",
                            type="primary", use_container_width=False)


# ==================================================
# ABOUT PROJECT
# ==================================================
elif page == "About Project":
    page_header("TOOLS", "About MineWatch V2",
                "Project positioning, dataset description and known limitations.")
    st.markdown("""
    <div class="mw-panel">
    <b>MineWatch</b> is an interactive Python-based EDA platform for analysing mining
    production, productivity, safety, environmental impact, waste, energy consumption
    and operational economics. It uses a realistic multi-year synthetic mining dataset,
    performs data cleaning and exploratory analysis, provides interactive Plotly
    visualizations and mine-level exploration, and optionally generates AI-assisted
    analytical observations.
    </div>

    #### Positioning
    This project is an **academic analytics prototype**, inspired by the SIH 2026
    problem statement SIH26024 (Coal Mine Governance & Compliance). It is **not**:
    - an official government compliance system
    - an official mining safety system or safety score
    - an accident prediction system
    - a real-time mine control system
    - a replacement for mining inspectors
    - an official SIH implementation

    #### Dataset
    - Approximately 3,000+ monthly records (2016-2024) across 30 fictional mines
    - Entirely synthetic / project-generated, with intentional data-quality issues
      (missing values, duplicates, inconsistent category labels, outliers) included
      to give the EDA notebook genuine cleaning work
    - Data relationships (production &rarr; energy, waste, revenue; reserves depleting
      over time; profit = revenue &minus; operating cost) are deliberately realistic

    #### Limitations
    - Forecasting is experimental (simple linear trend) and does not predict actual
      future production
    - The "MineWatch Analytical Indicator" (where shown) is a project-defined,
      transparent composite score - not an official compliance or safety score
    - Correlation shown throughout the dashboard indicates association, not causation
    - AI Insights are optional and constrained to a structured data summary; if no
      API key is configured, rule-based analytical observations are shown instead

    #### Tech Stack
    Python, Pandas, NumPy, Plotly, Streamlit, Scikit-learn, optional Gemini API.
    """, unsafe_allow_html=True)