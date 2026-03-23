import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from rca_engine import (
    load_datewise_kpi, load_hourly_kpi,
    add_prev_and_delta, add_prev_and_delta_hourly,
    get_top_dips, get_worst_hours_per_dip,
    normalize_cc_file, run_share_rca,
    build_dip_groups, CC_ACTION_MAP,
    generate_llm_rca,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Telecom KPI RCA Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────

ACCENT       = "#6C63FF"
ACCENT_LIGHT = "#8B83FF"
ACCENT_DARK  = "#4A42DB"
BG_CARD      = "rgba(255,255,255,0.04)"
BORDER       = "rgba(255,255,255,0.08)"
TEXT_DIM      = "rgba(255,255,255,0.5)"

st.markdown(f"""
<style>
  /* ── Global ─────────────────────────────────────────────────── */
  .main .block-container {{ padding-top: 1rem; max-width: 1400px; }}
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0f0f1a 0%, #141425 100%);
  }}
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stMarkdown li,
  section[data-testid="stSidebar"] label {{
      color: rgba(255,255,255,0.72) !important;
      font-size: 0.84rem;
  }}

  /* ── Sidebar brand block ────────────────────────────────────── */
  .sidebar-brand {{
      text-align: center; padding: 1.6rem 0.5rem 1.2rem;
  }}
  .sidebar-brand .logo {{
      width: 52px; height: 52px; margin: 0 auto 0.6rem;
      background: linear-gradient(135deg, {ACCENT} 0%, #a78bfa 100%);
      border-radius: 14px; display: flex; align-items: center;
      justify-content: center; font-size: 1.6rem;
      box-shadow: 0 4px 20px rgba(108,99,255,0.35);
  }}
  .sidebar-brand h2 {{
      margin: 0; font-size: 1.05rem; font-weight: 700;
      letter-spacing: 0.3px; color: #fff;
  }}
  .sidebar-brand p {{
      margin: 0.15rem 0 0; font-size: 0.72rem;
      color: {TEXT_DIM}; letter-spacing: 1.2px; text-transform: uppercase;
  }}

  /* ── Sidebar section labels ─────────────────────────────────── */
  .sb-section {{
      display: flex; align-items: center; gap: 0.45rem;
      margin: 1.2rem 0 0.5rem; padding-bottom: 0.35rem;
      border-bottom: 1px solid {BORDER};
      font-size: 0.7rem; font-weight: 600; letter-spacing: 1px;
      text-transform: uppercase; color: {ACCENT_LIGHT};
  }}

  /* ── Metric cards ───────────────────────────────────────────── */
  .m-card {{
      background: {BG_CARD};
      border: 1px solid {BORDER};
      border-radius: 12px; padding: 1rem 1.2rem;
      position: relative; overflow: hidden;
  }}
  .m-card::before {{
      content: ""; position: absolute; top: 0; left: 0;
      width: 100%; height: 3px;
      background: linear-gradient(90deg, {ACCENT}, #a78bfa);
  }}
  .m-card .label {{
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px;
      color: {TEXT_DIM}; margin-bottom: 0.25rem;
  }}
  .m-card .value {{
      font-size: 1.65rem; font-weight: 700; color: #fff;
      line-height: 1.15;
  }}

  /* ── Step headers ───────────────────────────────────────────── */
  .step-hdr {{
      display: flex; align-items: center; gap: 0.75rem;
      margin: 2rem 0 1rem; padding: 0.85rem 1.2rem;
      background: {BG_CARD};
      border: 1px solid {BORDER};
      border-left: 4px solid {ACCENT};
      border-radius: 8px;
  }}
  .step-hdr .num {{
      background: {ACCENT}; color: #fff;
      width: 28px; height: 28px; border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
  }}
  .step-hdr .txt {{
      font-size: 0.95rem; font-weight: 600; color: #e0e0e0;
  }}

  /* ── Info banners ───────────────────────────────────────────── */
  .info-banner {{
      background: rgba(108,99,255,0.08);
      border: 1px solid rgba(108,99,255,0.2);
      border-radius: 10px; padding: 2rem; text-align: center;
      margin: 1rem 0;
  }}
  .info-banner .icon {{ font-size: 2.2rem; margin-bottom: 0.4rem; }}
  .info-banner .title {{ font-size: 1.05rem; font-weight: 600; color: #e0e0e0; }}
  .info-banner .sub {{ font-size: 0.82rem; color: {TEXT_DIM}; margin-top: 0.2rem; }}

  /* ── RCA summary card ───────────────────────────────────────── */
  .rca-card {{
      background: {BG_CARD};
      border: 1px solid {BORDER};
      border-radius: 10px; padding: 1.2rem 1.4rem;
      margin-bottom: 0.8rem;
  }}
  .rca-card .rca-head {{
      display: flex; align-items: center; gap: 0.5rem;
      margin-bottom: 0.6rem;
  }}
  .rca-card .rca-head .badge {{
      background: {ACCENT}; color: #fff; font-size: 0.68rem;
      padding: 0.15rem 0.55rem; border-radius: 6px; font-weight: 600;
  }}
  .rca-card .rca-body {{
      font-size: 0.88rem; line-height: 1.65; color: rgba(255,255,255,0.78);
  }}

  /* ── CC detail pill ─────────────────────────────────────────── */
  .cc-pill {{
      display: inline-flex; align-items: center; gap: 0.35rem;
      background: {BG_CARD}; border: 1px solid {BORDER};
      border-radius: 8px; padding: 0.4rem 0.75rem;
      margin: 0.25rem 0.15rem; font-size: 0.82rem;
  }}
  .cc-pill .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .cc-pill .dot.up {{ background: #ef4444; }}
  .cc-pill .dot.dn {{ background: #22c55e; }}

  /* ── Download buttons ───────────────────────────────────────── */
  .stDownloadButton > button {{
      background: {BG_CARD} !important;
      border: 1px solid {BORDER} !important;
      color: #e0e0e0 !important; border-radius: 8px !important;
      font-size: 0.82rem !important; transition: all 0.2s;
  }}
  .stDownloadButton > button:hover {{
      border-color: {ACCENT} !important;
      background: rgba(108,99,255,0.1) !important;
  }}

  /* ── Misc ────────────────────────────────────────────────────── */
  div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px 8px 0 0; font-size: 0.82rem;
  }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def step_header(title, step_num):
    st.markdown(
        f'<div class="step-hdr">'
        f'<div class="num">{step_num}</div>'
        f'<div class="txt">{title}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f'<div class="m-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def info_banner(icon, title, subtitle):
    st.markdown(
        f'<div class="info-banner">'
        f'<div class="icon">{icon}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="sub">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def read_uploaded(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, low_memory=False)
    return pd.read_excel(uploaded_file)


def sb_section(label):
    st.sidebar.markdown(
        f'<div class="sb-section">{label}</div>',
        unsafe_allow_html=True,
    )


# ── Plotly theme ──────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[ACCENT, "#a78bfa", "#38bdf8", "#22c55e", "#f59e0b",
              "#ef4444", "#ec4899", "#14b8a6"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    '<div class="sidebar-brand">'
    '<div class="logo">📡</div>'
    '<h2>KPI RCA Engine</h2>'
    '<p>Telecom Analytics</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── File Uploads ──────────────────────────────────────────────────────────────

sb_section("DATA SOURCES")

datewise_file = st.sidebar.file_uploader(
    "Date-wise KPI file",
    type=["csv", "xlsx", "xls"],
    help="Daily aggregated KPI values (e.g. date_wise_new.xlsx)",
    key="datewise",
)
hourly_file = st.sidebar.file_uploader(
    "Hour-wise KPI file",
    type=["csv", "xlsx", "xls"],
    help="Hourly KPI values (e.g. date_hour_wise_new.xlsx)",
    key="hourly",
)
cc_file = st.sidebar.file_uploader(
    "Clear Code file",
    type=["csv", "xlsx", "xls"],
    help="MSC Clear Code detail (e.g. MSC_CC.csv or new_lusr.xlsx)",
    key="cc",
)

# ── API Key ───────────────────────────────────────────────────────────────────

sb_section("AI CONFIGURATION")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-proj-...",
    help="Required for AI-generated RCA summaries. Your key is never stored.",
)

llm_model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0,
    help="Select the OpenAI model for RCA narrative generation.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

sb_section("ANALYSIS PARAMETERS")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────

if not datewise_file:
    st.markdown("")  # spacer
    info_banner(
        "📡",
        "Telecom KPI Root Cause Analysis",
        "Upload your date-wise KPI file in the sidebar to begin automated dip detection and CC-share RCA."
    )
    st.stop()

# ── Step 0: Detect columns ───────────────────────────────────────────────────

datewise_raw = read_uploaded(datewise_file)

date_col_candidates = [c for c in datewise_raw.columns
                       if "date" in c.lower() or "d1date" in c.lower()]
date_col = st.sidebar.selectbox(
    "Date column",
    date_col_candidates or datewise_raw.columns.tolist(),
    index=0,
)

skip_patterns = ["date", "hour", "d1date", "d1hour", "msc", "unnamed"]
kpi_candidates = [
    c for c in datewise_raw.columns
    if datewise_raw[c].dtype in ["float64", "int64", "float32", "int32"]
    and not any(p in c.lower() for p in skip_patterns)
]
if not kpi_candidates:
    for c in datewise_raw.columns:
        if not any(p in c.lower() for p in skip_patterns):
            try:
                pd.to_numeric(datewise_raw[c], errors="raise")
                kpi_candidates.append(c)
            except Exception:
                pass

kpi_col = st.sidebar.selectbox("KPI column", kpi_candidates)

kpi_name_lower = kpi_col.lower()
if "cssr" in kpi_name_lower or "call setup" in kpi_name_lower:
    detected_kpi = "CSSR"
elif "paging" in kpi_name_lower or "psr" in kpi_name_lower:
    detected_kpi = "PSR"
elif "location" in kpi_name_lower or "lusr" in kpi_name_lower or "lu success" in kpi_name_lower:
    detected_kpi = "LUSR"
else:
    detected_kpi = kpi_col[:10]

kpi_label = st.sidebar.text_input("KPI display name", value=detected_kpi)
top_n_dips = st.sidebar.slider("Top N dip days", 1, 20, 5)
ref_days = st.sidebar.number_input("Reference offset (days)", 1, 30, 7)
fallback_days = st.sidebar.number_input("Fallback offset (days)", 0, 60, 14,
                                        help="Set to 0 to disable")
fallback_days = fallback_days if fallback_days else None
top_n_cc = st.sidebar.slider("Top N CCs per dip", 1, 10, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DAILY KPI TREND
# ══════════════════════════════════════════════════════════════════════════════

step_header("Daily KPI Trend & Dip Detection", 1)

date_df = load_datewise_kpi(datewise_raw, date_col, kpi_col)
date_df = add_prev_and_delta(date_df, kpi_col)
top_dips = get_top_dips(date_df, kpi_col, top_n=top_n_dips)

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("KPI", kpi_label)
with c2: metric_card("Data Points", str(len(date_df)))
with c3: metric_card(f"Avg {kpi_label}", f"{date_df[kpi_col].mean():.2f}")
with c4: metric_card("Dip Days", str(len(top_dips)))

# Daily trend — adaptive Y-axis range
kpi_min = date_df[kpi_col].min()
kpi_max = date_df[kpi_col].max()
kpi_range = kpi_max - kpi_min if kpi_max != kpi_min else 1
y_pad = kpi_range * 0.15  # 15% padding above and below
y_lo = kpi_min - y_pad
y_hi = kpi_max + y_pad

fig_daily = go.Figure()
fig_daily.add_trace(go.Scatter(
    x=date_df["date"], y=date_df[kpi_col],
    mode="lines+markers", name=kpi_label,
    line=dict(color=ACCENT, width=2.5, shape="spline"),
    marker=dict(size=5),
))
if not top_dips.empty:
    dip_dates_set = set(pd.to_datetime(top_dips["date"]).dt.normalize())
    dip_points = date_df[date_df["date"].isin(dip_dates_set)]
    fig_daily.add_trace(go.Scatter(
        x=dip_points["date"], y=dip_points[kpi_col],
        mode="markers", name="Dip Days",
        marker=dict(color="#ef4444", size=11, symbol="triangle-down",
                    line=dict(width=1, color="#fff")),
    ))
fig_daily.update_layout(
    **PLOT_LAYOUT, height=420,
    title=f"Daily {kpi_label} Trend",
    xaxis_title="Date", yaxis_title=kpi_label,
    yaxis=dict(range=[y_lo, y_hi]),
    legend=dict(orientation="h", y=-0.12),
)
st.plotly_chart(fig_daily, use_container_width=True)

col_a, col_b = st.columns([3, 2])
with col_a:
    fig_delta = px.histogram(
        date_df.dropna(subset=["delta"]), x="delta", nbins=30,
        title=f"Day-over-Day {kpi_label} Delta Distribution",
        color_discrete_sequence=["#a78bfa"],
    )
    fig_delta.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_delta.update_layout(**PLOT_LAYOUT, height=350)
    st.plotly_chart(fig_delta, use_container_width=True)

with col_b:
    st.markdown(f"**Top {top_n_dips} Worst Dip Days**")
    disp_dips = top_dips.copy()
    disp_dips["date"] = pd.to_datetime(disp_dips["date"]).dt.strftime("%Y-%m-%d")
    for c in [kpi_col, "prev", "delta"]:
        if c in disp_dips.columns:
            disp_dips[c] = disp_dips[c].round(4)
    st.dataframe(disp_dips, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — HOURLY DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════

if not hourly_file:
    info_banner("🕐", "Hourly Drill-Down", "Upload an hour-wise KPI file to identify worst hours per dip day.")
    st.stop()

step_header("Hourly Drill-Down — Worst Hours per Dip Day", 2)

hourly_raw = read_uploaded(hourly_file)

hour_col_candidates = [c for c in hourly_raw.columns if "hour" in c.lower() or "d1hour" in c.lower()]
hour_col_detected = hour_col_candidates[0] if hour_col_candidates else hourly_raw.columns[1]

date_col_hourly_candidates = [c for c in hourly_raw.columns if "date" in c.lower() or "d1date" in c.lower()]
date_col_hourly = date_col_hourly_candidates[0] if date_col_hourly_candidates else hourly_raw.columns[0]

if kpi_col not in hourly_raw.columns:
    hourly_kpi_candidates = [c for c in hourly_raw.columns if c in kpi_candidates or c == kpi_col]
    if not hourly_kpi_candidates:
        st.error(f"Column **{kpi_col}** not found in hourly file. Available: {hourly_raw.columns.tolist()}")
        st.stop()

hour_df = load_hourly_kpi(hourly_raw, date_col_hourly, hour_col_detected, kpi_col)
hour_df = add_prev_and_delta_hourly(hour_df, kpi_col)
worst_hours = get_worst_hours_per_dip(hour_df, top_dips)

if worst_hours.empty:
    st.warning("No matching hourly data found for the detected dip dates.")
    st.stop()

c1, c2 = st.columns(2)
with c1: metric_card("Hourly Records", f"{len(hour_df):,}")
with c2: metric_card("Worst Hours Found", str(len(worst_hours)))

# Heatmap
dip_date_list = pd.to_datetime(top_dips["date"]).dt.normalize().tolist()
dip_hourly = hour_df[hour_df["date"].isin(dip_date_list)].copy()

if not dip_hourly.empty:
    dip_hourly["date_str"] = dip_hourly["date"].dt.strftime("%Y-%m-%d")
    pivot = dip_hourly.pivot_table(index="date_str", columns="hour",
                                   values=kpi_col, aggfunc="mean")
    fig_heatmap = px.imshow(
        pivot, aspect="auto",
        title=f"Hourly {kpi_label} Heatmap (Dip Days)",
        color_continuous_scale="RdYlGn",
    )
    fig_heatmap.update_layout(**PLOT_LAYOUT, height=320)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Per-day tabs
tabs = st.tabs([f"{pd.Timestamp(d).strftime('%b %d, %Y')}" for d in dip_date_list[:5]])
for tab, dip_date in zip(tabs, dip_date_list[:5]):
    with tab:
        day_data = hour_df[hour_df["date"] == dip_date].copy()
        if day_data.empty:
            st.write("No hourly data available.")
            continue

        worst_row = worst_hours[worst_hours["date"].dt.normalize() == dip_date]
        worst_h = int(worst_row["worst_hour"].iloc[0]) if not worst_row.empty else None

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=day_data["hour"], y=day_data[kpi_col],
            mode="lines+markers", name=kpi_label,
            line=dict(color=ACCENT, width=2.5, shape="spline"),
            marker=dict(size=5),
        ))
        if worst_h is not None:
            worst_val = day_data[day_data["hour"] == worst_h][kpi_col]
            if not worst_val.empty:
                fig_hour.add_trace(go.Scatter(
                    x=[worst_h], y=[worst_val.iloc[0]],
                    mode="markers", name="Worst Hour",
                    marker=dict(color="#ef4444", size=14, symbol="star",
                                line=dict(width=1.5, color="#fff")),
                ))
        h_min = day_data[kpi_col].min()
        h_max = day_data[kpi_col].max()
        h_rng = h_max - h_min if h_max != h_min else 1
        fig_hour.update_layout(
            **PLOT_LAYOUT, height=300,
            title=f"{kpi_label} — {pd.Timestamp(dip_date).strftime('%b %d, %Y')}  (worst hour: {worst_h})",
            xaxis_title="Hour", yaxis_title=kpi_label,
            yaxis=dict(range=[h_min - h_rng * 0.15, h_max + h_rng * 0.15]),
        )
        st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("**Worst Hours Summary**")
worst_display = worst_hours.copy()
worst_display["date"] = pd.to_datetime(worst_display["date"]).dt.strftime("%Y-%m-%d")
worst_display["delta"] = worst_display["delta"].round(4)
st.dataframe(worst_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — CC-SHARE RCA
# ══════════════════════════════════════════════════════════════════════════════

if not cc_file:
    info_banner("🔍", "Clear Code RCA", "Upload a Clear Code file to run share-based root cause analysis.")
    st.stop()

step_header("Clear Code Share-Based RCA", 3)

cc_raw = read_uploaded(cc_file)

with st.expander("Column Mapping (CC File)", expanded=False):
    cc_cols = cc_raw.columns.tolist()
    cc_date_col = st.selectbox(
        "Date column",
        [c for c in cc_cols if "date" in c.lower()] or cc_cols, key="cc_date"
    )
    cc_hour_col = st.selectbox(
        "Hour column",
        [c for c in cc_cols if "hour" in c.lower()] or cc_cols, key="cc_hour"
    )
    cc_msc_col = st.selectbox(
        "MSC column",
        [c for c in cc_cols if "msc" in c.lower()] or cc_cols, key="cc_msc"
    )
    cc_id_col = st.selectbox(
        "CC ID column",
        [c for c in cc_cols if "cc" in c.lower() and "id" in c.lower()] or
        [c for c in cc_cols if "cc" in c.lower()] or cc_cols, key="cc_id"
    )
    numeric_cc_cols = [c for c in cc_cols
                       if cc_raw[c].dtype in ["float64", "int64", "float32", "int32"]]
    value_candidates = (
        [c for c in numeric_cc_cols if "internal" in c.lower()] or
        [c for c in numeric_cc_cols if "failure" in c.lower() or "reason" in c.lower()] or
        [c for c in numeric_cc_cols if "clear" in c.lower()] or
        numeric_cc_cols or cc_cols
    )
    cc_value_col = st.selectbox("Value column", value_candidates, key="cc_value")

# Target CCs
default_ccs = {
    "CSSR": "A03, 706, 603, B13, B1A, B16, B1C, B2D, B1B, B2C, 817, A09, B32, B17, 811, B1E",
    "PSR": "10, 12, 603",
    "LUSR": "0307, 0811, 0812, 081B, 081C, 0B13",
}
target_cc_input = st.sidebar.text_area(
    "Target CC IDs (comma-separated)",
    value=default_ccs.get(kpi_label, ""),
    help="Clear Codes to include in the RCA",
)
target_cc_ids = [t.strip() for t in target_cc_input.split(",") if t.strip()]

if not target_cc_ids:
    st.warning("Enter at least one target CC ID in the sidebar.")
    st.stop()

cc_df = normalize_cc_file(
    cc_raw, date_col=cc_date_col, hour_col=cc_hour_col,
    msc_col=cc_msc_col, ccid_col=cc_id_col, value_col=cc_value_col,
)

c1, c2, c3 = st.columns(3)
with c1: metric_card("CC Records", f"{len(cc_df):,}")
with c2: metric_card("Unique MSCs", str(cc_df["msc"].nunique()))
with c3: metric_card("Unique CC IDs", str(cc_df["cc_id"].nunique()))

with st.spinner(f"Running CC-share RCA for {kpi_label}..."):
    rca_df, skip_df = run_share_rca(
        worst_hours, cc_df, target_cc_ids, kpi_label,
        ref_days=ref_days, fallback_days=fallback_days,
    )

if rca_df.empty:
    st.warning("No RCA results. Check files and parameters.")
    if not skip_df.empty:
        with st.expander("Skipped Cases"):
            st.dataframe(skip_df, use_container_width=True, hide_index=True)
    st.stop()

n_events = rca_df.groupby(["Date", "Hour", "MSC"]).ngroups
st.success(f"**{len(rca_df)}** CC entries across **{n_events}** dip events")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

step_header("RCA Visualizations", 4)

cc_impact = (
    rca_df.groupby("CC_ID")["Share_Delta"]
    .agg(["mean", "count", "sum"]).reset_index()
    .rename(columns={"mean": "Avg_Delta", "count": "Occurrences", "sum": "Total_Delta"})
    .sort_values("Total_Delta", key=abs, ascending=False)
)

col_a, col_b = st.columns(2)
with col_a:
    fig = px.bar(
        cc_impact.head(10), x="CC_ID", y="Total_Delta",
        color="Total_Delta", color_continuous_scale="RdBu_r",
        title=f"Top 10 Impactful Clear Codes ({kpi_label})",
    )
    fig.update_layout(**PLOT_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = px.bar(
        cc_impact.head(10), x="CC_ID", y="Occurrences",
        title=f"CC Frequency ({kpi_label})",
        color_discrete_sequence=[ACCENT_LIGHT],
    )
    fig.update_layout(**PLOT_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)

msc_impact = (
    rca_df.groupby("MSC")["Share_Delta"]
    .sum().reset_index()
    .sort_values("Share_Delta", key=abs, ascending=False)
)
fig = px.bar(
    msc_impact.head(15), x="MSC", y="Share_Delta",
    color="Share_Delta", color_continuous_scale="RdBu_r",
    title=f"Net Share Delta by MSC ({kpi_label})",
)
fig.update_layout(**PLOT_LAYOUT, height=350)
st.plotly_chart(fig, use_container_width=True)

# Per-event tabs
event_keys = rca_df.groupby(["Date", "Hour", "MSC"]).size().reset_index(name="n")
event_tabs = st.tabs([
    f"{r['Date']}  H{r['Hour']}  {r['MSC']}"
    for _, r in event_keys.head(10).iterrows()
])

for tab, (_, evt) in zip(event_tabs, event_keys.head(10).iterrows()):
    with tab:
        evt_data = rca_df[
            (rca_df["Date"] == evt["Date"]) &
            (rca_df["Hour"] == evt["Hour"]) &
            (rca_df["MSC"] == evt["MSC"])
        ].copy()

        ca, cb = st.columns([5, 3])
        with ca:
            fig_evt = go.Figure()
            fig_evt.add_trace(go.Bar(
                x=evt_data["CC_ID"], y=evt_data["Share_Current"],
                name="Current", marker_color=ACCENT,
            ))
            fig_evt.add_trace(go.Bar(
                x=evt_data["CC_ID"], y=evt_data["Share_Reference"],
                name="Reference", marker_color="rgba(255,255,255,0.15)",
            ))
            fig_evt.update_layout(**PLOT_LAYOUT, barmode="group",
                                  title="Current vs Reference Share", height=320)
            st.plotly_chart(fig_evt, use_container_width=True)

        with cb:
            for _, row in evt_data.iterrows():
                dot_cls = "up" if row["Direction"] == "Increased" else "dn"
                cc_def = CC_ACTION_MAP.get(row["CC_ID"], "Unknown")
                st.markdown(
                    f'<div class="cc-pill">'
                    f'<span class="dot {dot_cls}"></span>'
                    f'<strong>{row["CC_ID"]}</strong> '
                    f'R{int(row["Impact_Rank"])} '
                    f'{row["Share_Delta"]:+.4f}'
                    f'<br/><small style="color:{TEXT_DIM};font-size:0.72rem">{cc_def}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — AI-GENERATED RCA SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

step_header("AI-Generated RCA Narratives", 5)

dip_groups = build_dip_groups(rca_df, top_n=top_n_cc)

if not api_key:
    info_banner(
        "🔑",
        "Enter your OpenAI API key",
        "Add your API key in the sidebar under AI Configuration to generate natural-language RCA summaries.",
    )
else:
    if dip_groups:
        if st.button("Generate AI RCA Summaries", type="primary",
                     use_container_width=True):
            with st.spinner("Generating RCA narratives with AI... this may take a moment."):
                try:
                    rca_summaries = generate_llm_rca(
                        dip_groups, kpi_label, api_key, model=llm_model,
                    )
                    st.session_state["rca_summaries"] = rca_summaries
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

        if "rca_summaries" in st.session_state:
            for row in st.session_state["rca_summaries"]:
                st.markdown(
                    f'<div class="rca-card">'
                    f'<div class="rca-head">'
                    f'<span class="badge">{row["date"]}  H{row["hour"]}</span>'
                    f'<strong>{row["msc"]}</strong>'
                    f'<span style="color:{TEXT_DIM};font-size:0.78rem;margin-left:auto">'
                    f'Top CCs: {row["top_ccs"]}</span>'
                    f'</div>'
                    f'<div class="rca-body">{row["summary"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Download AI summaries
            ai_df = pd.DataFrame(st.session_state["rca_summaries"])
            ai_xlsx = io.BytesIO()
            ai_df.to_excel(ai_xlsx, index=False, engine="openpyxl")
            st.download_button(
                "Download AI RCA Summaries (.xlsx)",
                data=ai_xlsx.getvalue(),
                file_name=f"ai_rca_{kpi_label.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ── Dip group details (always shown) ─────────────────────────────────────────

if dip_groups:
    st.markdown("**Dip Group Details**")
    for group in dip_groups:
        with st.expander(f"{group['date']}  Hour {group['hour']} — {group['msc']}"):
            for cc in group["cc_details"]:
                dot_cls = "up" if cc["share_delta"] > 0 else "dn"
                st.markdown(
                    f'<div class="cc-pill">'
                    f'<span class="dot {dot_cls}"></span>'
                    f'<strong>{cc["cc_id"]}</strong> — {cc["definition"]} — '
                    f'Delta: {cc["share_delta"]:+.6f}  (Rank {cc["impact_rank"]})'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — FULL TABLE & EXPORT
# ══════════════════════════════════════════════════════════════════════════════

step_header("Complete RCA Data & Export", 6)

display_rca = rca_df.copy()
display_rca["Date"] = pd.to_datetime(display_rca["Date"]).dt.strftime("%Y-%m-%d")
if "Ref_Date" in display_rca.columns:
    display_rca["Ref_Date"] = pd.to_datetime(display_rca["Ref_Date"]).dt.strftime("%Y-%m-%d")

st.dataframe(
    display_rca, use_container_width=True, hide_index=True,
    column_config={
        "Share_Delta": st.column_config.NumberColumn(format="%.6f"),
        "Share_Current": st.column_config.NumberColumn(format="%.6f"),
        "Share_Reference": st.column_config.NumberColumn(format="%.6f"),
    },
)

c1, c2, c3 = st.columns(3)
with c1:
    buf = io.StringIO()
    rca_df.to_csv(buf, index=False)
    st.download_button(
        "Download CSV", data=buf.getvalue(),
        file_name=f"rca_{kpi_label.lower()}.csv", mime="text/csv",
    )
with c2:
    buf = io.BytesIO()
    rca_df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "Download Excel", data=buf.getvalue(),
        file_name=f"rca_{kpi_label.lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c3:
    if not skip_df.empty:
        buf = io.StringIO()
        skip_df.to_csv(buf, index=False)
        st.download_button(
            "Download Skipped", data=buf.getvalue(),
            file_name=f"skipped_{kpi_label.lower()}.csv", mime="text/csv",
        )

if not skip_df.empty:
    with st.expander(f"Skipped Cases ({len(skip_df)})"):
        st.dataframe(skip_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;padding:2rem 0 1rem;color:{TEXT_DIM};font-size:0.75rem">'
    f'KPI RCA Engine &nbsp;&bull;&nbsp; Telecom Analytics Dashboard'
    f'</div>',
    unsafe_allow_html=True,
)
