from __future__ import annotations

from pathlib import Path
import sys
import logging

import streamlit as st

# Make sure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_excel import load_excel, discover_signals, extract_signal
from src.processing import moving_average
from src.plotting import (
    make_3x3_figure,
    make_frequency_polygon_1x3,
    make_plotly_3d_signals,
    fig_to_png_bytes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

st.set_page_config(page_title="Process Signals Dashboard", layout="wide")
st.title("Process Signals Dashboard")


# =========================================================
# Cached helpers (define once, near top)
# =========================================================
@st.cache_data(show_spinner="Reading Excel file...")
def load_excel_cached(uploaded_file):
    return load_excel(uploaded_file)


@st.cache_data(show_spinner=False)
def discover_signals_cached(df):
    return discover_signals(df)


@st.cache_data(show_spinner=False)
def extract_signal_cached(df, sig):
    return extract_signal(df, sig)


# =========================================================
# Sidebar Controls
# =========================================================
with st.sidebar:
    st.header("Controls")

    ma_window = st.number_input("MA window", min_value=1, max_value=5000, value=20, step=1)
    hist_bins = st.number_input("Histogram bins", min_value=5, max_value=200, value=30, step=1)

    st.divider()
    show_freq_poly = st.checkbox("Show frequency polygon (1×3)", value=True)

    st.divider()
    show_3d = st.checkbox("Show interactive 3D (Plotly)", value=True)
    use_ma_3d = st.checkbox("Use MA for 3D", value=False)
    color_by = st.selectbox(
        "3D color by",
        [
            "Sample index",
            "Row 1 time",
            "Row 2 time",
            "Row 3 time",
            "Row 1 value",
            "Row 2 value",
            "Row 3 value",
        ],
        index=0,
    )
    max_points_3d = st.number_input("Max 3D points", min_value=200, max_value=50000, value=5000, step=500)
    marker_size = st.slider("3D marker size", min_value=1, max_value=10, value=3, step=1)

# =========================================================
# Upload Excel
# =========================================================
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

# =========================================================
# Load + detect signals (cached + safe)
# =========================================================
try:
    df = load_excel_cached(uploaded)
except Exception as exc:
    st.error(f"Failed to read Excel file: {exc}")
    st.stop()

signals = discover_signals_cached(df)
if not signals:
    st.error(
        "No signals found. Expected columns like:\n"
        "- `Time - <signal name>`\n"
        "- `<anything> - <signal name>`"
    )
    st.stop()

names = [s.name for s in signals]
by_name = {s.name: s for s in signals}

st.subheader("Detected signals")
st.write(names)

# =========================================================
# Signal selection (3 rows)
# =========================================================
with st.sidebar:
    st.header("Select 3 signals")
    s1 = st.selectbox("Row 1", names, index=0)
    s2 = st.selectbox("Row 2", names, index=1 if len(names) > 1 else 0)
    s3 = st.selectbox("Row 3", names, index=2 if len(names) > 2 else 0)

selected = [s1, s2, s3]

# Build rows: (label, t, y, y_ma)
rows = []
for name in selected:
    sig = by_name[name]

    try:
        t_s, y_s = extract_signal_cached(df, sig)
    except Exception as exc:
        st.error(f"Failed to extract signal '{name}': {exc}")
        st.stop()

    if len(t_s) < 4:
        st.error(f"Signal '{name}' has too few valid samples.")
        st.stop()

    t = t_s.to_numpy(dtype=float)
    y = y_s.to_numpy(dtype=float)
    y_ma = moving_average(y, int(ma_window))

    rows.append((name, t, y, y_ma))

# =========================================================
# 3x3 plot (Matplotlib) + download
# =========================================================
st.subheader("3×3 Analysis Plot (Time + Histogram + FFT)")
fig_3x3 = make_3x3_figure(rows, bins=int(hist_bins))
png_3x3 = fig_to_png_bytes(fig_3x3)

st.image(png_3x3, use_container_width=True)
st.download_button("Download 3×3 plot PNG", png_3x3, "plot_3x3.png", "image/png")

# =========================================================
# Frequency polygon (Matplotlib) + download
# =========================================================
if show_freq_poly:
    st.subheader("1×3 Frequency Polygon")
    fig_poly = make_frequency_polygon_1x3(rows, bins=int(hist_bins))
    png_poly = fig_to_png_bytes(fig_poly)

    st.image(png_poly, use_container_width=True)
    st.download_button("Download frequency polygon PNG", png_poly, "freq_polygon.png", "image/png")

# =========================================================
# Interactive 3D Plotly (NO download)
# =========================================================
if show_3d:
    st.subheader("Interactive 3D Plot (Plotly) — X=Row1, Y=Row2, Z=Row3")
    try:
        fig3d = make_plotly_3d_signals(
            rows,
            use_ma=bool(use_ma_3d),
            max_points=int(max_points_3d),
            marker_size=int(marker_size),
            color_by=str(color_by),
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Note: 3D plot is interactive in the browser. (To download Plotly plots click camera emoji.)")
    except Exception as exc:
        st.error(f"Could not generate 3D plot: {exc}")
