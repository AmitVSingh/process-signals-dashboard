from __future__ import annotations

from pathlib import Path
import sys
import logging

import streamlit as st

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
    plotly_fig_to_png_bytes,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Process Signals Dashboard", layout="wide")
st.title("Process Signals Dashboard")

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Controls")

    ma_window = st.number_input("MA window", 1, 5000, 20)
    hist_bins = st.number_input("Histogram bins", 5, 200, 30)

    show_freq_poly = st.checkbox("Show frequency polygon", True)

    st.divider()
    show_3d = st.checkbox("Show interactive 3D plot", True)
    use_ma_3d = st.checkbox("Use MA for 3D", False)

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
    )

    max_points_3d = st.number_input("Max 3D points", 200, 50000, 5000, step=500)
    marker_size = st.slider("3D marker size", 1, 10, 3)
    export_scale = st.slider("3D PNG scale", 1.0, 4.0, 2.0, 0.5)

# =========================================================
# Upload
# =========================================================
uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded is None:
    st.stop()

df = load_excel(uploaded)
signals = discover_signals(df)
names = [s.name for s in signals]
by_name = {s.name: s for s in signals}

# =========================================================
# Signal selection
# =========================================================
with st.sidebar:
    st.header("Select 3 signals")
    s1 = st.selectbox("Row 1", names, index=0)
    s2 = st.selectbox("Row 2", names, index=1)
    s3 = st.selectbox("Row 3", names, index=2)

rows = []
for name in [s1, s2, s3]:
    sig = by_name[name]
    t, y = extract_signal(df, sig)
    y_ma = moving_average(y.to_numpy(float), int(ma_window))
    rows.append((name, t.to_numpy(float), y.to_numpy(float), y_ma))

# =========================================================
# 3x3 plot
# =========================================================
st.subheader("3×3 Analysis Plot")
fig = make_3x3_figure(rows, bins=int(hist_bins))
png = fig_to_png_bytes(fig)
st.image(png, use_container_width=True)
st.download_button("Download 3×3 PNG", png, "plot_3x3.png", "image/png")

# =========================================================
# Frequency polygon
# =========================================================
if show_freq_poly:
    st.subheader("1×3 Frequency Polygon")
    figp = make_frequency_polygon_1x3(rows, bins=int(hist_bins))
    pngp = fig_to_png_bytes(figp)
    st.image(pngp, use_container_width=True)
    st.download_button("Download polygon PNG", pngp, "freq_polygon.png", "image/png")

# =========================================================
# Interactive 3D Plotly
# =========================================================
if show_3d:
    st.subheader("Interactive 3D Plot")
    fig3d = make_plotly_3d_signals(
        rows,
        use_ma=use_ma_3d,
        max_points=int(max_points_3d),
        marker_size=int(marker_size),
        color_by=color_by,
    )

    st.plotly_chart(fig3d, use_container_width=True)

    png3d = plotly_fig_to_png_bytes(fig3d, scale=float(export_scale))
    st.download_button("Download 3D PNG", png3d, "signals_3d.png", "image/png")
