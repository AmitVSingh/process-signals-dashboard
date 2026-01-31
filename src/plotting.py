from __future__ import annotations

from io import BytesIO
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.models import SeriesData
from src.processing import fft_magnitude


# =========================================================
# Matplotlib plots (downloadable PNG)
# =========================================================
def make_3x3_figure(series: Sequence[SeriesData], bins: int = 30) -> plt.Figure:
    """
    3x3:
      Col 1: time series (raw + MA)
      Col 2: histogram (raw)
      Col 3: FFT magnitude (raw)
    """
    if len(series) != 3:
        raise ValueError("Exactly 3 rows required")

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True, squeeze=False)

    for i, s in enumerate(series):
        # Time series
        ax = axes[i, 0]
        ax.plot(s.t, s.y, label="raw")
        ax.plot(s.t, s.y_ma, label="MA")
        ax.set_title(f"{s.name} vs Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(s.name)
        ax.legend(loc="best")

        # Histogram
        ax = axes[i, 1]
        ax.hist(s.y, bins=bins)
        ax.set_title(f"{s.name} Histogram")
        ax.set_xlabel(s.name)
        ax.set_ylabel("Count")

        # FFT
        ax = axes[i, 2]
        f, m = fft_magnitude(s.t, s.y)
        if f.size:
            ax.plot(f, m)
        ax.set_title(f"{s.name} FFT Magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")

    return fig


def make_frequency_polygon_1x3(series: Sequence[SeriesData], bins: int = 30) -> plt.Figure:
    """
    1x3 frequency polygons (histogram as a line), one subplot per selected signal.
    """
    if len(series) != 3:
        raise ValueError("Exactly 3 rows required")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True, squeeze=False)

    for i, s in enumerate(series):
        ax = axes[0, i]  # <-- key fix
        counts, edges = np.histogram(s.y, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, marker="o")
        ax.set_title(f"{s.name} Frequency Polygon")
        ax.set_xlabel(s.name)
        ax.set_ylabel("Count")

    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    """
    Convert Matplotlib figure to PNG bytes (Streamlit display + download).
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================================================
# Plotly interactive 3D (NO download/export)
# =========================================================
def make_plotly_3d_signals(
    series: Sequence[SeriesData],
    use_ma: bool = False,
    max_points: int = 5000,
    marker_size: int = 3,
    color_by: str = "Sample index",
) -> go.Figure:
    """
    Interactive Plotly 3D scatter:
      X = Row1 values, Y = Row2 values, Z = Row3 values
    Color can be chosen by:
      - Sample index
      - Row 1/2/3 time
      - Row 1/2/3 value (raw or MA depending on use_ma)
    Alignment: by index (trim to min length, optional downsample).
    """
    if len(series) != 3:
        raise ValueError("Exactly 3 rows required")

    s1, s2, s3 = series[0], series[1], series[2]

    x = s1.y_ma if use_ma else s1.y
    y = s2.y_ma if use_ma else s2.y
    z = s3.y_ma if use_ma else s3.y

    t1, t2, t3 = s1.t, s2.t, s3.t

    n = min(len(x), len(y), len(z), len(t1), len(t2), len(t3))
    if n < 5:
        raise ValueError("Not enough points for 3D plot")

    x, y, z = x[:n], y[:n], z[:n]
    t1, t2, t3 = t1[:n], t2[:n], t3[:n]

    # Downsample for performance
    if max_points and n > max_points:
        idx = np.linspace(0, n - 1, int(max_points)).astype(int)
        x, y, z = x[idx], y[idx], z[idx]
        t1, t2, t3 = t1[idx], t2[idx], t3[idx]
        n = int(max_points)

    # Choose color driver
    if color_by == "Row 1 time":
        c = t1
        ctitle = "Row 1 time"
    elif color_by == "Row 2 time":
        c = t2
        ctitle = "Row 2 time"
    elif color_by == "Row 3 time":
        c = t3
        ctitle = "Row 3 time"
    elif color_by == "Row 1 value":
        c = x
        ctitle = "Row 1 value"
    elif color_by == "Row 2 value":
        c = y
        ctitle = "Row 2 value"
    elif color_by == "Row 3 value":
        c = z
        ctitle = "Row 3 value"
    else:
        c = np.arange(n)
        ctitle = "Sample index"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=c,
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar=dict(title=ctitle),
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"3D Signal Relationship ({'MA' if use_ma else 'Raw'})",
        scene=dict(
            xaxis_title=s1.name,
            yaxis_title=s2.name,
            zaxis_title=s3.name,
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig
