from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.processing import fft_magnitude


RowTuple = Tuple[str, np.ndarray, np.ndarray, np.ndarray]  # (label, t, y, y_ma)


# =========================================================
# Matplotlib plots (downloadable PNG)
# =========================================================
def make_3x3_figure(rows: List[RowTuple], bins: int = 30) -> plt.Figure:
    """
    3x3:
      Col 1: time series (raw + MA)
      Col 2: histogram (raw)
      Col 3: FFT magnitude (raw)
    """
    if len(rows) != 3:
        raise ValueError("Exactly 3 rows required")

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)

    for i, (label, t, y, y_ma) in enumerate(rows):
        # Time series
        ax = axes[i, 0]
        ax.plot(t, y, label="raw")
        ax.plot(t, y_ma, label="MA")
        ax.set_title(f"{label} vs Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.legend(loc="best")

        # Histogram
        ax = axes[i, 1]
        ax.hist(y, bins=bins)
        ax.set_title(f"{label} Histogram")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

        # FFT
        ax = axes[i, 2]
        f, m = fft_magnitude(t, y)
        if f.size:
            ax.plot(f, m)
        ax.set_title(f"{label} FFT Magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")

    return fig


def make_frequency_polygon_1x3(rows: List[RowTuple], bins: int = 30) -> plt.Figure:
    """
    1x3 frequency polygons (histogram as a line), one subplot per selected signal.
    """
    if len(rows) != 3:
        raise ValueError("Exactly 3 rows required")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for i, (label, _t, y, _y_ma) in enumerate(rows):
        counts, edges = np.histogram(y, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[i].plot(centers, counts, marker="o")
        axes[i].set_title(f"{label} Frequency Polygon")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Count")

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
    rows: List[RowTuple],
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
    if len(rows) != 3:
        raise ValueError("Exactly 3 rows required")

    (lx, tx, yx, yx_ma) = rows[0]
    (ly, ty, yy, yy_ma) = rows[1]
    (lz, tz, yz, yz_ma) = rows[2]

    x = yx_ma if use_ma else yx
    y = yy_ma if use_ma else yy
    z = yz_ma if use_ma else yz

    n = min(len(x), len(y), len(z), len(tx), len(ty), len(tz))
    if n < 5:
        raise ValueError("Not enough points for 3D plot")

    x, y, z = x[:n], y[:n], z[:n]
    tx, ty, tz = tx[:n], ty[:n], tz[:n]

    # Downsample for performance
    if max_points and n > max_points:
        idx = np.linspace(0, n - 1, int(max_points)).astype(int)
        x, y, z = x[idx], y[idx], z[idx]
        tx, ty, tz = tx[idx], ty[idx], tz[idx]
        n = int(max_points)

    # Choose color driver
    if color_by == "Row 1 time":
        c = tx; ctitle = "Row 1 time"
    elif color_by == "Row 2 time":
        c = ty; ctitle = "Row 2 time"
    elif color_by == "Row 3 time":
        c = tz; ctitle = "Row 3 time"
    elif color_by == "Row 1 value":
        c = x; ctitle = "Row 1 value"
    elif color_by == "Row 2 value":
        c = y; ctitle = "Row 2 value"
    elif color_by == "Row 3 value":
        c = z; ctitle = "Row 3 value"
    else:
        c = np.arange(n); ctitle = "Sample index"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
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
            xaxis_title=lx,
            yaxis_title=ly,
            zaxis_title=lz,
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig
