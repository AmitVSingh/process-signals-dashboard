from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.processing import fft_magnitude


RowTuple = Tuple[str, np.ndarray, np.ndarray, np.ndarray]  # (label, t, y, y_ma)

# =========================================================
# Matplotlib plots
# =========================================================
def make_3x3_figure(rows: List[RowTuple], bins: int = 30) -> plt.Figure:
    if len(rows) != 3:
        raise ValueError("Exactly 3 rows required")

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)

    for i, (label, t, y, y_ma) in enumerate(rows):
        # Time series
        axes[i, 0].plot(t, y, label="raw")
        axes[i, 0].plot(t, y_ma, label="MA")
        axes[i, 0].set_title(f"{label} vs Time")
        axes[i, 0].legend()

        # Histogram
        axes[i, 1].hist(y, bins=bins)
        axes[i, 1].set_title(f"{label} Histogram")

        # FFT
        f, m = fft_magnitude(t, y)
        if f.size:
            axes[i, 2].plot(f, m)
        axes[i, 2].set_title(f"{label} FFT")

    return fig


def make_frequency_polygon_1x3(rows: List[RowTuple], bins: int = 30) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for i, (label, _t, y, _y_ma) in enumerate(rows):
        counts, edges = np.histogram(y, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[i].plot(centers, counts, marker="o")
        axes[i].set_title(f"{label} Frequency Polygon")

    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================================================
# Plotly interactive 3D
# =========================================================
def make_plotly_3d_signals(
    rows: List[RowTuple],
    use_ma: bool = False,
    max_points: int = 5000,
    marker_size: int = 3,
    color_by: str = "Sample index",
) -> go.Figure:
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

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        x, y, z = x[idx], y[idx], z[idx]
        tx, ty, tz = tx[idx], ty[idx], tz[idx]
        n = max_points

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


def plotly_fig_to_png_bytes(fig: go.Figure, scale: float = 2.0) -> bytes:
    return fig.to_image(format="png", scale=scale, engine="kaleido")
