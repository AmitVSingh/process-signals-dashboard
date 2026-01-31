import numpy as np

from src.models import SeriesData
from src.plotting import (
    make_3x3_figure,
    make_frequency_polygon_1x3,
    fig_to_png_bytes,
    make_plotly_3d_signals,
)


def _dummy_series(name: str, n: int = 500) -> SeriesData:
    # Uniform time base
    t = np.linspace(0.0, 5.0, n)

    # A simple, non-degenerate signal
    y = np.sin(2 * np.pi * 2.0 * t) + 0.1 * np.cos(2 * np.pi * 7.0 * t)

    # Pretend MA (for plotting tests we just need shape-match)
    y_ma = y.copy()

    return SeriesData(name=name, t=t, y=y, y_ma=y_ma)


def test_make_3x3_figure_returns_matplotlib_figure():
    series = [_dummy_series("A"), _dummy_series("B"), _dummy_series("C")]
    fig = make_3x3_figure(series, bins=25)
    assert fig is not None
    # Basic Matplotlib Figure attribute
    assert hasattr(fig, "savefig")


def test_fig_to_png_bytes_returns_nonempty_bytes():
    series = [_dummy_series("A"), _dummy_series("B"), _dummy_series("C")]
    fig = make_3x3_figure(series, bins=20)
    png = fig_to_png_bytes(fig)
    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 1000  # should be a real PNG payload


def test_make_frequency_polygon_1x3_returns_matplotlib_figure():
    series = [_dummy_series("A"), _dummy_series("B"), _dummy_series("C")]
    fig = make_frequency_polygon_1x3(series, bins=30)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_make_plotly_3d_signals_returns_plotly_figure():
    series = [_dummy_series("A"), _dummy_series("B"), _dummy_series("C")]
    fig = make_plotly_3d_signals(
        series,
        use_ma=False,
        max_points=1000,
        marker_size=2,
        color_by="Sample index",
    )
    assert fig is not None
    # Plotly Figure has to_dict()
    assert hasattr(fig, "to_dict")


def test_make_plotly_3d_signals_requires_three_series():
    series = [_dummy_series("A"), _dummy_series("B")]
    try:
        make_plotly_3d_signals(series)
        assert False, "Expected ValueError for non-3 series input"
    except ValueError:
        assert True
