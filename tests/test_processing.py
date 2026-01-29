import numpy as np

from src.processing import moving_average, fft_magnitude


def test_moving_average_length_preserved():
    x = np.arange(100, dtype=float)
    y = moving_average(x, window=20)  # default kind
    assert len(y) == len(x)


def test_moving_average_no_edge_drop_like_zero_padding():
    # Constant signal should remain constant after MA
    x = np.ones(200, dtype=float) * 123.4
    y = moving_average(x, window=20)
    assert np.allclose(y, x, atol=1e-9)


def test_moving_average_window_1_returns_input():
    x = np.random.randn(50)
    y = moving_average(x, window=1)
    assert np.allclose(y, x)


def test_fft_magnitude_basic_shapes():
    # 1 second of a 10 Hz sine sampled at 1000 Hz
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    f0 = 10.0
    x = np.sin(2 * np.pi * f0 * t)

    f, m = fft_magnitude(t, x)
    assert f.ndim == 1 and m.ndim == 1
    assert len(f) == len(m)
    assert len(f) > 0
    assert f[0] == 0.0
    assert np.all(m >= 0.0)
