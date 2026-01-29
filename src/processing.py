from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average with edge padding (prevents the MA curve from dropping at the
    beginning/end due to convolution boundary effects).

    - Returns same-length output as x
    - If window <= 1, returns x unchanged
    """
    if window is None or int(window) <= 1:
        return x

    w = int(window)
    kernel = np.ones(w, dtype=float) / float(w)

    # Pad using edge values to avoid boundary drop
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")

    # 'valid' on padded signal returns same length as original
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def fft_magnitude(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT magnitude using a uniform-sampling approximation.
    dt is estimated as median(diff(time)).
    Returns (freq_hz, magnitude).

    Notes:
    - Detrends by subtracting mean.
    - Returns empty arrays if sampling info is invalid.
    """
    if time.size < 4 or signal.size < 4:
        return np.array([]), np.array([])

    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return np.array([]), np.array([])

    dt_med = float(np.median(dt))
    if dt_med <= 0:
        return np.array([]), np.array([])

    x = signal - float(np.mean(signal))
    n = x.size

    freq = np.fft.rfftfreq(n, d=dt_med)
    mag = np.abs(np.fft.rfft(x)) / n
    return freq, mag
