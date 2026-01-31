from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int, mode: str = "trailing") -> np.ndarray:
    """
    Moving average with length preserved.

    Parameters
    ----------
    x : np.ndarray
        Input 1D signal.
    window : int
        Window size (>= 1).
    mode : str
        "trailing" (causal) or "centered" (non-causal).

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as x.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if window <= 1 or n == 0:
        return x.copy()
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > n:
        # Degenerate case: window larger than signal length
        # trailing: use mean up to each point; centered: use global mean
        if mode == "trailing":
            c = np.cumsum(x)
            return c / np.arange(1, n + 1)
        return np.full_like(x, np.mean(x))

    w = int(window)

    if mode == "trailing":
        # Pad the beginning with the first sample so the output length matches input
        x_pad = np.pad(x, (w - 1, 0), mode="edge")
        kernel = np.ones(w, dtype=float) / w
        y = np.convolve(x_pad, kernel, mode="valid")
        return y

    if mode == "centered":
        # Symmetric padding; window centered around each point
        left = (w - 1) // 2
        right = w - 1 - left
        x_pad = np.pad(x, (left, right), mode="edge")
        kernel = np.ones(w, dtype=float) / w
        y = np.convolve(x_pad, kernel, mode="valid")
        return y

    raise ValueError("mode must be 'trailing' or 'centered'")


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
