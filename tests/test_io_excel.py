import pandas as pd

from src.io_excel import discover_signals, extract_signal


def test_discover_signals_from_headers():
    df = pd.DataFrame(
        {
            "Time - Measured Velocity": [0.0, 0.1, 0.2],
            "Spool Rev/Sec - Measured Velocity": [1.0, 2.0, 3.0],
            "Time - Measured Diameter": [0.0, 0.1, 0.2],
            "Diameter (mm) - Measured Diameter": [10.0, 10.1, 10.2],
        }
    )

    signals = discover_signals(df)
    names = sorted([s.name for s in signals])

    assert "Measured Velocity" in names
    assert "Measured Diameter" in names


def test_extract_signal_returns_aligned_numeric_series():
    df = pd.DataFrame(
        {
            "Time - Temperature": [0.0, 0.1, 0.2, 0.3],
            "Diameter (mm) - Temperature": ["20.0", "21.0", None, "22.0"],  # includes str + None
        }
    )

    sig = [s for s in discover_signals(df) if s.name == "Temperature"][0]
    t_s, y_s = extract_signal(df, sig)

    # Must be aligned and numeric and with NaNs removed
    assert len(t_s) == len(y_s)
    assert len(t_s) == 3  # one row removed due to None
    assert t_s.dtype.kind in ("i", "u", "f")  # numeric
    assert y_s.dtype.kind in ("i", "u", "f")  # numeric
