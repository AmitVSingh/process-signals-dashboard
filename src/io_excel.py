from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

TIME_PREFIX = "Time - "


@dataclass(frozen=True)
class Signal:
    name: str
    time_col: str
    value_col: str


def load_excel(file_like, sheet_name: int | str = 0) -> pd.DataFrame:
    """
    Loads an Excel file into a DataFrame.
    `file_like` can be:
      - a filesystem path (str/Path)
      - a Streamlit UploadedFile object
    """
    df = pd.read_excel(file_like, sheet_name=sheet_name, engine="openpyxl")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def discover_signals(df: pd.DataFrame) -> List[Signal]:
    """
    Discovers signals using:
      time column:  'Time - <name>'
      value column: '<anything> - <name>' (excluding the time column itself)
    """
    cols = [c for c in df.columns if isinstance(c, str)]
    time_cols = [c for c in cols if c.startswith(TIME_PREFIX)]

    signals: List[Signal] = []
    for tcol in time_cols:
        name = tcol[len(TIME_PREFIX):].strip()
        if not name:
            continue

        suffix = f" - {name}"
        candidates = [c for c in cols if c != tcol and c.endswith(suffix)]

        if not candidates:
            logging.warning("No value column found for signal '%s' (time col: '%s')", name, tcol)
            continue

        # If multiple candidates exist, pick the first deterministic one.
        vcol = candidates[0]
        signals.append(Signal(name=name, time_col=tcol, value_col=vcol))

    return signals


def extract_signal(df: pd.DataFrame, signal: Signal) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (time, value) as numeric Series, NaNs removed & aligned.
    """
    t = pd.to_numeric(df[signal.time_col], errors="coerce")
    y = pd.to_numeric(df[signal.value_col], errors="coerce")

    mask = t.notna() & y.notna()
    t = t[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    return t, y
