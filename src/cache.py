from __future__ import annotations

import streamlit as st
import pandas as pd

from src.io_excel import load_excel, discover_signals, extract_signal, Signal


@st.cache_data(show_spinner="Reading Excel file...")
def load_excel_cached(uploaded_file) -> pd.DataFrame:
    """Load an uploaded Excel file into a DataFrame (cached by file content)."""
    return load_excel(uploaded_file)


@st.cache_data(show_spinner=False)
def discover_signals_cached(df: pd.DataFrame) -> list[Signal]:
    """Detect available signals from DataFrame headers (cached)."""
    return discover_signals(df)


@st.cache_data(show_spinner=False)
def extract_signal_cached(df: pd.DataFrame, sig: Signal):
    """Extract (time, value) series for a signal from df (cached)."""
    return extract_signal(df, sig)
