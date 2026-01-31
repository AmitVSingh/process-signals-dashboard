from __future__ import annotations

import pandas as pd
import streamlit as st

from src.io_excel import load_excel


@st.cache_data(show_spinner="Reading Excel file...")
def load_excel_cached(uploaded_file) -> pd.DataFrame:
    """
    Cached Excel loader.

    We cache only the DataFrame because it is serializable and large enough
    to benefit from caching. Signal discovery/extraction stays uncached
    for simplicity and reliability across environments.
    """
    return load_excel(uploaded_file)
