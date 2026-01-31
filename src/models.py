from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SeriesData:
    """A single signal series prepared for plotting."""
    name: str
    t: np.ndarray
    y: np.ndarray
    y_ma: np.ndarray
