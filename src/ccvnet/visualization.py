from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def set_paper_style() -> None:
    """Apply shared matplotlib style for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def load_result_table(path: str | Path) -> pd.DataFrame:
    """Load a generated result CSV for visualization notebooks."""
    return pd.read_csv(path)


