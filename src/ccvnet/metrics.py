from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


METRIC_COLS = ["n", "rmse", "mae", "mape", "pearson_r"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "pearson_r": np.nan}

    yt = y_true[valid]
    yp = y_pred[valid]
    nonzero = np.abs(yt) > 1e-8
    mape = np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100 if nonzero.any() else np.nan
    pearson_r = (
        np.corrcoef(yt, yp)[0, 1]
        if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0
        else np.nan
    )

    return {
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": float(mape),
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
    }


def summarize_metric_repeats(metric_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary_cols = [*group_cols, *METRIC_COLS, *[f"{metric}_std" for metric in METRIC_COLS]]
    if metric_df.empty:
        return pd.DataFrame(columns=summary_cols)

    summary = metric_df.groupby(group_cols, dropna=False)[METRIC_COLS].agg(["mean", "std"]).reset_index()
    flattened_columns = []
    for col in summary.columns:
        if isinstance(col, tuple):
            left, right = col
            flattened_columns.append(left if right == "" else f"{left}_{right}")
        else:
            flattened_columns.append(col)
    summary.columns = flattened_columns

    rename_map = {f"{metric}_mean": metric for metric in METRIC_COLS}
    summary = summary.rename(columns=rename_map)
    for metric in METRIC_COLS:
        std_col = f"{metric}_std"
        if std_col not in summary.columns:
            summary[std_col] = 0.0
    return summary[summary_cols]

