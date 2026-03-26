import pandas as pd
import numpy as np


def get_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_cells": df.shape[0] * df.shape[1],
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).reset_index().rename(columns={"index": "Column"})


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    stats = numeric_df.describe().T
    stats["skewness"] = numeric_df.skew().round(3)
    stats["kurtosis"] = numeric_df.kurtosis().round(3)
    return stats.round(3).reset_index().rename(columns={"index": "Column"})


def get_categorical_summary(df: pd.DataFrame) -> dict:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary = {}
    for col in cat_cols:
        summary[col] = {
            "unique_values": int(df[col].nunique()),
            "top_5": df[col].value_counts().head(5).to_dict(),
            "null_count": int(df[col].isnull().sum()),
        }
    return summary


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)


def detect_outliers(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        outliers[col] = {
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(df) * 100, 2),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
        }
    return outliers


def build_summary_for_ai(df: pd.DataFrame) -> dict:
    """Packages all stats into a single dict to send to AI narrator."""
    overview = get_overview(df)
    stats = get_descriptive_stats(df)
    missing = get_missing_info(df)
    outliers = detect_outliers(df)
    cat_summary = get_categorical_summary(df)

    return {
        "overview": overview,
        "descriptive_stats": stats.to_dict(orient="records") if not stats.empty else [],
        "missing_values": missing[missing["Missing Count"] > 0].to_dict(orient="records"),
        "outliers": {k: v for k, v in outliers.items() if v["outlier_count"] > 0},
        "categorical_columns": cat_summary,
    }
