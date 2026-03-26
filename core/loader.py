import pandas as pd
import streamlit as st


def load_csv(uploaded_file) -> pd.DataFrame | None:
    """Load and validate a CSV file uploaded via Streamlit."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of warnings about the dataframe.
    Empty list means no issues found.
    """
    warnings = []

    if df.empty:
        warnings.append("Dataset is empty.")
        return warnings

    if df.shape[0] < 10:
        warnings.append(f"Very small dataset: only {df.shape[0]} rows.")

    if df.shape[1] < 2:
        warnings.append("Only 1 column found. Analysis may be limited.")

    missing_pct = df.isnull().mean().max() * 100
    if missing_pct > 50:
        warnings.append(f"Some columns have more than 50% missing values ({missing_pct:.1f}% max).")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate rows detected.")

    return warnings
