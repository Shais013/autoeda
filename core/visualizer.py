import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

PLOT_THEME = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Vivid


def plot_missing_heatmap(df: pd.DataFrame):
    missing_matrix = df.isnull().astype(int)
    fig = px.imshow(
        missing_matrix,
        color_continuous_scale=["#e8f5e9", "#e53935"],
        title="Missing Value Heatmap",
        labels={"color": "Missing"},
        template=PLOT_THEME,
    )
    fig.update_layout(height=400, coloraxis_showscale=True)
    return fig


def plot_missing_bar(df: pd.DataFrame):
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        return None
    fig = px.bar(
        x=missing_pct.index,
        y=missing_pct.values,
        labels={"x": "Column", "y": "Missing (%)"},
        title="Missing Values per Column (%)",
        color=missing_pct.values,
        color_continuous_scale="Reds",
        template=PLOT_THEME,
    )
    fig.update_layout(showlegend=False, height=350)
    return fig


def plot_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    figs = {}
    for col in numeric_cols:
        data = df[col].dropna()
        fig = px.histogram(
            data,
            x=col,
            nbins=30,
            marginal="box",
            title=f"Distribution: {col}",
            template=PLOT_THEME,
            color_discrete_sequence=["#5c6bc0"],
        )
        fig.update_layout(height=350)
        figs[col] = fig
    return figs


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        template=PLOT_THEME,
    )
    fig.update_layout(height=500)
    return fig


def plot_categorical_bars(df: pd.DataFrame, top_n: int = 10):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    figs = {}
    for col in cat_cols:
        counts = df[col].value_counts().head(top_n).reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            title=f"Top {top_n} values: {col}",
            template=PLOT_THEME,
            color="Count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=350, showlegend=False)
        figs[col] = fig
    return figs


def plot_outlier_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    df_melted = df[numeric_cols].melt(var_name="Feature", value_name="Value")
    fig = px.box(
        df_melted,
        x="Feature",
        y="Value",
        color="Feature",
        title="Outlier Detection (IQR Boxplots)",
        template=PLOT_THEME,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(height=420, showlegend=False)
    return fig
