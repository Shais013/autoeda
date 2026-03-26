import streamlit as st
import pandas as pd
from core.loader import load_csv, validate_dataframe
from core.eda_engine import (
    get_overview,
    get_missing_info,
    get_descriptive_stats,
    get_categorical_summary,
    get_correlation_matrix,
    detect_outliers,
    build_summary_for_ai,
)
from core.visualizer import (
    plot_missing_heatmap,
    plot_missing_bar,
    plot_distributions,
    plot_correlation_heatmap,
    plot_categorical_bars,
    plot_outlier_boxplots,
)
from core.ai_narrator import generate_insights

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoEDA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AutoEDA")
    st.caption("Automated Exploratory Data Analysis")
    st.divider()
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    st.divider()
    st.markdown("**Settings**")
    top_n_categories = st.slider("Top N categories to show", 5, 20, 10)
    show_ai_insights = st.toggle("AI Insights (requires API key)", value=True)

# ── Main content ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.title("Welcome to AutoEDA 👋")
    st.markdown(
        """
        Upload a CSV file from the sidebar to get started.

        **What you'll get:**
        - Dataset overview (shape, types, memory)
        - Missing value analysis
        - Descriptive statistics
        - Distribution plots
        - Correlation heatmap
        - Outlier detection
        - AI-powered plain English insights
        """
    )
    st.info("📂 Upload a CSV from the left sidebar to begin.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_csv(uploaded_file)
if df is None:
    st.stop()

warnings = validate_dataframe(df)
for w in warnings:
    st.warning(w)

st.title(f"📊 Analysis: `{uploaded_file.name}`")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_missing, tab_stats, tab_viz, tab_corr, tab_ai = st.tabs([
    "📋 Overview",
    "❓ Missing Values",
    "📈 Statistics",
    "📊 Distributions",
    "🔗 Correlations",
    "🤖 AI Insights",
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    overview = get_overview(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{overview['rows']:,}")
    col2.metric("Columns", overview["columns"])
    col3.metric("Memory", f"{overview['memory_usage_kb']} KB")
    col4.metric("Duplicates", overview["duplicate_rows"])

    st.subheader("Column Types")
    dtypes_df = pd.DataFrame(
        {"Column": list(overview["dtypes"].keys()), "Type": list(overview["dtypes"].values())}
    )
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ── Tab 2: Missing Values ─────────────────────────────────────────────────────
with tab_missing:
    missing_df = get_missing_info(df)
    has_missing = missing_df["Missing Count"].sum() > 0

    if not has_missing:
        st.success("✅ No missing values found in this dataset!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_missing_bar(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_missing_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Missing Values Table")
        st.dataframe(
            missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 3: Descriptive Stats ──────────────────────────────────────────────────
with tab_stats:
    stats_df = get_descriptive_stats(df)
    if stats_df.empty:
        st.info("No numeric columns found for descriptive statistics.")
    else:
        st.subheader("Numeric Columns")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    cat_summary = get_categorical_summary(df)
    if cat_summary:
        st.subheader("Categorical Columns")
        for col, info in cat_summary.items():
            with st.expander(f"**{col}** — {info['unique_values']} unique values"):
                st.write(f"**Null count:** {info['null_count']}")
                st.write("**Top values:**")
                st.dataframe(
                    pd.DataFrame.from_dict(info["top_5"], orient="index", columns=["Count"]),
                    use_container_width=True,
                )

# ── Tab 4: Distributions ──────────────────────────────────────────────────────
with tab_viz:
    dist_figs = plot_distributions(df)
    cat_figs = plot_categorical_bars(df, top_n=top_n_categories)
    outlier_fig = plot_outlier_boxplots(df)

    if dist_figs:
        st.subheader("Numeric Distributions")
        cols = st.columns(2)
        for i, (col_name, fig) in enumerate(dist_figs.items()):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

    if cat_figs:
        st.subheader("Categorical Distributions")
        cols = st.columns(2)
        for i, (col_name, fig) in enumerate(cat_figs.items()):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

    if outlier_fig:
        st.subheader("Outlier Overview")
        st.plotly_chart(outlier_fig, use_container_width=True)

    outliers = detect_outliers(df)
    cols_with_outliers = {k: v for k, v in outliers.items() if v["outlier_count"] > 0}
    if cols_with_outliers:
        st.subheader("Outlier Summary")
        outlier_df = pd.DataFrame(cols_with_outliers).T.reset_index()
        outlier_df.columns = ["Column", "Outlier Count", "Outlier %", "Lower Bound", "Upper Bound"]
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)

# ── Tab 5: Correlations ───────────────────────────────────────────────────────
with tab_corr:
    corr_fig = plot_correlation_heatmap(df)
    if corr_fig is None:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        st.plotly_chart(corr_fig, use_container_width=True)

        corr_matrix = get_correlation_matrix(df)
        st.subheader("Strong Correlations (|r| > 0.7)")
        strong = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    strong.append({
                        "Column A": corr_matrix.columns[i],
                        "Column B": corr_matrix.columns[j],
                        "Correlation": round(val, 3),
                    })
        if strong:
            st.dataframe(pd.DataFrame(strong), use_container_width=True, hide_index=True)
        else:
            st.info("No strong correlations (|r| > 0.7) found.")

# ── Tab 6: AI Insights ────────────────────────────────────────────────────────
with tab_ai:
    st.subheader("🤖 AI-Generated Insights")
    if not show_ai_insights:
        st.info("Enable AI Insights in the sidebar to use this feature.")
    else:
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing your data..."):
                try:
                    summary = build_summary_for_ai(df)
                    insights = generate_insights(summary)
                    st.session_state["ai_insights"] = insights
                except Exception as e:
                    st.error(f"AI generation failed: {e}")
                    st.info("Make sure your ANTHROPIC_API_KEY is set in .env or Streamlit Secrets.")

        if "ai_insights" in st.session_state:
            st.markdown(st.session_state["ai_insights"])
