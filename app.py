# app.py
"""
Streamlit UI (full-feature, revised)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from lab_randomizer import (
    generate_valid_pairs,
    list_all_orders,
    random_run_order,
    export_csv_run_sheet,
    make_plate_map,
    compute_error_distribution,
    error_propagation_summary_uniform,
    read_csv_bytes,
    remove_outliers,
    posthoc_auto_select_and_run,
    format_pvalue,
    ensure_numeric_series,
    _HAS_STATSMODELS,
    _HAS_PINGOUIN,
    _HAS_SCIPOST
)

st.set_page_config(page_title="LabFlow", layout="wide")
st.title("LabFlow — Randomization, Error Propagation & Statistics")

# ===============================
# Sidebar
# ===============================
section = st.sidebar.radio(
    "Section",
    [
        "Randomizer (pairs & run order)",
        "Error propagation",
        "Upload & Preview CSV",
        "Distribution & Outliers",
        "Statistical Tests & Post-hoc",
        "Plate map & Export",
    ],
)

# ===============================
# Utilities
# ===============================
def to_long_format(df, group_cols, value_cols):
    long_df = df.copy()
    long_df["__row_id__"] = np.arange(len(long_df))

    melted = long_df.melt(
        id_vars=group_cols + ["__row_id__"],
        value_vars=value_cols,
        var_name="Value_Column",
        value_name="Value",
    )

    for g in group_cols:
        melted[g] = melted[g].astype(str)

    melted["Group"] = melted[group_cols].agg(" | ".join, axis=1)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

    return melted.dropna(subset=["Value"])

# ===============================
# Randomizer
# ===============================
if section == "Randomizer (pairs & run order)":
    st.header("Randomizer")

    samples_text = st.text_input("Samples (comma-separated)", "A, B")
    controls_text = st.text_input("Controls (comma-separated)", "C1, C2")

    samples = [s.strip() for s in samples_text.split(",") if s.strip()]
    controls = [c.strip() for c in controls_text.split(",") if c.strip()]

    pairs = generate_valid_pairs(samples, controls)

    st.subheader("Valid pairs")
    st.write(pairs if pairs else "No pairs generated.")

    if st.button("Pick random run order"):
        st.session_state["order"] = random_run_order(pairs)

    order = st.session_state.get("order", [])
    if order:
        st.table(pd.DataFrame({"Position": range(1, len(order) + 1), "Pair": order}))
        st.download_button(
            "Download run sheet",
            export_csv_run_sheet(order),
            "run_sheet.csv",
        )

    total, perms = list_all_orders(pairs, limit=10000)
    st.write(f"Total permutations: {total:,}")

# ===============================
# Error propagation
# ===============================
elif section == "Error propagation":
    st.header("Error propagation")

    mode = st.radio(
        "Variability mode",
        ["Uniform (same % per step)", "Per-step (specify each step)"],
    )

    if mode.startswith("Uniform"):
        steps = st.number_input("Number of steps", 0, value=4)
        pct = st.number_input("Per-step ±%", 0.0, value=10.0)

        summary = error_propagation_summary_uniform(steps, pct / 100)
        st.write(summary)

    else:
        vals = st.text_area("Comma-separated ±%", "10,8,3,5")
        v_list = [float(x) / 100 for x in vals.split(",") if x.strip()]
        summary = compute_error_distribution(v_list)
        st.write(summary)

# ===============================
# Upload & Preview CSV
# ===============================
elif section == "Upload & Preview CSV":
    st.header("Upload CSV")

    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

    if uploaded:
        df_raw = read_csv_bytes(uploaded.getvalue())
        st.session_state["raw_df"] = df_raw

    df_raw = st.session_state.get("raw_df")

    if df_raw is not None:
        st.subheader("Preview")
        st.dataframe(df_raw.head(50))

        group_cols = st.multiselect(
            "Select group column(s)", df_raw.columns
        )
        value_cols = st.multiselect(
            "Select value column(s)", df_raw.columns
        )

        if group_cols and value_cols:
            long_df = to_long_format(df_raw, group_cols, value_cols)
            st.session_state["uploaded_df"] = long_df
            st.session_state["group_col"] = "Group"
            st.session_state["value_col"] = "Value"

            st.success("Data prepared in long format")
            st.dataframe(long_df.head(20))

# ===============================
# Distribution & Outliers
# ===============================
elif section == "Distribution & Outliers":
    st.header("Distribution & Outliers")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload data first.")
    else:
        groups = sorted(df["Group"].unique())
        selected = st.multiselect("Select groups", groups, default=groups)

        plot_df = df[df["Group"].isin(selected)]

        st.subheader("Distribution (KDE)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=plot_df, x="Value", hue="Group", ax=ax)
        st.pyplot(fig)

        st.subheader("Group comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=plot_df, x="Group", y="Value", ax=ax)
        st.pyplot(fig)

        st.subheader("Outlier detection")
        method = st.selectbox("Method", ["IQR", "Z", "Grubbs"])

        if st.button("Detect outliers"):
            cleaned, report = remove_outliers(
                df, "Group", "Value", method=method
            )
            st.session_state["last_cleaned_df"] = cleaned

            st.subheader("Outliers removed (summary)")
            st.json(report)

            if "removed_table" in report and not report["removed_table"].empty:
                st.subheader("Removed outliers (detailed)")
                st.dataframe(report["removed_table"])

        if st.button("Preview before vs after"):
            cleaned = st.session_state.get("last_cleaned_df")
            if cleaned is not None:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.violinplot(data=df, x="Group", y="Value", ax=ax[0])
                ax[0].set_title("Before")
                sns.violinplot(data=cleaned, x="Group", y="Value", ax=ax[1])
                ax[1].set_title("After")
                st.pyplot(fig)

        if st.button("Apply removal"):
            cleaned = st.session_state.get("last_cleaned_df")
            if cleaned is not None:
                st.session_state["uploaded_df"] = cleaned
                st.success("Outliers permanently removed")

# ===============================
# Statistical Tests & Post-hoc
# ===============================
elif section == "Statistical Tests & Post-hoc":
    st.header("Statistical tests")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload data first.")
    else:
        result = posthoc_auto_select_and_run(
            df, "Group", "Value"
        )

        st.write("Chosen method:", result.get("method", result.get("chosen")))

        if "results" in result:
            tab = pd.DataFrame(result["results"])
            for c in tab.columns:
                if "p" in c.lower():
                    tab[c] = tab[c].apply(format_pvalue)
            st.dataframe(tab)

# ===============================
# Plate map & Export
# ===============================
elif section == "Plate map & Export":
    st.header("Plate map")

    order = st.session_state.get("order", [])
    if not order:
        st.info("Generate run order first.")
    else:
        plate = make_plate_map(order)
        st.dataframe(pd.DataFrame(plate))
        st.download_button(
            "Download plate CSV",
            pd.DataFrame(plate).to_csv(index=False),
            "plate_map.csv",
        )

# ===============================
# Sidebar diagnostics
# ===============================
st.sidebar.markdown("---")
st.sidebar.write("statsmodels:", _HAS_STATSMODELS)
st.sidebar.write("pingouin:", _HAS_PINGOUIN)
st.sidebar.write("scikit-posthocs:", _HAS_SCIPOST)
