# app.py
"""
LabFlow Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    _HAS_STATSMODELS,
    _HAS_PINGOUIN,
    _HAS_SCIPOST,
)

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="LabFlow", layout="wide")
st.title("LabFlow — Randomization, Error Propagation & Statistics")

# =========================================================
# Sidebar navigation
# =========================================================
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

# =========================================================
# Utilities
# =========================================================
def to_long_format(df, group_cols, value_cols):
    df = df.copy()

    # ----------------------------------
    # HARD SANITIZATION (CRITICAL)
    # ----------------------------------

    # Convert all column names to strings
    df.columns = df.columns.map(str)

    # Strip whitespace
    df.columns = df.columns.str.strip()

    # Make column names UNIQUE (KEY FIX)
    if df.columns.duplicated().any():
        df.columns = pd.io.parsers.ParserBase(
            {"names": df.columns}
        )._maybe_dedup_names(df.columns)

    # Clean user selections
    group_cols = [str(c).strip() for c in group_cols]
    value_cols = [str(c).strip() for c in value_cols]

    # Remove overlaps
    group_cols = list(dict.fromkeys(group_cols))
    value_cols = [c for c in value_cols if c not in group_cols]

    # Validate existence
    missing_g = [c for c in group_cols if c not in df.columns]
    missing_v = [c for c in value_cols if c not in df.columns]

    if missing_g or missing_v:
        raise ValueError(
            f"Invalid column selection.\n"
            f"Missing group columns: {missing_g}\n"
            f"Missing value columns: {missing_v}"
        )

    if not value_cols:
        raise ValueError("No valid value columns selected.")

    # Track original row
    df["__row_id__"] = np.arange(len(df))

    # ----------------------------------
    # MELT (NOW SAFE)
    # ----------------------------------
    melted = df.melt(
        id_vars=group_cols + ["__row_id__"],
        value_vars=value_cols,
        var_name="Variable",
        value_name="Value",
    )

    melted = melted.dropna(subset=["Value"])

    # Combine group columns
    melted["Group"] = (
        melted[group_cols]
        .astype(str)
        .agg("_".join, axis=1)
    )

    return melted[["Group", "Value", "Variable", "__row_id__"]]




# =========================================================
# RANDOMIZER
# =========================================================
if section == "Randomizer (pairs & run order)":
    st.header("Randomizer")

    samples_text = st.text_input("Samples (comma-separated)", "A, B")
    controls_text = st.text_input("Controls (comma-separated)", "C1, C2")

    samples = [s.strip() for s in samples_text.split(",") if s.strip()]
    controls = [c.strip() for c in controls_text.split(",") if c.strip()]

    pairs = generate_valid_pairs(samples, controls)

    st.subheader("Valid sample–control pairs")
    if pairs:
        st.write(", ".join(pairs))
    else:
        st.info("No valid pairs generated.")

    if st.button("Pick random run order"):
        st.session_state["order"] = random_run_order(pairs)

    order = st.session_state.get("order", [])

    if order:
        df_order = pd.DataFrame(
            {"Position": range(1, len(order) + 1), "Pair": order}
        )
        st.table(df_order)

        st.download_button(
            "Download run sheet (CSV)",
            export_csv_run_sheet(order),
            "run_sheet.csv",
        )

    st.subheader("All possible permutations (if feasible)")
    total, perms = list_all_orders(pairs, limit=10000)
    st.write(f"Total permutations: {total:,}")

    if perms:
        idx = st.number_input(
            "Show permutation #",
            min_value=1,
            max_value=len(perms),
            value=1,
        )
        st.code(perms[idx - 1])
    else:
        st.info("Too many permutations to list.")

# =========================================================
# ERROR PROPAGATION
# =========================================================
elif section == "Error propagation":
    st.header("Error propagation")

    mode = st.radio(
        "Variability mode",
        ["Uniform (same % per step)", "Per-step (specify each step)"],
    )

    if mode.startswith("Uniform"):
        steps = st.number_input("Number of steps", min_value=0, value=4)
        pct = st.number_input("Per-step ±%", min_value=0.0, value=10.0)
        summary = error_propagation_summary_uniform(steps, pct / 100)
        st.write(summary)
    else:
        vals = st.text_area("Comma-separated ±%", "10,8,3,5")
        v_list = [float(v) / 100 for v in vals.split(",") if v.strip()]
        summary = compute_error_distribution(v_list)
        st.write(summary)

# =========================================================
# UPLOAD & PREVIEW
# =========================================================
elif section == "Upload & Preview CSV":
    st.header("Upload CSV")

    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
    if uploaded:
        st.session_state["raw_df"] = read_csv_bytes(uploaded.getvalue())

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
            st.success("Data prepared in long format")
            st.dataframe(long_df.head(20))

# =========================================================
# DISTRIBUTION & OUTLIERS
# =========================================================
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
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.kdeplot(data=plot_df, x="Value", hue="Group", ax=ax)
        st.pyplot(fig)

        st.subheader("Group comparison")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.violinplot(data=plot_df, x="Group", y="Value", ax=ax)
        st.pyplot(fig)

        st.subheader("Outlier detection")
        method = st.selectbox("Method", ["IQR", "Z", "Grubbs"])

        if st.button("Detect outliers"):
            cleaned, report = remove_outliers(
                df, "Group", "Value", method=method
            )
            st.session_state["last_cleaned_df"] = cleaned
            st.json(report)

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

# =========================================================
# STATISTICS
# =========================================================
elif section == "Statistical Tests & Post-hoc":
    st.header("Statistical tests")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload data first.")
    else:
        result = posthoc_auto_select_and_run(df, "Group", "Value")
        st.write("Chosen method:", result.get("method", result.get("chosen")))

        if "results" in result:
            tab = pd.DataFrame(result["results"])
            for c in tab.columns:
                if "p" in c.lower():
                    tab[c] = tab[c].apply(format_pvalue)
            st.dataframe(tab)

# =========================================================
# PLATE MAP
# =========================================================
elif section == "Plate map & Export":
    st.header("Plate map")

    order = st.session_state.get("order", [])
    if not order:
        st.info("Generate run order first.")
    else:
        plate = make_plate_map(order)
        df_plate = pd.DataFrame(plate)
        st.dataframe(df_plate)
        st.download_button(
            "Download plate CSV",
            df_plate.to_csv(index=False),
            "plate_map.csv",
        )

# =========================================================
# Diagnostics
# =========================================================
st.sidebar.markdown("---")
st.sidebar.write("statsmodels:", _HAS_STATSMODELS)
st.sidebar.write("pingouin:", _HAS_PINGOUIN)
st.sidebar.write("scikit-posthocs:", _HAS_SCIPOST)



