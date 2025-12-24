# app.py
"""
LabFlow — Streamlit UI (full, stable)
Run with: streamlit run app.py
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
    group_summary,
    shapiro_test,
    levene_test,
    remove_outliers,
    posthoc_auto_select_and_run,
    format_pvalue,
    ensure_numeric_series,
    _HAS_STATSMODELS,
    _HAS_PINGOUIN,
    _HAS_SCIPOST
)

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="LabFlow", layout="wide")
st.title("LabFlow — Randomization, Error Propagation & Statistics")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
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

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def to_long_format(df, group_cols, value_cols):
    df = df.copy()

    # Strip column names (prevents invisible bugs)
    df.columns = df.columns.str.strip()

    # Track original row
    df["__row_id__"] = np.arange(len(df))

    # Remove overlaps safely
    group_cols = list(dict.fromkeys(group_cols))
    value_cols = [c for c in value_cols if c not in group_cols]

    if len(value_cols) == 0:
        raise ValueError("No valid value columns selected after removing overlaps.")

    # Melt
    melted = df.melt(
        id_vars=group_cols + ["__row_id__"],
        value_vars=value_cols,
        var_name="Variable",
        value_name="Value",
    )

    # ✅ EMPTY CELLS → ZERO (as requested)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce").fillna(0)

    # Combine group columns into one Group label
    melted["Group"] = (
        melted[group_cols]
        .astype(str)
        .apply(lambda x: "_".join(x), axis=1)
    )

    # -------------------------------
    # GROUP SIZE VALIDATION
    # -------------------------------
    group_counts = melted.groupby("Group")["Value"].count()

    warn_groups = group_counts[group_counts < 3]
    drop_groups = group_counts[group_counts < 2]

    # Store report for UI
    melted.attrs["group_validation_report"] = pd.DataFrame({
        "Group": group_counts.index,
        "Observations": group_counts.values,
        "Status": [
            (
                "Excluded (<2 observations)"
                if g in drop_groups.index
                else "Warning (<3 observations)"
                if g in warn_groups.index
                else "OK"
            )
            for g in group_counts.index
        ]
    })

    #Remove groups with <2 observations
    melted = melted[~melted["Group"].isin(drop_groups.index)]

    return melted[["Group", "Value", "Variable", "__row_id__"]]


# ==================================================
# RANDOMIZER
# ==================================================
if section == "Randomizer (pairs & run order)":
    st.header("Randomizer")

    samples_text = st.text_input("Samples (comma-separated)", "A, B")
    controls_text = st.text_input("Controls (comma-separated)", "C1, C2")

    samples = [s.strip() for s in samples_text.split(",") if s.strip()]
    controls = [c.strip() for c in controls_text.split(",") if c.strip()]

    pairs = generate_valid_pairs(samples, controls)

    st.subheader("Valid pairs")
    st.write(", ".join(pairs) if pairs else "No valid pairs.")

    if st.button("Pick random run order"):
        st.session_state["order"] = random_run_order(pairs)

    order = st.session_state.get("order", [])

    if order:
        st.table(pd.DataFrame({
            "Position": range(1, len(order) + 1),
            "Pair": order
        }))
        st.download_button(
            "Download run sheet",
            export_csv_run_sheet(order),
            file_name="run_sheet.csv"
        )

    st.subheader("All permutations (if small)")
    total, perms = list_all_orders(pairs, limit=10000)
    st.write(f"Total permutations: {total:,}")

    if perms:
        idx = st.number_input(
            "Show permutation #",
            min_value=1,
            max_value=len(perms),
            value=1
        )
        st.code(perms[idx - 1])
    else:
        st.info("Too many permutations to list.")


# ==================================================
# ERROR PROPAGATION
# ==================================================
elif section == "Error propagation":
    st.header("Error propagation")

    mode = st.radio(
        "Variability mode",
        ["Uniform (same % per step)", "Per-step (specify each step)"]
    )

    if mode.startswith("Uniform"):
        steps = st.number_input("Steps", min_value=0, value=4)
        pct = st.number_input("±% per step", min_value=0.0, value=10.0)

        summary = error_propagation_summary_uniform(steps, pct / 100)

        summary_df = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"]
        )
        st.table(summary_df)

    else:
        paste = st.text_area("Comma-separated ±%", "10,8,3,5")
        v_list = [float(x) / 100 for x in paste.split(",") if x.strip()]

        summary = compute_error_distribution(v_list)

        summary_df = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"]
        )
        st.table(summary_df)


# ==================================================
# UPLOAD & PREVIEW
# ==================================================
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

        group_cols = st.multiselect("Group column(s)", df_raw.columns)
        value_cols = st.multiselect("Value column(s)", df_raw.columns)

        if group_cols and value_cols:
            long_df = to_long_format(df_raw, group_cols, value_cols)

            # 🔥 HARD RESET: remove ALL pandas metadata
            long_df = long_df.copy()
            long_df.attrs = {}                  # <-- THIS was the real bug
            long_df = long_df.reset_index(drop=True)

            # force Arrow-safe dtypes
            for c in long_df.columns:
                if c != "Value":
                    long_df[c] = long_df[c].astype(str)
            long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

            # store clean data
            st.session_state["uploaded_df"] = long_df
            st.session_state["group_col"] = "Group"
            st.session_state["value_col"] = "Value"

            st.subheader("Prepared data")
            st.dataframe(long_df.head(20))
            st.success("Data loaded and formatted.")



# ==================================================
# DISTRIBUTION & OUTLIERS
# ==================================================
elif section == "Distribution & Outliers":
    st.header("Distribution & Outliers")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload data first.")
    else:
        groups = sorted(df["Group"].unique())
        selected = st.multiselect("Select groups", groups, default=groups)

        plot_df = df[df["Group"].isin(selected)]

        # -------------------------------
        # KDE distribution
        # -------------------------------
        st.subheader("Distribution (KDE)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=plot_df, x="Value", hue="Group", ax=ax)
        st.pyplot(fig)

        # SVG download
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        st.download_button(
            "Download KDE plot (SVG)",
            buf.getvalue(),
            file_name="distribution_kde.svg",
            mime="image/svg+xml",
        )

        # -------------------------------
        # Violin plot
        # -------------------------------
        st.subheader("Group comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=plot_df, x="Group", y="Value", ax=ax)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        st.download_button(
            "Download violin plot (SVG)",
            buf.getvalue(),
            file_name="group_violin.svg",
            mime="image/svg+xml",
        )

        # -------------------------------
        # Outlier detection
        # -------------------------------
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

                buf = io.BytesIO()
                fig.savefig(buf, format="svg", bbox_inches="tight")
                st.download_button(
                    "Download before/after plot (SVG)",
                    buf.getvalue(),
                    file_name="outliers_before_after.svg",
                    mime="image/svg+xml",
                )

        if st.button("Apply removal"):
            cleaned = st.session_state.get("last_cleaned_df")
            if cleaned is not None:
                st.session_state["uploaded_df"] = cleaned
                st.success("Outliers permanently removed")


# ==================================================
# STATISTICS & POST-HOC (RESTORED FULLY)
# ==================================================
elif section == "Statistical Tests & Post-hoc":
    st.header("Statistical Tests & Post-hoc")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload data first.")
    else:
        group_col = "Group"
        value_col = "Value"

        paired = st.checkbox("Paired?", value=False)

        # -----------------------------
        # Prepare groups safely
        # -----------------------------
        groups = sorted(df[group_col].unique())
        group_values = {}
        size_rows = []

        for g in groups:
            vals = ensure_numeric_series(
                df[df[group_col] == g][value_col]
            ).dropna().values

            if len(vals) > 0:
                group_values[g] = vals

            size_rows.append({
                "Group": g,
                "n": len(vals)
            })

        size_df = pd.DataFrame(size_rows)
        st.subheader("Sample sizes")
        st.table(size_df)

        valid_groups = [g for g, v in group_values.items() if len(v) >= 2]

        if len(valid_groups) < 2:
            st.error("At least two groups with n ≥ 2 are required for statistical testing.")
            st.stop()

        # -----------------------------
        # Assumption checks
        # -----------------------------
        st.subheader("Assumption checks")

        shapiro_rows = []
        normal_flags = []

        for g in valid_groups:
            vals = group_values[g]
            sh = shapiro_test(vals)
            p = sh.get("pvalue")

            normal = (p is not None and p >= 0.05)
            normal_flags.append(normal)

            shapiro_rows.append({
                "Group": g,
                "n": len(vals),
                "Shapiro p": format_pvalue(p),
                "Normal?": "Yes" if normal else "No"
            })

        st.table(pd.DataFrame(shapiro_rows))

        # Levene test (variance equality)
        try:
            lev = levene_test([group_values[g] for g in valid_groups])
            lev_p = lev.get("p")
            equal_var = lev_p is not None and lev_p >= 0.05
            st.write("Levene p:", format_pvalue(lev_p))
        except Exception:
            lev_p = None
            equal_var = False
            st.warning("Levene test could not be computed.")

        # -----------------------------
        # Recommend global test
        # -----------------------------
        if len(valid_groups) == 2:
            if all(normal_flags) and equal_var:
                recommended_global = "t-test"
            elif all(normal_flags):
                recommended_global = "Welch's t-test"
            else:
                recommended_global = "Mann–Whitney U"
        else:
            if all(normal_flags) and equal_var:
                recommended_global = "One-way ANOVA"
            else:
                recommended_global = "Kruskal–Wallis"

        st.subheader("Recommended global test")
        st.info(recommended_global)

        # -----------------------------
        # Post-hoc selection
        # -----------------------------
        st.subheader("Post-hoc analysis")

        auto_result = posthoc_auto_select_and_run(
            df[df[group_col].isin(valid_groups)],
            group_col,
            value_col,
            paired=paired
        )

        st.write("Auto-selected post-hoc:", auto_result.get("chosen"))

        method_choice = st.selectbox(
            "Post-hoc method",
            [
                "Auto (recommended)",
                "Tukey HSD",
                "Games-Howell",
                "Dunn (Bonferroni)",
                "Bonferroni pairwise"
            ]
        )

        if st.button("Run post-hoc"):
            if method_choice == "Auto (recommended)":
                result = auto_result
            else:
                result = posthoc_auto_select_and_run(
                    df[df[group_col].isin(valid_groups)],
                    group_col,
                    value_col,
                    paired=paired,
                    force=method_choice
                )

            st.subheader("Post-hoc results")

            if "error" in result:
                st.error(result["error"])

            elif isinstance(result.get("results"), list):
                tab = pd.DataFrame(result["results"])

                if tab.empty:
                    st.info("No significant pairwise differences found.")
                else:
                    for c in tab.columns:
                        if "p" in c.lower():
                            tab[c] = tab[c].apply(format_pvalue)

                    st.dataframe(tab)

            else:
                st.write(result)


# ==================================================
# PLATE MAP
# ==================================================
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

# ==================================================
# Sidebar diagnostics
# ==================================================
st.sidebar.markdown("---")
st.sidebar.write("statsmodels:", _HAS_STATSMODELS)
st.sidebar.write("pingouin:", _HAS_PINGOUIN)
st.sidebar.write("scikit-posthocs:", _HAS_SCIPOST)









