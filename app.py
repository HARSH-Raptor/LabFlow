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
 df = st.session_state.get('uploaded_df')
 if df is None:
 st.info("Upload CSV first.")
 else:
 group_col = st.session_state['group_col']
 value_col = st.session_state['value_col']
 st.write("Groups:", df[group_col].unique())
 paired = st.checkbox("Paired?", value=False)
 groups = df[group_col].unique()
 group_values = [ensure_numeric_series(df[df[group_col] == g][value_col]).values for g in groups]
 lev = None
 try:
 lev_stat, lev_p = stats.levene(*group_values)
 lev = {'stat': float(lev_stat), 'p': float(lev_p)}
 except Exception:
 lev = {'stat': None, 'p': None}
 st.write("Levene p:", format_pvalue(lev.get('p')))
 sh_list = []
 for g in groups:
 vals = ensure_numeric_series(df[df[group_col] == g][value_col]).values
 sh = shapiro_test(vals)
 sh_list.append((g, sh.get('pvalue')))
 st.table(pd.DataFrame(sh_list, columns=['Group', 'Shapiro_p']).set_index('Group'))
 
 # Recommend & run main test
 norm_flags = [p is not None and p >= 0.05 for (_, p) in sh_list]
 equal_var_flag = lev.get('p') is not None and lev.get('p') >= 0.05
 
 if len(groups) == 2:
 if all(norm_flags) and equal_var_flag:
 recommended = "t-test (two-sample)"
 elif all(norm_flags) and not equal_var_flag:
 recommended = "Welch's t-test"
 else:
 recommended = "Mann-Whitney U"
 else:
 if all(norm_flags) and equal_var_flag:
 recommended = "One-way ANOVA"
 else:
 recommended = "Kruskal-Wallis"
 
 st.info(f"Recommended: {recommended}")
 if st.button("Run recommended test"):
 if recommended == "t-test (two-sample)" or recommended == "Welch's t-test":
 v1 = group_values[0]; v2 = group_values[1]
 if recommended == "t-test (two-sample)":
 res = stats.ttest_ind(v1, v2, equal_var=True, nan_policy='omit')
 else:
 res = stats.ttest_ind(v1, v2, equal_var=False, nan_policy='omit')
 st.write(f"stat = {res[0]:.4f}, p = {format_pvalue(res[1])}")
 elif recommended == "Mann-Whitney U":
 v1 = group_values[0]; v2 = group_values[1]
 res = stats.mannwhitneyu(v1, v2, alternative='two-sided')
 st.write(f"U = {res[0]}, p = {format_pvalue(res[1])}")
 elif recommended == "One-way ANOVA":
 res = stats.f_oneway(*group_values)
 st.write(f"ANOVA: F = {res[0]:.4f}, p = {format_pvalue(res[1])}")
 else:
 res = stats.kruskal(*group_values)
 st.write(f"Kruskal-Wallis: H = {res[0]:.4f}, p = {format_pvalue(res[1])}")
 
 # Post-hoc UI with persistent state
 if 'posthoc_open' not in st.session_state:
 st.session_state.posthoc_open = False
 st.session_state.posthoc_choice = None
 
 st.session_state.posthoc_open = st.checkbox("Show post-hoc options", value=st.session_state.posthoc_open)
 if st.session_state.posthoc_open:
 # auto-select suggestion
 suggestion = posthoc_auto_select_and_run(df, group_col, value_col, paired=paired)
 suggested = suggestion.get('chosen')
 st.write("Auto-selected:", suggested)
 st.write("Auto-detect details: Shapiro p-values per group and Levene p-value above.")
 st.write("You can run the suggested method, or select a manual method.")
 method_choice = st.selectbox("Post-hoc method", options=['Auto (suggested)','Tukey HSD','Games-Howell','Dunn (Bonferroni)','Bonferroni pairwise'], index=0)
 run_ph = st.button("Run post-hoc")
 if run_ph:
 chosen = None
 if method_choice == 'Auto (suggested)':
 chosen = suggested
 elif method_choice == 'Tukey HSD':
 chosen = 'tukey'
 elif method_choice == 'Games-Howell':
 chosen = 'gameshowell'
 elif method_choice == 'Dunn (Bonferroni)':
 chosen = 'dunn'
 else:
 chosen = 'bonferroni'
 
 # run using the same internal function by tweaking df if needed
 res = posthoc_auto_select_and_run(df, group_col, value_col, paired=paired)
 # if user selected manual method, override execution where possible
 if chosen != res.get('chosen'):
 # attempt manual runs here
 if chosen == 'tukey' and _HAS_STATSMODELS:
 try:
 data = df[[group_col, value_col]].dropna()
 tuk = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=0.05)
 tuk_df = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
 out = {'method': 'TukeyHSD', 'results': tuk_df.to_dict(orient='records')}
 except Exception as e:
 out = {'method': 'TukeyHSD', 'error': str(e), 'results': []}
 elif chosen == 'gameshowell' and _HAS_PINGOUIN:
 try:
 outdf = pg.pairwise_gameshowell(data=df[[group_col, value_col]].dropna(), dv=value_col, between=group_col)
 out = {'method': 'Games-Howell', 'results': outdf.to_dict(orient='records')}
 except Exception as e:
 out = {'method': 'Games-Howell', 'error': str(e), 'results': []}
 elif chosen == 'dunn' and _HAS_SCIPOST:
 try:
 ph = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
 pairs = []
 groups_list = list(ph.index)
 for i, g1 in enumerate(groups_list):
 for j, g2 in enumerate(groups_list):
 if j <= i:
 continue
 pval = ph.iloc[i, j]
 pairs.append({'group1': g1, 'group2': g2, 'pvalue': float(pval), 'p_adj': float(pval), 'significant': float(pval) < 0.05})
 out = {'method': 'Dunn (Bonferroni)', 'results': pairs}
 except Exception as e:
 out = {'method': 'Dunn', 'error': str(e), 'results': []}
 else:
 out = posthoc_auto_select_and_run(df, group_col, value_col, paired=paired)
 else:
 out = res
 
 st.subheader("Post-hoc results")
 st.write("Method:", out.get('method', out.get('chosen', 'Unknown')))
 if 'error' in out:
 st.error("Error running post-hoc: " + str(out['error']))
 if isinstance(out.get('results'), list):
 # pretty print table
 tab = pd.DataFrame(out['results'])
 if not tab.empty:
 # format p-values if present
 if 'pvalue' in tab.columns:
 tab['pvalue'] = tab['pvalue'].apply(format_pvalue)
 if 'p_adj' in tab.columns:
 tab['p_adj'] = tab['p_adj'].apply(format_pvalue)
 st.dataframe(tab)
 else:
 st.write("No results returned.")
 else:
 st.write(out.get('results'))

# =========================================================
# PLATE MAP
# =========================================================
# ------------------------------
# Plate map & Export
# ------------------------------
elif section == "Plate map & Export":
    st.header("Plate map & Export")
    order = st.session_state.get('order', [])
    if not order:
        st.info("Generate run order in Randomizer first.")
    else:
        st.write("Current run order length:", len(order))
        cols = st.selectbox("Columns", [12, 8, 6], index=0)
        rows = st.selectbox("Rows", [8, 6, 4], index=0)
        if st.button("Create plate map"):
            plate = make_plate_map(order, num_columns=cols, num_rows=rows)
            st.dataframe(pd.DataFrame(plate))
            csv_bytes = pd.DataFrame(plate).to_csv(index=False).encode()
            st.download_button("Download plate CSV", csv_bytes, file_name="plate_map.csv", mime="text/csv")

# Sidebar note
st.sidebar.markdown("---")
st.sidebar.write("Optional libs: statsmodels: " + ("Yes" if _HAS_STATSMODELS else "No"))
st.sidebar.write("pingouin: " + ("Yes" if _HAS_PINGOUIN else "No"))
st.sidebar.write("scikit-posthocs: " + ("Yes" if _HAS_SCIPOST else "No"))





