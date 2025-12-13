# app.py
"""
Streamlit UI (full-feature)
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

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
    guess_columns,
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

st.set_page_config(page_title="Lab Randomizer + Stats (Full)", layout="wide")
st.title("Lab Randomizer — Randomization, Error Propagation & Stats")

# Sidebar
section = st.sidebar.radio("Section", [
    "Randomizer (pairs & run order)",
    "Error propagation",
    "Upload & Preview CSV",
    "Distribution & Outliers",
    "Statistical Tests & Post-hoc",
    "Plate map & Export"
])

# ------------------------------
# Randomizer
# ------------------------------
if section == "Randomizer (pairs & run order)":
    st.header("Randomizer")
    samples_text = st.text_input("Samples (comma-separated)", "A, B")
    controls_text = st.text_input("Controls (comma-separated)", "C1, C2")
    samples = [s.strip() for s in samples_text.split(',') if s.strip()]
    controls = [c.strip() for c in controls_text.split(',') if c.strip()]
    pairs = generate_valid_pairs(samples, controls)
    st.subheader("Valid pairs")
    if pairs:
        st.write(", ".join(pairs))
    else:
        st.info("No pairs generated.")
    if st.button("Pick random run order"):
        st.session_state['order'] = random_run_order(pairs)
    order = st.session_state.get('order', [])
    if order:
        df_order = pd.DataFrame({"Position": range(1, len(order) + 1), "Pair": order})
        st.table(df_order)
        st.download_button("Download run sheet (CSV)", export_csv_run_sheet(order), file_name="run_sheet.csv", mime="text/csv")
    st.subheader("All permutations (if small)")
    total, perms = list_all_orders(pairs, limit=10000)
    st.write(f"Total permutations: {total:,}")
    if perms:
        idx = st.number_input("Show permutation # (1-based)", min_value=1, max_value=len(perms), value=1)
        st.write(perms[idx - 1])
    else:
        st.info("Too many permutations to list.")

# ------------------------------
# Error propagation
# ------------------------------
elif section == "Error propagation":
    st.header("Error propagation")
    mode = st.radio("Variability mode", ["Uniform (same % per step)", "Per-step (specify each step)"])
    if mode.startswith("Uniform"):
        steps = st.number_input("Number of steps (n)", min_value=0, value=4, step=1)
        pct = st.number_input("Per-step variability ±% (e.g. 10)", min_value=0.0, value=10.0, step=0.1)
        summary = error_propagation_summary_uniform(steps, pct / 100.0)
        st.write(f"Combinations: {summary['combinations']}")
        st.write(f"Worst-case change: +{summary['worst_pct']:.2f}%")
        st.write(f"Best-case change: {summary['best_pct']:.2f}%")
        st.warning("Suggestion: " + ("Consider reducing steps or choosing lower-variance kit" if summary['worst_pct'] > 30 else "Acceptable; consider validating with replicates"))
    else:
        st.info("Provide per-step ±% values")
        method = st.radio("Input method", ["Paste comma list", "Enter number of steps and fill boxes"])
        v_list = []
        if method.startswith("Paste"):
            paste = st.text_area("Paste comma-separated ±% values (e.g. 10,8,3,5)", "10,8,3,5")
            try:
                v_list = [float(x.strip()) / 100.0 for x in paste.split(',') if x.strip()]
                st.write(f"Parsed {len(v_list)} steps.")
            except Exception:
                st.error("Could not parse list. Ensure comma-separated numbers.")
                v_list = []
        else:
            n_steps = st.number_input("Number of steps (n)", min_value=1, value=4, step=1)
            cols = st.columns(4)
            for i in range(n_steps):
                col = cols[i % 4]
                default = 10.0 if i < 4 else 5.0
                v = col.number_input(f"Step {i+1} ±%", min_value=0.0, value=float(default), step=0.1, key=f"step_{i}")
                v_list.append(v / 100.0)
        if v_list:
            summary = compute_error_distribution(v_list, enumerate_limit=2 ** 16, mc_samples=10000)
            st.write(f"n = {summary['n']}; mode = {summary['mode']}; total combos = {summary['combinations']}")
            st.write(f"Worst-case: +{(summary['worst_mult'] - 1.0) * 100:.2f}%")
            st.write(f"Best-case: {(summary['best_mult'] - 1.0) * 100:.2f}%")
            st.write(f"Median: {(summary['median_mult'] - 1.0) * 100:.2f}%")
            st.write(f"5th percentile: {(summary['p05_mult'] - 1.0) * 100:.2f}%")
            st.write(f"95th percentile: {(summary['p95_mult'] - 1.0) * 100:.2f}%")
            fig, ax = plt.subplots()
            ax.hist((summary['samples'] - 1.0) * 100.0, bins=40)
            ax.set_xlabel("Percent change")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of possible percent changes")
            st.pyplot(fig)
            contrib_df = pd.DataFrame(summary['contributions']).sort_values('contribution_fraction', ascending=False)
            contrib_df['v_pct'] = contrib_df['v'] * 100.0
            st.subheader("Per-step contributions to worst-case")
            st.table(contrib_df[['step', 'v_pct', 'contribution_fraction']].rename(columns={'step': 'Step', 'v_pct': '±%'}))
            outdf = pd.DataFrame({'mult': summary['samples']})
            outdf['percent_change'] = (outdf['mult'] - 1.0) * 100.0
            csv_bytes = outdf.to_csv(index=False).encode()
            st.download_button("Download outcomes CSV", csv_bytes, file_name="error_outcomes.csv", mime="text/csv")

# ------------------------------
# Upload & Preview CSV
# ------------------------------
elif section == "Upload & Preview CSV":
    st.header("Upload CSV")
    uploaded = st.file_uploader("Upload CSV file (columns: Group, Value)", type=['csv', 'txt'])
    if uploaded is not None:
        b = uploaded.getvalue()
        df = read_csv_bytes(b)
        st.subheader("Preview")
        st.dataframe(df.head(20))
        st.write("Columns:", list(df.columns))
        suggested_grp, suggested_val = guess_columns(df)
        st.write("Suggested group column:", suggested_grp, "Suggested value column:", suggested_val)
        group_col = st.selectbox("Choose group column", options=[None] + list(df.columns), index=1 if suggested_grp in df.columns else 0)
        value_col = st.selectbox("Choose value column", options=list(df.columns), index=list(df.columns).index(suggested_val) if suggested_val in df.columns else 0)
        if group_col is None:
            st.error("Select a group column")
        else:
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            st.subheader("Group summary")
            st.table(group_summary(df, group_col, value_col))
            st.session_state['uploaded_df'] = df
            st.session_state['group_col'] = group_col
            st.session_state['value_col'] = value_col
            st.success("CSV loaded. Proceed to Distribution & Outliers or Statistical Tests.")

# ------------------------------
# Distribution & Outliers
# ------------------------------
elif section == "Distribution & Outliers":
    st.header("Distribution analysis & Outliers")
    df = st.session_state.get('uploaded_df')

    if df is None:
        st.info("Please upload a CSV first.")
    else:
        group_col = st.session_state['group_col']
        value_col = st.session_state['value_col']
        groups = df[group_col].unique()

        # --------------------------
        # Distribution per group
        # --------------------------
        for g in groups:
            vals = ensure_numeric_series(
                df[df[group_col] == g][value_col]
            ).values

            st.subheader(f"Group: {g} (n={len(vals)})")

            fig, axes = plt.subplots(1, 2, figsize=(8, 2.5))
            axes[0].hist(vals, bins=12)
            axes[0].set_title("Histogram")

            stats.probplot(vals, dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q plot")

            st.pyplot(fig)

            sh = shapiro_test(vals)
            sh_str = format_pvalue(sh.get('pvalue')) if sh.get('pvalue') is not None else "NA"
            st.write(f"Shapiro–Wilk p = {sh_str}")

            if sh.get('pvalue') is not None and sh['pvalue'] < 0.05:
                st.warning("Likely non-normal distribution (p < 0.05)")
            else:
                st.info("Cannot reject normality (p ≥ 0.05)")

        # --------------------------
        # Outlier detection section
        # --------------------------
        st.subheader("Outlier detection")

        method = st.selectbox(
            "Detection method",
            options=['IQR', 'Z', 'Grubbs']
        )

        if st.button("Detect outliers (preview)"):
            cleaned, report = remove_outliers(
                df,
                group_col,
                value_col,
                method=method
            )

            # Group-level summary
            st.subheader("Outlier summary (per group)")
            summary_rows = []
            for k, v in report.items():
                if isinstance(v, dict) and 'removed_count' in v:
                    summary_rows.append({
                        'Group': k,
                        'Removed count': v['removed_count']
                    })

            if summary_rows:
                st.table(pd.DataFrame(summary_rows))
            else:
                st.info("No outliers detected in any group.")

            # Detailed removed outliers table
            if 'removed_table' in report and not report['removed_table'].empty:
                st.subheader("Removed outliers (detailed)")
                st.dataframe(report['removed_table'])
            else:
                st.info("No individual outlier rows to display.")

            st.session_state['last_cleaned_df'] = cleaned
            st.success("Preview ready. Use 'Apply removal' to commit changes.")

        # --------------------------
        # Before vs After preview
        # --------------------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Preview before vs after"):
                cleaned = st.session_state.get('last_cleaned_df')
                if cleaned is None:
                    st.error("Run outlier detection first.")
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

                    df.boxplot(
                        column=value_col,
                        by=group_col,
                        ax=ax[0]
                    )
                    ax[0].set_title("Before")

                    cleaned.boxplot(
                        column=value_col,
                        by=group_col,
                        ax=ax[1]
                    )
                    ax[1].set_title("After")

                    plt.suptitle("")
                    st.pyplot(fig)

        # --------------------------
        # Apply removal permanently
        # --------------------------
        with col2:
            if st.button("Apply removal"):
                cleaned = st.session_state.get('last_cleaned_df')
                if cleaned is None:
                    st.error("Run outlier detection first.")
                else:
                    st.session_state['uploaded_df'] = cleaned
                    st.success("Outliers removed and data updated.")


# ------------------------------
# Statistical Tests & Post-hoc
# ------------------------------
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
