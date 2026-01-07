# lab_randomizer.py
"""
Backend utilities for Lab Randomizer (full-feature version)
Includes:
 - pairing & randomization
 - plate mapping
 - error propagation
 - CSV helpers & grouping
 - distribution tests (Shapiro, Levene)
 - outlier detection/removal (IQR, Z, Grubbs)
 - statistical tests (t-test, ANOVA, Mann-Whitney, Kruskal)
 - post-hoc: Tukey (statsmodels), Games-Howell (pingouin), Dunn (scikit-posthocs), Bonferroni pairwise
 - p-value formatting utility
"""

from typing import List, Dict, Any, Tuple, Optional
import itertools
import math
import random
import csv
import io

import numpy as np
import pandas as pd
from scipy import stats

# optional libs for post-hoc
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except Exception:
    _HAS_PINGOUIN = False

try:
    import scikit_posthocs as sp
    _HAS_SCIPOST = True
except Exception:
    _HAS_SCIPOST = False


# -------------------------
# Utilities
# -------------------------
def format_pvalue(p: Optional[float], sig_digits: int = 2) -> str:
    """Nicely format p-values: decimals up to 6 places; if smaller, use scientific notation."""
    if p is None:
        return "NA"
    try:
        p = float(p)
    except Exception:
        return str(p)
    if p == 0.0:
        return "<1e-300"
    # show normal decimals if >= 1e-6
    if p >= 1e-6:
        return f"{p:.6f}"
    # otherwise scientific
    return f"{p:.{sig_digits}e}"

def add_significance_and_formatting(rows, alpha=0.05):
    """
    Normalize p-values across post-hoc outputs.
    Adds:
      - p_adj_raw
      - p_adj_formatted
      - significant
    """
    for r in rows:
        # detect p-value key
        p = None
        for k in ('p_adj', 'p-adj', 'pval', 'pvalue', 'p_tukey'):
            if k in r and r[k] is not None:
                p = float(r[k])
                break

        r['p_adj_raw'] = p
        r['p_adj_formatted'] = format_pvalue(p)
        r['significant'] = (p is not None) and (p < alpha)

    return rows


# -------------------------
# Pair & ordering utilities
# -------------------------
def generate_valid_pairs(samples: List[str], controls: List[str]) -> List[str]:
    s = [x.strip() for x in samples if x and x.strip()]
    c = [x.strip() for x in controls if x and x.strip()]
    return [f"{si}{cj}" for si in s for cj in c]


def list_all_orders(pairs: List[str], limit: int = 100000) -> Tuple[int, List[Tuple[str, ...]]]:
    n = len(pairs)
    total = math.factorial(n) if n >= 0 else 0
    if total > limit:
        return total, []
    return total, list(itertools.permutations(pairs))


def random_run_order(pairs: List[str]) -> List[str]:
    p = pairs.copy()
    random.shuffle(p)
    return p


def export_csv_run_sheet(run_order: List[str]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Position', 'Pair'])
    for i, p in enumerate(run_order, start=1):
        w.writerow([i, p])
    return buf.getvalue().encode()


# -------------------------
# Plate map
# -------------------------
def make_plate_map(pairs: List[str], num_columns: int = 12, num_rows: int = 8) -> List[List[str]]:
    slots = num_rows * num_columns
    grid = [['' for _ in range(num_columns)] for _ in range(num_rows)]
    for idx, item in enumerate(pairs[:slots]):
        r = idx // num_columns
        c = idx % num_columns
        grid[r][c] = item
    return grid


# -------------------------
# Error propagation
# -------------------------
def _product(values: List[float]) -> float:
    p = 1.0
    for v in values:
        p *= v
    return p

def error_propagation_summary_uniform(steps, pct):
    """
    pct should be a fraction (e.g. 0.10 for 10%)
    """
    if steps == 0:
        return {
            "Steps": 0,
            "Per-step uncertainty (%)": 0.0,
            "Best case total error (%)": 0.0,
            "Worst-case total error (%)": 0.0,
            "Total uncertainty range (%)": "±0.0"
        }

    total_uncertainty = np.sqrt(steps) * pct * 100

    return {
        "Steps": steps,
        "Per-step uncertainty (%)": round(pct * 100, 3),
        "Best case total error (%)": 0.0,
        "Worst-case total error (%)": round(total_uncertainty, 3),
        "Total uncertainty range (%)": f"±{round(total_uncertainty, 3)}"
    }


def compute_error_distribution(v_list):
    """
    v_list should be fractions (e.g. 0.10 for 10%)
    """
    if len(v_list) == 0:
        return {
            "Steps": 0,
            "Best case total error (%)": 0.0,
            "Worst-case total error (%)": 0.0,
            "Total uncertainty range (%)": "±0.0"
        }

    squared = np.square(v_list)
    total_uncertainty = np.sqrt(np.sum(squared)) * 100

    contribution_rows = []
    for i, v in enumerate(v_list, start=1):
        contribution = (v**2 / np.sum(squared)) * 100
        contribution_rows.append({
            "Step": i,
            "Step uncertainty (%)": round(v * 100, 3),
            "Contribution to total (%)": round(contribution, 2)
        })

    summary = {
        "Steps": len(v_list),
        "Best case total error (%)": 0.0,
        "Worst-case total error (%)": round(total_uncertainty, 3),
        "Total uncertainty range (%)": f"±{round(total_uncertainty, 3)}",
        "Step-wise contributions": pd.DataFrame(contribution_rows)
    }

    return summary

# -------------------------
# CSV helpers
# -------------------------
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    txt = contents.decode('utf-8')
    df = pd.read_csv(io.StringIO(txt))
    return df


def guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df.shape[1] == 1:
        return None, df.columns[0]
    grp_candidates = [c for c in df.columns if c.lower() in ('group', 'label', 'condition', 'sample')]
    val_candidates = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not grp_candidates:
        grp_candidates = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    group_col = grp_candidates[0] if grp_candidates else None
    value_col = val_candidates[0] if val_candidates else None
    return group_col, value_col


def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[value_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    g = g.rename(columns={'count': 'n', 'std': 'sd'})
    g['sem'] = g['sd'] / np.sqrt(g['n'])
    return g


# -------------------------
# Distribution tests
# -------------------------
def shapiro_test(values: np.ndarray) -> Dict[str, Any]:
    if len(values) < 3:
        return {'statistic': None, 'pvalue': None, 'warning': 'Too few points (<3) for Shapiro-Wilk'}
    stat, p = stats.shapiro(values)
    return {'statistic': float(stat), 'pvalue': float(p)}


def levene_test(*groups: np.ndarray) -> Dict[str, Any]:
    if any(len(g) < 2 for g in groups):
        return {'statistic': None, 'pvalue': None, 'warning': 'One or more groups too small for Levene'}
    stat, p = stats.levene(*groups)
    return {'statistic': float(stat), 'pvalue': float(p)}


# -------------------------
# Outlier detection
# -------------------------
def detect_outliers_iqr(values: np.ndarray) -> np.ndarray:
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (values < lower) | (values > upper)
    return mask


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    zs = np.abs(stats.zscore(values, nan_policy='omit'))
    zs = np.nan_to_num(zs, nan=0.0)
    return zs > threshold


def detect_outliers_grubbs(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    vals = values.copy().astype(float)
    mask = np.zeros(len(vals), dtype=bool)
    if len(vals) < 3:
        return mask
    remaining_idx = np.arange(len(vals))
    while True:
        if len(remaining_idx) < 3:
            break
        sub = vals[remaining_idx]
        mean = np.mean(sub)
        sd = np.std(sub, ddof=1)
        diffs = np.abs(sub - mean)
        max_idx = np.argmax(diffs)
        G = diffs[max_idx] / sd if sd > 0 else 0.0
        n = len(sub)
        t = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t ** 2 / (n - 2 + t ** 2))
        if G > crit:
            out_idx = remaining_idx[max_idx]
            mask[out_idx] = True
            remaining_idx = np.delete(remaining_idx, max_idx)
        else:
            break
    return mask


def remove_outliers(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    method: str = 'IQR'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    removed_indices = []
    report = {}
    detailed_rows = []   # NEW: per-outlier details

    for grp, grp_df in df.groupby(group_col):
        vals = grp_df[value_col].values

        if method == 'IQR':
            mask = detect_outliers_iqr(vals)
        elif method == 'Z':
            mask = detect_outliers_zscore(vals)
        elif method == 'Grubbs':
            mask = detect_outliers_grubbs(vals)
        else:
            raise ValueError("Unknown method")

        outlier_idx = grp_df.index[mask]
        removed_indices.extend(outlier_idx.tolist())

        # group-level summary (unchanged behaviour)
        report[str(grp)] = {
            'removed_count': int(mask.sum()),
            'removed_indices': outlier_idx.tolist()
        }

        # NEW: detailed row-wise info
        for idx in outlier_idx:
            detailed_rows.append({
                'group': grp,
                'row_index': int(idx),            # original CSV row index
                'value': df.loc[idx, value_col],
                'method': method
            })

    # create cleaned dataframe
    cleaned = df.drop(index=removed_indices).reset_index(drop=True)

    # global summary
    report['total_removed'] = len(removed_indices)

    # NEW: detailed table (easy for Streamlit display)
    report['removed_table'] = pd.DataFrame(detailed_rows)

    return cleaned, report



# -------------------------
# Statistical tests & post-hoc
# -------------------------
def t_test(group1: np.ndarray, group2: np.ndarray, paired: bool = False, equal_var: bool = True) -> Dict[str, Any]:
    if paired:
        if len(group1) != len(group2):
            return {'error': 'Paired test requires equal-length groups'}
        stat, p = stats.ttest_rel(group1, group2, nan_policy='omit')
    else:
        stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var, nan_policy='omit')
    return {'statistic': float(stat), 'pvalue': float(p)}


def mannwhitney(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return {'statistic': float(stat), 'pvalue': float(p)}


def one_way_anova(*groups: np.ndarray) -> Dict[str, Any]:
    stat, p = stats.f_oneway(*groups)
    return {'statistic': float(stat), 'pvalue': float(p)}


def kruskal_wallis(*groups: np.ndarray) -> Dict[str, Any]:
    stat, p = stats.kruskal(*groups)
    return {'statistic': float(stat), 'pvalue': float(p)}


def pairwise_bonferroni(df: pd.DataFrame, group_col: str, value_col: str, paired: bool = False, alpha: float = 0.05) -> Dict[str, Any]:
    groups = df[group_col].unique()
    from itertools import combinations
    pairs = []
    for g1, g2 in combinations(groups, 2):
        v1 = df[df[group_col] == g1][value_col].dropna().values
        v2 = df[df[group_col] == g2][value_col].dropna().values
        res = t_test(v1, v2, paired=paired, equal_var=False)
        pairs.append({'group1': g1, 'group2': g2, 'statistic': res.get('statistic'), 'pvalue': res.get('pvalue')})
    m = len(pairs)
    for p in pairs:
        pval = p['pvalue']
        if pval is None:
            p['p_adj'] = None
            p['significant'] = None
        else:
            adj = min(pval * m, 1.0)
            p['p_adj'] = adj
            p['significant'] = adj < alpha
    return {'method': 'Bonferroni t-tests', 'results': pairs}


def posthoc_auto_select_and_run(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    paired: bool = False,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Decide which post-hoc to use based on data:
      - If all groups normal & equal variances -> Tukey HSD
      - If normal but unequal variances -> Games-Howell
      - If any group non-normal -> Dunn (post Kruskal)
      - If very small n -> Bonferroni pairwise
    Then run and return results.
    """

    # -------------------------
    # Summary stats
    # -------------------------
    groups = df[group_col].unique()
    grp_vals = [
        ensure_numeric_series(df[df[group_col] == g][value_col]).values
        for g in groups
    ]
    n_list = [len(v) for v in grp_vals]

    # Shapiro–Wilk
    sh_pvals = []
    for arr in grp_vals:
        if len(arr) < 3:
            sh_pvals.append(None)
        else:
            _, p = stats.shapiro(arr)
            sh_pvals.append(float(p))

    normals = [p is not None and p >= 0.05 for p in sh_pvals]

    # Levene
    try:
        lev_stat, lev_p = stats.levene(*grp_vals)
        lev = {'stat': float(lev_stat), 'p': float(lev_p)}
    except Exception:
        lev = {'stat': None, 'p': None}

    equal_var_flag = lev['p'] is not None and lev['p'] >= 0.05
    small_n_flag = any(n < 3 for n in n_list)

    # -------------------------
    # Choose post-hoc method
    # -------------------------
    if all(normals) and equal_var_flag and not small_n_flag:
        chosen = 'tukey'
    elif all(normals) and not equal_var_flag and _HAS_PINGOUIN and not small_n_flag:
        chosen = 'gameshowell'
    elif not all(normals) and _HAS_SCIPOST:
        chosen = 'dunn'
    else:
        chosen = 'bonferroni'

    results = {
        'chosen': chosen,
        'shapiro_pvals': sh_pvals,
        'levene': lev,
        'n_list': n_list
    }

    # -------------------------
    # Run selected post-hoc
    # -------------------------
    try:
        if chosen == 'tukey' and _HAS_STATSMODELS:
            data = df[[group_col, value_col]].dropna()
            tuk = pairwise_tukeyhsd(
                endog=data[value_col],
                groups=data[group_col],
                alpha=alpha
            )
            tuk_df = pd.DataFrame(
                tuk._results_table.data[1:],
                columns=tuk._results_table.data[0]
            )
            rows = tuk_df.to_dict(orient='records')
            results['method'] = 'TukeyHSD'

        elif chosen == 'gameshowell' and _HAS_PINGOUIN:
            data = df[[group_col, value_col]].dropna()
            gh = pg.pairwise_gameshowell(
                data=data,
                dv=value_col,
                between=group_col
            )
            rows = gh.to_dict(orient='records')
            results['method'] = 'Games-Howell'

        elif chosen == 'dunn' and _HAS_SCIPOST:
            data = df[[group_col, value_col]].dropna()
            ph = sp.posthoc_dunn(
                data,
                val_col=value_col,
                group_col=group_col,
                p_adjust='bonferroni'
            )

            rows = []
            group_names = list(ph.index)
            for i, g1 in enumerate(group_names):
                for j, g2 in enumerate(group_names):
                    if j <= i:
                        continue
                    pval = float(ph.iloc[i, j])
                    rows.append({
                        'group1': g1,
                        'group2': g2,
                        'p_adj': pval
                    })

            results['method'] = 'Dunn (Bonferroni)'

        else:
            rows = pairwise_bonferroni(
                df,
                group_col,
                value_col,
                paired=paired,
                alpha=alpha
            )['results']
            results['method'] = 'Bonferroni'

    except Exception as e:
        results['error'] = str(e)
        results['results'] = []
        return results

    # -------------------------
    # Normalize p-values + significance
    # -------------------------
    for r in rows:
        p = None
        for key in ('p_adj', 'p-adj', 'pval', 'pvalue'):
            if key in r and r[key] is not None:
                p = float(r[key])
                break

        r['p_adj_raw'] = p
        r['p_adj'] = format_pvalue(p)
        r['significant'] = (p is not None) and (p < alpha)

    results['results'] = rows
    return results



# -------------------------
# helpers
# -------------------------
def ensure_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').dropna()

