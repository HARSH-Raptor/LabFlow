# test_lab_randomizer.py
import numpy as np
import pandas as pd
from lab_randomizer import (
    generate_valid_pairs,
    compute_error_distribution,
    error_propagation_summary_uniform,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_grubbs,
    remove_outliers,
    posthoc_auto_select_and_run,
    format_pvalue,
    ensure_numeric_series
)


def test_generate_pairs_basic():
    s = ['A', 'B']
    c = ['C1', 'C2']
    pairs = generate_valid_pairs(s, c)
    assert set(pairs) == {'AC1', 'AC2', 'BC1', 'BC2'}


def test_error_propagation_uniform():
    res = error_propagation_summary_uniform(4, 0.10)
    expected_worst = 1.1 ** 4
    expected_best = 0.9 ** 4
    assert abs(res['worst_mult'] - expected_worst) < 1e-12
    assert abs(res['best_mult'] - expected_best) < 1e-12


def test_compute_error_distribution_enum():
    vlist = [0.10, 0.08, 0.03, 0.05]
    res = compute_error_distribution(vlist, enumerate_limit=2 ** 10, mc_samples=1000, random_seed=1)
    assert res['n'] == 4
    assert res['combinations'] == 16
    assert res['mode'] == 'enumeration'
    assert 'samples' in res
    assert res['worst_mult'] > res['best_mult']


def test_outlier_detectors():
    arr = np.array([1.0, 1.1, 0.9, 50.0])
    iqr_mask = detect_outliers_iqr(arr)
    z_mask = detect_outliers_zscore(arr)
    grubbs_mask = detect_outliers_grubbs(arr)
    assert any([iqr_mask[3], z_mask[3], grubbs_mask[3]])


def test_remove_outliers():
    df = pd.DataFrame({'G': ['A'] * 4, 'V': [1.0, 1.1, 0.9, 50.0]})
    cleaned, report = remove_outliers(df, 'G', 'V', method='IQR')
    assert 'total_removed' in report
    assert isinstance(report['total_removed'], int)


def test_posthoc_auto_select():
    # simple dataset with clear differences
    df = pd.DataFrame({'G': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
                       'V': [10, 9.8, 10.2, 9.7, 10.1, 18, 17.8, 18.1, 17.9, 18.2, 26, 25.8, 26.2, 25.9, 26.1]})
    res = posthoc_auto_select_and_run(df, 'G', 'V')
    assert 'chosen' in res
    assert 'results' in res


def test_format_pvalue():
    assert format_pvalue(0.5) == "0.500000"
    assert "e" in format_pvalue(1e-20)
    assert format_pvalue(None) == "NA"


def test_ensure_numeric_series():
    s = pd.Series(['1.0', '2.1', 'bad', None])
    cleaned = ensure_numeric_series(s)
    assert all(isinstance(x, (int, float, np.floating, np.integer)) for x in cleaned)
