#!/usr/bin/env python
"""
Balance Check Module for Simulation

Performs three levels of balance checks:
1. Treatment vs Control (DH treated vs DH control within all DH bundles)
2. Survey sample (DH treated vs DH control within D2DS bundles)
3. Survey representativeness (D2DS vs non-D2DS)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_segment_demographics(demo_file: Path = None):
    """
    Load segment-level demographics data.

    Args:
        demo_file: Path to demographics parquet file

    Returns:
        DataFrame with segment_id and demographic variables
    """
    if demo_file is None:
        # Default location
        from sd311_fieldprep.utils import paths
        root, cfg, out_root = paths()
        demo_file = out_root / "segments_with_demographics.parquet"

    if not demo_file.exists():
        raise FileNotFoundError(
            f"Demographics file not found: {demo_file}\n"
            f"Run: python tests/generate_segment_demographics.py"
        )

    demo = pd.read_parquet(demo_file)

    # Ensure segment_id is string
    demo['segment_id'] = demo['segment_id'].astype(str)

    return demo


def calculate_balance_stats(group1_values, group2_values, var_name):
    """
    Calculate balance statistics for two groups.

    Args:
        group1_values: Array of values for group 1
        group2_values: Array of values for group 2
        var_name: Variable name

    Returns:
        Dictionary with balance statistics
    """
    # Remove NaN values
    g1 = group1_values[~np.isnan(group1_values)]
    g2 = group2_values[~np.isnan(group2_values)]

    if len(g1) == 0 or len(g2) == 0:
        return {
            'variable': var_name,
            'group1_mean': np.nan,
            'group2_mean': np.nan,
            'difference': np.nan,
            'std_diff': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'group1_n': len(g1),
            'group2_n': len(g2),
        }

    # Means
    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    diff = mean1 - mean2

    # Standardized difference (Cohen's d)
    pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
    std_diff = diff / pooled_std if pooled_std > 0 else 0

    # T-test
    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)

    return {
        'variable': var_name,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'difference': diff,
        'std_diff': std_diff,
        't_stat': t_stat,
        'p_value': p_value,
        'group1_n': len(g1),
        'group2_n': len(g2),
    }


def balance_check_treatment_vs_control(
    dh_bundles,
    treatment_results,
    bundle_segments,
    demographics
):
    """
    Level 1: Compare DH treated vs DH control across all DH bundles.

    Args:
        dh_bundles: Set of DH bundle IDs
        treatment_results: Dict with 'control', 'full', 'partial' segment sets
        bundle_segments: Dict mapping bundle_id -> set of segment_ids
        demographics: DataFrame with segment demographics

    Returns:
        DataFrame with balance statistics
    """
    # Get segments in DH bundles
    dh_segments = set()
    for bid in dh_bundles:
        if bid in bundle_segments:
            dh_segments.update(bundle_segments[bid])

    # Convert to strings
    dh_segments = {str(s) for s in dh_segments}

    # Get treated segments (full + partial)
    treated_segs = (
        treatment_results['full']['segments'] |
        treatment_results['partial']['segments']
    )
    treated_segs = {str(s) for s in treated_segs}

    # Get control segments
    control_segs = treatment_results['control']['segments']
    control_segs = {str(s) for s in control_segs}

    # Filter to segments in demographics
    treated_segs_demo = treated_segs & set(demographics['segment_id'])
    control_segs_demo = control_segs & set(demographics['segment_id'])

    # Get demographic values
    treated_demo = demographics[demographics['segment_id'].isin(treated_segs_demo)]
    control_demo = demographics[demographics['segment_id'].isin(control_segs_demo)]

    # Calculate balance for each variable
    variables = {
        'PCI': 'pci',
        'Per Capita Income': 'per_capita_income',
        'Share College+': 'share_college',
        'Share Hispanic': 'share_hispanic',
        'Share White (NH)': 'share_white_nh',
        'Share Black (NH)': 'share_black_nh',
        'Share Asian (NH)': 'share_asian_nh',
    }

    results = []
    for display_name, col_name in variables.items():
        if col_name not in demographics.columns:
            continue

        stat = calculate_balance_stats(
            treated_demo[col_name].values,
            control_demo[col_name].values,
            display_name
        )
        stat['comparison'] = 'DH Treated vs DH Control (All DH bundles)'
        stat['group1_label'] = 'DH Treated'
        stat['group2_label'] = 'DH Control'
        results.append(stat)

    return pd.DataFrame(results)


def balance_check_survey_sample(
    d2ds_bundles,
    dh_bundles,
    treatment_results,
    bundle_segments,
    demographics
):
    """
    Level 2: Compare DH treated vs DH control within D2DS bundles only.

    This checks whether the survey sample (D2DS addresses) has balanced
    treatment assignment.

    Args:
        d2ds_bundles: Set of D2DS bundle IDs
        dh_bundles: Set of DH bundle IDs
        treatment_results: Dict with 'control', 'full', 'partial' segment sets
        bundle_segments: Dict mapping bundle_id -> set of segment_ids
        demographics: DataFrame with segment demographics

    Returns:
        DataFrame with balance statistics
    """
    # Get segments in bundles that received BOTH DH and D2DS
    both_bundles = d2ds_bundles & dh_bundles

    both_segments = set()
    for bid in both_bundles:
        if bid in bundle_segments:
            both_segments.update(bundle_segments[bid])

    both_segments = {str(s) for s in both_segments}

    # Get treated and control segments, filtered to "both" bundles
    treated_segs = (
        treatment_results['full']['segments'] |
        treatment_results['partial']['segments']
    )
    treated_segs = {str(s) for s in treated_segs} & both_segments

    control_segs = treatment_results['control']['segments']
    control_segs = {str(s) for s in control_segs} & both_segments

    # Filter to segments in demographics
    treated_segs_demo = treated_segs & set(demographics['segment_id'])
    control_segs_demo = control_segs & set(demographics['segment_id'])

    # Get demographic values
    treated_demo = demographics[demographics['segment_id'].isin(treated_segs_demo)]
    control_demo = demographics[demographics['segment_id'].isin(control_segs_demo)]

    # Calculate balance for each variable
    variables = {
        'PCI': 'pci',
        'Per Capita Income': 'per_capita_income',
        'Share College+': 'share_college',
        'Share Hispanic': 'share_hispanic',
        'Share White (NH)': 'share_white_nh',
        'Share Black (NH)': 'share_black_nh',
        'Share Asian (NH)': 'share_asian_nh',
    }

    results = []
    for display_name, col_name in variables.items():
        if col_name not in demographics.columns:
            continue

        stat = calculate_balance_stats(
            treated_demo[col_name].values,
            control_demo[col_name].values,
            display_name
        )
        stat['comparison'] = 'DH Treated vs DH Control (D2DS survey sample only)'
        stat['group1_label'] = 'DH Treated (in D2DS)'
        stat['group2_label'] = 'DH Control (in D2DS)'
        results.append(stat)

    return pd.DataFrame(results)


def balance_check_survey_representativeness(
    d2ds_bundles,
    all_bundles,
    bundle_segments,
    demographics
):
    """
    Level 3: Compare D2DS vs non-D2DS segments.

    This checks whether the survey sample is representative of the
    overall population.

    Args:
        d2ds_bundles: Set of D2DS bundle IDs
        all_bundles: Set of all bundle IDs
        bundle_segments: Dict mapping bundle_id -> set of segment_ids
        demographics: DataFrame with segment demographics

    Returns:
        DataFrame with balance statistics
    """
    # Get segments in D2DS bundles
    d2ds_segments = set()
    for bid in d2ds_bundles:
        if bid in bundle_segments:
            d2ds_segments.update(bundle_segments[bid])

    d2ds_segments = {str(s) for s in d2ds_segments}

    # Get segments NOT in D2DS bundles
    non_d2ds_bundles = all_bundles - d2ds_bundles
    non_d2ds_segments = set()
    for bid in non_d2ds_bundles:
        if bid in bundle_segments:
            non_d2ds_segments.update(bundle_segments[bid])

    non_d2ds_segments = {str(s) for s in non_d2ds_segments}

    # Filter to segments in demographics
    d2ds_segs_demo = d2ds_segments & set(demographics['segment_id'])
    non_d2ds_segs_demo = non_d2ds_segments & set(demographics['segment_id'])

    # Get demographic values
    d2ds_demo = demographics[demographics['segment_id'].isin(d2ds_segs_demo)]
    non_d2ds_demo = demographics[demographics['segment_id'].isin(non_d2ds_segs_demo)]

    # Calculate balance for each variable
    variables = {
        'PCI': 'pci',
        'Per Capita Income': 'per_capita_income',
        'Share College+': 'share_college',
        'Share Hispanic': 'share_hispanic',
        'Share White (NH)': 'share_white_nh',
        'Share Black (NH)': 'share_black_nh',
        'Share Asian (NH)': 'share_asian_nh',
    }

    results = []
    for display_name, col_name in variables.items():
        if col_name not in demographics.columns:
            continue

        stat = calculate_balance_stats(
            d2ds_demo[col_name].values,
            non_d2ds_demo[col_name].values,
            display_name
        )
        stat['comparison'] = 'Survey Representativeness (D2DS vs non-D2DS)'
        stat['group1_label'] = 'D2DS'
        stat['group2_label'] = 'Non-D2DS'
        results.append(stat)

    return pd.DataFrame(results)


def run_all_balance_checks(
    dh_bundles,
    d2ds_bundles,
    all_bundles,
    treatment_results,
    bundle_segments,
    demographics_file=None
):
    """
    Run all three levels of balance checks.

    Args:
        dh_bundles: Set of DH bundle IDs
        d2ds_bundles: Set of D2DS bundle IDs
        all_bundles: Set of all bundle IDs used
        treatment_results: Dict with 'control', 'full', 'partial' containing
                          {'addresses': list, 'segments': set}
        bundle_segments: Dict mapping bundle_id -> set of segment_ids
        demographics_file: Path to demographics parquet (optional)

    Returns:
        DataFrame with all balance check results
    """
    # Load demographics
    demographics = load_segment_demographics(demographics_file)

    print(f"[balance] Loaded demographics for {len(demographics):,} segments")

    # Run three levels of checks
    print(f"[balance] Running balance checks...")

    level1 = balance_check_treatment_vs_control(
        dh_bundles, treatment_results, bundle_segments, demographics
    )

    level2 = balance_check_survey_sample(
        d2ds_bundles, dh_bundles, treatment_results, bundle_segments, demographics
    )

    level3 = balance_check_survey_representativeness(
        d2ds_bundles, all_bundles, bundle_segments, demographics
    )

    # Combine all results
    all_results = pd.concat([level1, level2, level3], ignore_index=True)

    # Reorder columns
    col_order = [
        'comparison', 'variable',
        'group1_label', 'group1_mean', 'group1_n',
        'group2_label', 'group2_mean', 'group2_n',
        'difference', 'std_diff', 't_stat', 'p_value'
    ]
    all_results = all_results[col_order]

    print(f"[balance] Completed {len(all_results)} balance comparisons")

    return all_results
