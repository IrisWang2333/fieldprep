#!/usr/bin/env python
"""
Sampling Utilities for DH Bundle Selection

Implements the single-layer randomization design:
- Day 1 (Week 1): 24 DH bundles (4 conditional + 20 random), NO D2DS
  - 6 interviewers × 4 bundles each = 24 total
- Day 2+ (Week 2+): 6 DH + 6 D2DS per week
  - DH: 4 conditional + 2 random = 6 total
  - D2DS: 4 from DH conditional + 2 random = 6 total

Conditional = bundle had at least one pothole in preceding week
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def sample_dh_bundles(
    current_date,
    eligible_bundles,
    all_bundles,
    is_day_1=False,
    seed=None
):
    """
    Sample DH bundles according to the design.

    Day 1 (Week 1): 24 DH bundles (4 conditional + 20 random)
    Day 2+ (Week 2+): 6 DH bundles per week (4 conditional + 2 random)

    Parameters
    ----------
    current_date : datetime-like
        Current date
    eligible_bundles : set or list
        Bundle IDs that had potholes in preceding week
    all_bundles : set or list
        All available DH bundle IDs
    is_day_1 : bool
        Whether this is the first day (special sampling)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        {
            'conditional': list of bundle IDs sampled from eligible,
            'random': list of bundle IDs sampled randomly,
            'all_sampled': list of all sampled bundle IDs
        }
    """
    rng = np.random.default_rng(seed)

    eligible_bundles = set(eligible_bundles)
    all_bundles = set(all_bundles)

    # Determine sample sizes
    if is_day_1:
        n_conditional = 4
        n_random = 20  # 6 interviewers × 4 bundles = 24 total (4 cond + 20 random)
        print(f"\n[Day 1 Sampling] Sampling {n_conditional} conditional + {n_random} random DH bundles (6 interviewers × 4 bundles)")
    else:
        n_conditional = 4
        n_random = 2  # 6 interviewers: 4 get 1 DH, 2 get random → 6 total (4 cond + 2 random)
        print(f"\n[Regular Sampling] Sampling {n_conditional} conditional + {n_random} random DH bundles")

    # Sample conditional bundles (from eligible pool)
    if len(eligible_bundles) < n_conditional:
        print(f"  WARNING: Only {len(eligible_bundles)} eligible bundles available, need {n_conditional}")
        print(f"  Sampling all {len(eligible_bundles)} eligible bundles")
        conditional_sample = list(eligible_bundles)
    else:
        conditional_sample = list(rng.choice(
            list(eligible_bundles),
            size=n_conditional,
            replace=False
        ))

    print(f"  Conditional bundles sampled: {len(conditional_sample)}")

    # Sample random bundles (from all bundles, excluding already sampled)
    remaining_bundles = all_bundles - set(conditional_sample)

    if len(remaining_bundles) < n_random:
        print(f"  WARNING: Only {len(remaining_bundles)} bundles remain for random sampling, need {n_random}")
        random_sample = list(remaining_bundles)
    else:
        random_sample = list(rng.choice(
            list(remaining_bundles),
            size=n_random,
            replace=False
        ))

    print(f"  Random bundles sampled: {len(random_sample)}")

    all_sampled = conditional_sample + random_sample

    return {
        'conditional': conditional_sample,
        'random': random_sample,
        'all_sampled': all_sampled,
        'date': current_date,
        'is_day_1': is_day_1
    }


def select_d2ds_bundles(
    conditional_bundles,
    all_bundles,
    bundles_df,
    n_from_conditional=4,
    n_random=2,
    seed=None,
    segment_col='segment_id'
):
    """
    Select D2DS bundles.

    D2DS design:
    - 4 bundles from conditional DH bundles (reuse all 4 DH conditional)
    - 2 additional random bundles (no pothole requirement)

    Total: 6 D2DS bundles

    Parameters
    ----------
    conditional_bundles : list
        List of conditional DH bundle IDs (4 bundles with potholes in preceding week)
    all_bundles : set or list
        All available bundle IDs
    bundles_df : pd.DataFrame
        Bundle-segment mapping
    n_from_conditional : int
        Number to take from conditional (default 4)
    n_random : int
        Number to sample randomly (default 2)
    seed : int, optional
        Random seed
    segment_col : str
        Column name for segment ID in bundles_df

    Returns
    -------
    dict
        {
            'd2ds_conditional': list of bundle IDs from conditional DH,
            'd2ds_random': list of random bundle IDs,
            'd2ds_all': list of all D2DS bundle IDs,
            'd2ds_segments': list of segments in all D2DS bundles
        }
    """
    rng = np.random.default_rng(seed)

    print(f"\n[D2DS Selection] Selecting {n_from_conditional} from conditional + {n_random} random bundles")

    # Take all 4 conditional bundles for D2DS
    if len(conditional_bundles) < n_from_conditional:
        print(f"  WARNING: Only {len(conditional_bundles)} conditional bundles available, need {n_from_conditional}")
        d2ds_conditional = list(conditional_bundles)
    else:
        # Use all 4 conditional bundles
        d2ds_conditional = list(conditional_bundles)[:n_from_conditional]

    print(f"  D2DS from conditional: {len(d2ds_conditional)}")

    # Sample additional random bundles (excluding already selected)
    remaining_bundles = set(all_bundles) - set(d2ds_conditional)

    if len(remaining_bundles) < n_random:
        print(f"  WARNING: Only {len(remaining_bundles)} bundles remain for random D2DS, need {n_random}")
        d2ds_random = list(remaining_bundles)
    else:
        d2ds_random = list(rng.choice(
            list(remaining_bundles),
            size=n_random,
            replace=False
        ))

    print(f"  D2DS random: {len(d2ds_random)}")

    # Combine
    d2ds_all = d2ds_conditional + d2ds_random

    # Get all segments in D2DS bundles
    d2ds_segments = bundles_df[
        bundles_df['bundle_id'].isin(d2ds_all)
    ][segment_col].unique().tolist()

    print(f"  Total D2DS bundles: {len(d2ds_all)}")
    print(f"  Total D2DS segments: {len(d2ds_segments)}")

    return {
        'd2ds_conditional': d2ds_conditional,
        'd2ds_random': d2ds_random,
        'd2ds_all': d2ds_all,
        'd2ds_segments': d2ds_segments
    }


def create_sampling_plan(
    start_date,
    end_date,
    eligible_bundles_by_week,
    all_dh_bundles,
    bundles_df,
    n_d2ds=4,
    seed=None,
    segment_col='segment_id'
):
    """
    Create complete sampling plan for experiment period.

    Parameters
    ----------
    start_date : datetime-like
        Experiment start date (Day 1)
    end_date : datetime-like
        Experiment end date
    eligible_bundles_by_week : dict
        {week_start: set of eligible bundle IDs}
    all_dh_bundles : set or list
        All DH bundle IDs
    bundles_df : pd.DataFrame
        Bundle-segment mapping
    n_d2ds : int
        Number of D2DS bundles per week
    seed : int, optional
        Random seed
    segment_col : str
        Column name for segment ID

    Returns
    -------
    pd.DataFrame
        Sampling plan with columns:
        - date
        - bundle_id
        - bundle_type: 'conditional' or 'random'
        - is_d2ds: whether selected for D2DS
        - week_number: week number in experiment
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    rng = np.random.default_rng(seed)

    # Generate weekly schedule
    current = start_date
    week_num = 0
    plan_records = []

    while current <= end_date:
        week_num += 1
        is_day_1 = (current == start_date)

        # Get eligible bundles for this week
        week_start = get_week_start(current)
        eligible = eligible_bundles_by_week.get(week_start, set())

        print(f"\n{'='*70}")
        print(f"Week {week_num}: {current.date()}")
        print(f"  Eligible bundles (had pothole last week): {len(eligible)}")

        # Sample DH bundles
        dh_sample = sample_dh_bundles(
            current_date=current,
            eligible_bundles=eligible,
            all_bundles=all_dh_bundles,
            is_day_1=is_day_1,
            seed=rng.integers(0, 1e9) if seed is not None else None
        )

        # Add DH bundles to plan
        for bundle_id in dh_sample['conditional']:
            plan_records.append({
                'date': current,
                'week_number': week_num,
                'bundle_id': bundle_id,
                'bundle_type': 'conditional',
                'is_dh': True,
                'is_d2ds': False  # Will update below
            })

        for bundle_id in dh_sample['random']:
            plan_records.append({
                'date': current,
                'week_number': week_num,
                'bundle_id': bundle_id,
                'bundle_type': 'random',
                'is_dh': True,
                'is_d2ds': False
            })

        # Select D2DS bundles (only for Day 2+)
        if not is_day_1:
            d2ds_selection = select_d2ds_bundles(
                conditional_bundles=dh_sample['conditional'],
                all_bundles=all_dh_bundles,
                bundles_df=bundles_df,
                n_from_conditional=4,
                n_random=2,
                seed=rng.integers(0, 1e9) if seed is not None else None,
                segment_col=segment_col
            )

            # Mark conditional D2DS bundles (these come from DH)
            for bundle_id in d2ds_selection['d2ds_conditional']:
                for record in plan_records:
                    if (record['bundle_id'] == bundle_id and
                        record['date'] == current):
                        record['is_d2ds'] = True

            # Add random D2DS bundles (these are NEW bundles, not in DH)
            for bundle_id in d2ds_selection['d2ds_random']:
                plan_records.append({
                    'date': current,
                    'week_number': week_num,
                    'bundle_id': bundle_id,
                    'bundle_type': 'd2ds_random',
                    'is_dh': False,
                    'is_d2ds': True
                })
        else:
            print(f"\n[Day 1] No D2DS selection")

        # Move to next week
        current += timedelta(days=7)

    plan_df = pd.DataFrame(plan_records)

    print(f"\n{'='*70}")
    print(f"SAMPLING PLAN COMPLETE")
    print(f"{'='*70}")
    print(f"Total weeks: {week_num}")
    print(f"Total DH bundle-weeks: {len(plan_df[plan_df['is_dh']])}")
    print(f"  Conditional: {len(plan_df[plan_df['bundle_type']=='conditional'])}")
    print(f"  Random: {len(plan_df[plan_df['bundle_type']=='random'])}")
    print(f"Total D2DS bundle-weeks: {len(plan_df[plan_df['is_d2ds']])}")

    return plan_df


def get_week_start(date):
    """
    Get the Saturday start of the week for a given date.

    Week definition: Saturday (day 0) to Friday (day 6)
    """
    if pd.isna(date):
        return pd.NaT

    date = pd.Timestamp(date)
    days_since_saturday = (date.weekday() + 2) % 7
    week_start = date - timedelta(days=days_since_saturday)

    return pd.Timestamp(week_start.date())


if __name__ == "__main__":
    # Test sampling functions
    print("Testing sampling utilities...")

    # Mock data
    all_bundles = set(range(1, 101))  # 100 bundles
    eligible = set(range(1, 21))  # 20 eligible

    # Test Day 1 sampling
    result = sample_dh_bundles(
        current_date=datetime(2026, 1, 6),
        eligible_bundles=eligible,
        all_bundles=all_bundles,
        is_day_1=True,
        seed=42
    )

    print(f"\n{'='*70}")
    print("Day 1 sample:")
    print(f"  Conditional ({len(result['conditional'])}): {result['conditional']}")
    print(f"  Random ({len(result['random'])}): {result['random'][:10]}...")

    # Test regular sampling
    result2 = sample_dh_bundles(
        current_date=datetime(2026, 1, 13),
        eligible_bundles=eligible,
        all_bundles=all_bundles,
        is_day_1=False,
        seed=43
    )

    print(f"\n{'='*70}")
    print("Regular week sample:")
    print(f"  Conditional ({len(result2['conditional'])}): {result2['conditional']}")
    print(f"  Random ({len(result2['random'])}): {result2['random']}")

    # Test D2DS selection
    mock_bundles_df = pd.DataFrame({
        'bundle_id': list(range(1, 101)) * 2,  # Each bundle has 2 segments
        'segment_id': [f'seg{i}' for i in range(1, 201)]
    })

    d2ds = select_d2ds_bundles(
        conditional_bundles=result2['conditional'],
        all_bundles=all_bundles,
        bundles_df=mock_bundles_df,
        n_from_conditional=4,
        n_random=2,
        seed=44
    )

    print(f"\n{'='*70}")
    print("D2DS selection:")
    print(f"  From conditional: {d2ds['d2ds_conditional']}")
    print(f"  Random: {d2ds['d2ds_random']}")
    print(f"  All D2DS bundles: {d2ds['d2ds_all']}")
    print(f"  Total segments: {len(d2ds['d2ds_segments'])}")
