#!/usr/bin/env python
"""
Quick script to simulate 30-day experiment

Usage:
    python tests/quick_simulate_30days.py
"""
from pathlib import Path
import sys
import pandas as pd

# Add src to path
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.simulation.multiday import simulate_multiday_experiment


if __name__ == "__main__":
    print("=" * 70)
    print("30-Day Experiment Simulation")
    print("=" * 70)
    print()

    # Run simulation
    plan, stats, overlap = simulate_multiday_experiment(
        n_days=30,
        n_interviewers=6,
        day1_bundles_per_interviewer=5,
        daily_dh_per_interviewer=1,
        daily_d2ds_from_completed=4,
        daily_d2ds_new=2,
        pothole_file="get_it_done_pothole_requests_datasd.csv",  # Load potholes
        start_date="2026-01-10",
        seed=42
    )

    print()
    print("=" * 70)
    print("✅ Simulation complete!")
    print("=" * 70)

    # ============================================================================
    # Summary Tables
    # ============================================================================
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Convert date to datetime
    plan['date'] = pd.to_datetime(plan['date'])

    # Daily summary
    daily_summary = plan.groupby('date').agg({
        'bundle_id': 'count',
        'task': lambda x: (x == 'D2DS').sum()
    }).reset_index()
    daily_summary.columns = ['date', 'total_assignments', 'd2ds_count']
    daily_summary['dh_count'] = daily_summary['total_assignments'] - daily_summary['d2ds_count']
    daily_summary['day'] = range(1, len(daily_summary) + 1)

    print("\nDaily Summary (first 10 days):")
    print(daily_summary[['day', 'date', 'total_assignments', 'dh_count', 'd2ds_count']].head(10).to_string(index=False))

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    total_days = len(daily_summary)
    total_bundles = plan['bundle_id'].nunique()
    total_dh = (plan['task'] == 'DH').sum()
    total_d2ds = (plan['task'] == 'D2DS').sum()

    print(f"\nExperiment Duration: {total_days} days")
    print(f"Unique Bundles Used: {total_bundles}")
    print(f"\nTask Assignments:")
    print(f"  DH: {total_dh}")
    print(f"  D2DS: {total_d2ds}")
    print(f"  Total: {total_dh + total_d2ds}")

    # Day 1 vs Day 2+ breakdown
    day1_data = plan[plan['date'] == plan['date'].min()]
    day2plus_data = plan[plan['date'] != plan['date'].min()]

    print(f"\nDay 1:")
    print(f"  DH: {(day1_data['task'] == 'DH').sum()}")
    print(f"  D2DS: {(day1_data['task'] == 'D2DS').sum()}")

    if len(day2plus_data) > 0:
        avg_dh_per_day = (day2plus_data['task'] == 'DH').sum() / (total_days - 1)
        avg_d2ds_per_day = (day2plus_data['task'] == 'D2DS').sum() / (total_days - 1)
        print(f"\nDays 2-{total_days} (average per day):")
        print(f"  DH: {avg_dh_per_day:.1f}")
        print(f"  D2DS: {avg_d2ds_per_day:.1f}")

    # Interviewer workload
    print("\n" + "=" * 70)
    print("INTERVIEWER WORKLOAD")
    print("=" * 70)

    interviewer_summary = plan.groupby('interviewer').agg({
        'bundle_id': 'count',
        'task': lambda x: (x == 'DH').sum()
    }).reset_index()
    interviewer_summary.columns = ['interviewer', 'total', 'dh_count']
    interviewer_summary['d2ds_count'] = interviewer_summary['total'] - interviewer_summary['dh_count']

    print("\nAssignments per Interviewer:")
    print(interviewer_summary.to_string(index=False))

    print()
    print("=" * 70)
    print("Output files:")
    print("  - outputs/simulation/plan_30days.csv")
    print("  - outputs/simulation/stats_30days.csv")
    print("  - outputs/simulation/overlap_30days.csv")
    print()
