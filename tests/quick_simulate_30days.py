#!/usr/bin/env python
"""
Quick script to simulate 30-day experiment

Usage:
    python tests/quick_simulate_30days.py
"""
from pathlib import Path
import sys

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
    print()
    print("Output files:")
    print("  - outputs/simulation/plan_30days.csv")
    print("  - outputs/simulation/stats_30days.csv")
    print("  - outputs/simulation/overlap_30days.csv")
    print()
    print("To view overlap analysis:")
    print("  import pandas as pd")
    print("  overlap = pd.read_csv('outputs/simulation/overlap_30days.csv')")
    print("  print(overlap)")
