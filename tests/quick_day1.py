#!/usr/bin/env python
"""
Quick script to generate Day 1 plan

Usage:
    python tests/quick_day1.py
"""
from pathlib import Path
import sys

# Add src to path
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.simulation.day1 import generate_day1_plan


if __name__ == "__main__":
    print("=" * 70)
    print("Day 1 Plan Generator")
    print("=" * 70)
    print()

    # Generate day 1 plan
    plan = generate_day1_plan(
        date="2025-01-06",           # Monday, January 6, 2025
        n_interviewers=6,            # A, B, C, D, E, F
        bundles_per_interviewer=5,   # 5 DH bundles each
        seed=42                      # Reproducible random selection
    )

    print()
    print("=" * 70)
    print("✅ Day 1 plan generated successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review: outputs/plans/day1_2025-01-06_plan.csv")
    print("  2. Generate daily files: run emit with this plan")
    print("  3. Example:")
    print("     from sd311_fieldprep.emit import run_emit")
    print("     run_emit(")
    print("         date='2025-01-06',")
    print("         plan_csv='outputs/plans/day1_2025-01-06_plan.csv',")
    print("         addr_assignment_file='outputs/sweep/locked/segment_addresses_b40_m2.parquet'")
    print("     )")
