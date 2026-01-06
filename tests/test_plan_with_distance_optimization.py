#!/usr/bin/env python
"""
Test plan.py with distance-based assignment optimization.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sd311_fieldprep.plan import run_plan

# Test with Jan 10, 2026 (Week 1 of official experiment)
plan_csv = run_plan(
    date="2026-01-10",
    interviewers=("Veronica", "Rene", "David", "Jessica D.", "Jessica A.", "Carlie"),
    tasks=("DH", "D2DS"),
    list_code=30,
    seed=42,
    bundle_file="outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet",
    is_week_1=True,  # Week 1: 24 DH bundles, no D2DS
    official_start_date="2026-01-10"
)

print(f"\n✅ Plan created successfully: {plan_csv}")
print("\nCheck the plan CSV to see distance-optimized assignments!")
