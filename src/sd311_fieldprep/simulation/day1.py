#!/usr/bin/env python
"""
Day 1 Plan Generator - DH Only

Generate day 1 field plan:
- 6 interviewers (A, B, C, D, E, F)
- 5 DH bundles per interviewer
- Total: 30 DH bundles
- No D2DS tasks on day 1
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


def generate_day1_plan(
    date: str = "2025-01-06",  # Monday
    n_interviewers: int = 6,
    bundles_per_interviewer: int = 5,
    bundle_file: str = "outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet",
    output_file: str = None,
    list_code: int = 30,
    seed: int = 42,
):
    """
    Generate day 1 plan with DH bundles only.

    Args:
        date: Date for day 1 (YYYY-MM-DD)
        n_interviewers: Number of interviewers (default: 6)
        bundles_per_interviewer: Bundles per interviewer (default: 5)
        bundle_file: Path to DH bundles parquet
        output_file: Output plan CSV path (default: outputs/plans/day1_{date}_plan.csv)
        list_code: List code for all assignments (default: 30)
        seed: Random seed for bundle selection
    """
    from sd311_fieldprep.utils import paths

    root, cfg, out_root = paths()

    # Load DH bundles
    bundle_path = Path(bundle_file)
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_file

    if not bundle_path.exists():
        raise FileNotFoundError(f"DH bundle file not found: {bundle_path}")

    bundles = gpd.read_parquet(bundle_path)

    # Get unique bundle IDs
    bundle_ids = sorted(bundles['bundle_id'].unique())
    print(f"[day1] Found {len(bundle_ids)} DH bundles")

    # Calculate total bundles needed
    total_bundles_needed = n_interviewers * bundles_per_interviewer

    if len(bundle_ids) < total_bundles_needed:
        raise ValueError(
            f"Not enough bundles! Need {total_bundles_needed} but only have {len(bundle_ids)}"
        )

    # Randomly select bundles
    rng = np.random.default_rng(seed)
    selected_bundles = rng.choice(bundle_ids, size=total_bundles_needed, replace=False)
    selected_bundles = sorted(selected_bundles)

    print(f"[day1] Selected {total_bundles_needed} bundles for {n_interviewers} interviewers")

    # Generate interviewer labels
    interviewer_labels = [chr(ord('A') + i) for i in range(n_interviewers)]

    # Create plan rows
    rows = []
    for i, interviewer in enumerate(interviewer_labels):
        # Get this interviewer's bundles
        start_idx = i * bundles_per_interviewer
        end_idx = start_idx + bundles_per_interviewer
        interviewer_bundles = selected_bundles[start_idx:end_idx]

        for bundle_id in interviewer_bundles:
            rows.append({
                "date": date,
                "interviewer": interviewer,
                "task": "DH",
                "bundle_id": int(bundle_id),
                "list_code": list_code,
            })

        print(f"[day1]   {interviewer}: bundles {list(interviewer_bundles)}")

    # Create DataFrame
    plan = pd.DataFrame(rows)

    # Set output path
    if output_file is None:
        output_file = out_root / "plans" / f"day1_{date}_plan.csv"
    else:
        output_file = Path(output_file)
        if not output_file.is_absolute():
            output_file = root / output_file

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write plan
    plan.to_csv(output_file, index=False)
    print(f"[day1] Wrote day 1 plan to {output_file}")

    # Print summary
    print(f"\n[day1] Day 1 Plan Summary:")
    print(f"  Date: {date}")
    print(f"  Interviewers: {n_interviewers}")
    print(f"  Bundles per interviewer: {bundles_per_interviewer}")
    print(f"  Total bundles: {len(plan)}")
    print(f"  Task breakdown:")
    print(plan.groupby('task').size())

    return plan


if __name__ == "__main__":
    # Generate day 1 plan
    plan = generate_day1_plan(
        date="2025-01-06",
        n_interviewers=6,
        bundles_per_interviewer=5,
        seed=42
    )
