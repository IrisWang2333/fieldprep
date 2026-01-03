#!/usr/bin/env python
"""
Weekly Plan and Emit Automation Script

Runs plan.py → emit.py for next Saturday's date.
This script is used by GitHub Actions but can also be run locally for testing.

Usage:
    python scripts/weekly_plan_emit.py [--date YYYY-MM-DD] [--week-1]

Arguments:
    --date: Optional date to generate plan for (default: next Saturday)
    --week-1: Flag to indicate this is Week 1 (30 DH bundles, no D2DS)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sd311_fieldprep.plan import run_plan
from sd311_fieldprep.emit import run_emit
from utils.data_fetcher import fetch_latest_notification_activities


def get_next_saturday():
    """Calculate the next Saturday date."""
    today = datetime.now()
    days_until_saturday = (5 - today.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    next_saturday = today + timedelta(days=days_until_saturday)
    return next_saturday.strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser(description='Weekly Plan and Emit Automation')
    parser.add_argument(
        '--date',
        type=str,
        help='Date to generate plan for (YYYY-MM-DD). Default: next Saturday'
    )
    parser.add_argument(
        '--week-1',
        action='store_true',
        help='Flag to indicate this is Week 1 (30 DH bundles, no D2DS)'
    )
    parser.add_argument(
        '--interviewers',
        type=str,
        nargs='+',
        default=['A', 'B', 'C', 'D', 'E', 'F'],
        help='List of interviewer IDs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--bundle-file',
        type=str,
        default='outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet',
        help='Path to bundle file'
    )

    args = parser.parse_args()

    # Determine date
    if args.date:
        date = args.date
        print(f"Using provided date: {date}")
    else:
        date = get_next_saturday()
        print(f"Calculated next Saturday: {date}")

    print(f"\n{'='*70}")
    print(f"WEEKLY PLAN AND EMIT AUTOMATION")
    print(f"{'='*70}")
    print(f"Date: {date}")
    print(f"Week 1: {args.week_1}")
    print(f"Interviewers: {', '.join(args.interviewers)}")
    print(f"Seed: {args.seed}")

    # Step 1: Download latest data
    print(f"\n{'='*70}")
    print("STEP 1: Downloading latest notification activities")
    print(f"{'='*70}")

    activities = fetch_latest_notification_activities(
        use_local=False,
        download_if_missing=True
    )
    print(f"✓ Downloaded {len(activities):,} pothole records")

    # Step 2: Run plan.py
    print(f"\n{'='*70}")
    print("STEP 2: Generating plan")
    print(f"{'='*70}")

    tasks = ('DH', 'D2DS') if not args.week_1 else ('DH',)

    plan_csv = run_plan(
        date=date,
        interviewers=tuple(args.interviewers),
        tasks=tasks,
        list_code=30,
        seed=args.seed,
        is_week_1=args.week_1
    )

    print(f"✓ Plan created: {plan_csv}")

    # Step 3: Run emit.py
    print(f"\n{'='*70}")
    print("STEP 3: Generating field files (emit)")
    print(f"{'='*70}")

    run_emit(
        date=date,
        plan_csv=plan_csv,
        bundle_file=args.bundle_file
    )

    print(f"✓ Emit completed!")

    # Step 4: Summary
    print(f"\n{'='*70}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")

    output_dir = Path(f"outputs/incoming/daily/{date}")
    print(f"\nGenerated files in: {output_dir}")

    if output_dir.exists():
        files = list(output_dir.glob('*'))
        for f in files:
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} bytes)")
    else:
        print(f"  ⚠ Output directory not found: {output_dir}")

    print(f"\n✅ Ready to upload to Google Drive!")
    print(f"Upload directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
