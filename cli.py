#!/usr/bin/env python
import argparse, sys
from pathlib import Path

# Ensure ./src is importable without packaging
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.sweep import run_sweep
from sd311_fieldprep.bundle_hard_constraint import run_bundle
from sd311_fieldprep.emit import run_emit
from sd311_fieldprep.doctor import run_doctor
from sd311_fieldprep.plan import run_plan
from sd311_fieldprep.export_crosswalk import export_address_crosswalk


def main():
    ap = argparse.ArgumentParser(prog="fieldprep", description="Eligibility sweep, bundles, and Tech‑Brief emits")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # plan
    p0 = sub.add_parser("plan", help="Pick bundles at random (with optional CPD/SFH filters) to create bundles_plan_date_<date>.csv")
    p0.add_argument("--date", required=True, help="YYYY-MM-DD")
    p0.add_argument("--cpd", default=None, help="Restrict to this CPD name (uses CPD layer + name_field)")
    p0.add_argument("--sfh_min", type=int, default=None, help="Keep bundles with at least this many SFH")
    p0.add_argument("--sfh_max", type=int, default=None, help="Keep bundles with at most this many SFH")
    p0.add_argument("--interviewers", nargs="+", default=list("ABCD"), help="Interviewer codes (default: A B C D)")
    p0.add_argument("--tasks", nargs="+", default=["DH","D2DS"], help="Tasks to plan (default: DH D2DS)")
    p0.add_argument("--list_code", type=int, default=30)
    p0.add_argument("--seed", type=int, default=42)
    p0.add_argument("--out", default=None, help="Optional explicit output CSV path")
    p0.add_argument("--bundle-file", default=None,
                    help="Optional: path to shared bundle file for all tasks "
                         "(e.g., outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet)")
    p0.set_defaults(func=lambda a: run_plan(
        date=a.date, interviewers=a.interviewers, tasks=a.tasks, cpd=a.cpd,
        sfh_min=a.sfh_min, sfh_max=a.sfh_max, list_code=a.list_code, seed=a.seed,
        out_csv=a.out, bundle_file=a.bundle_file
    ))

    p1 = sub.add_parser("sweep", help="Sweep eligibility over buffers & min SFH and render maps")
    p1.add_argument("--buffers", nargs="+", required=True, help="e.g., 15 25 35")
    p1.add_argument("--mins", nargs="+", required=True, help="e.g., 4 6 8 10")
    p1.add_argument("--tag", default="default", help="Label for outputs/sweep/<tag>")
    p1.set_defaults(func=lambda a: run_sweep(buffers=[int(x) for x in a.buffers],
                                             mins=[int(x) for x in a.mins],
                                             tag=a.tag))

    # bundle
    p2 = sub.add_parser("bundle", help="Build connected bundles (DH/D2DS)")
    p2.add_argument("--session", choices=["DH", "D2DS"], required=True,
                    help="DH=2–3pm, D2DS=3–7pm")
    p2.add_argument("--target_addrs", type=int, required=True,
                    help="Target # of doors per bundle")
    p2.add_argument("--join_tol_m", type=float, default=15.0,
                    help="Endpoint snapping tolerance (m)")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument(
        "--tag",
        default=None,
        help=("Sweep tag folder to read eligible_*.parquet from (e.g., 'locked'). "
              "If omitted, scans all outputs/sweep/ and picks the most recent.")
    )
    p2.add_argument("--min_bundle_sfh", type=int, default=None,
                    help="Merge bundles smaller than this SFH total into the nearest bundle")
    p2.add_argument("--method", choices=["greedy", "multi_bfs"], default="greedy",
                    help="Bundling algorithm: 'greedy' (default, fast) or 'multi_bfs' (balanced)")

    p2.set_defaults(func=lambda a: run_bundle(
        session=a.session,
        target_addrs=a.target_addrs,
        join_tol_m=a.join_tol_m,
        seed=a.seed,
        tag=a.tag,
        min_bundle_sfh=a.min_bundle_sfh,
        method=a.method
    ))

    p3 = sub.add_parser("emit", help="Emit Tech-Brief daily inputs for a given plan")
    p3.add_argument("--date", required=True, help="YYYY-MM-DD")
    p3.add_argument("--plan", required=False, default=None,
                    help="CSV with date,interviewer,task,bundle_id,list_code "
                         "(defaults to outputs/plans/bundles_plan_<date>.csv)")
    p3.add_argument("--bundle-file", default=None,
                    help="Optional: path to shared bundle file for all tasks "
                         "(e.g., outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet)")
    p3.add_argument("--addr-assignment", default=None,
                    help="Optional: path to segment-to-address assignment file from sweep "
                         "(e.g., outputs/sweep/locked/segment_addresses_b40_m2.parquet). "
                         "If provided, uses exact sweep assignments instead of re-running spatial join.")
    def _emit(a):
        from sd311_fieldprep.utils import paths
        _, _, out_root = paths()
        default_plan = out_root / "plans" / f"bundles_plan_{a.date}.csv"
        plan_csv = a.plan or str(default_plan)
        return run_emit(date=a.date, plan_csv=plan_csv, bundle_file=a.bundle_file,
                       addr_assignment_file=a.addr_assignment)
    p3.set_defaults(func=_emit)

    p4 = sub.add_parser("doctor", help="Print CRS/bounds/filter and nearest-join diagnostics")
    p4.add_argument("--buffer", type=int, default=40)
    p4.add_argument("--mins", type=int, default=5)
    p4.set_defaults(func=lambda a: run_doctor(buffer_m=a.buffer, mins=a.mins))

    p5 = sub.add_parser(
    "export-crosswalk",
    help="Export addr_id ↔ address crosswalk to CSV (and optional HTML map)"
    )

    p5.add_argument(
        "--out",
        required=True,
        help="Output CSV path, e.g. outputs/crosswalk/address_crosswalk.csv",
    )

    p5.add_argument(
        "--map-html",
        help="Optional HTML map output path, e.g. outputs/crosswalk/address_crosswalk_map.html",
    )

    p5.set_defaults(
        func=lambda a: export_address_crosswalk(
            out_csv=a.out,
            map_html=a.map_html,
        )
    )


    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
