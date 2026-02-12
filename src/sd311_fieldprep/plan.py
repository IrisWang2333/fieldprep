# fieldprep/src/sd311_fieldprep/plan.py
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import sys
from datetime import datetime, timedelta

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sd311_fieldprep.utils import (
    paths,
    ensure_dir,
    load_filters_layers,
    project_to,
    load_sources,
    apply_spatial_filters,
    addressable_mask,
)
from sd311_fieldprep.emit import _compose_address

# Import new data fetching and sampling utilities
from utils.data_fetcher import (
    fetch_latest_notification_activities,
    get_eligible_bundles_for_date
)
from utils.sampling import (
    sample_dh_bundles,
    select_d2ds_bundles,
    get_week_start
)
from utils.bundle_tracker import (
    get_used_bundles,
    get_previous_week_conditional_bundles,
    filter_available_bundles,
    print_usage_summary
)

# Import nearest-then-refine (best performing algorithm: 53.6% better than baseline)
from sd311_fieldprep.assign_bundles_nearest_then_refine import (
    assign_bundles_for_date_nearest_then_refine
)
from sd311_fieldprep.assign_bundles_minimax import (
    get_interviewers_for_date
)


def _endpoints_xy(geom):
    # robust to LineString / MultiLineString
    try:
        coords = np.asarray(geom.coords)
    except Exception:
        from shapely.ops import linemerge
        m = linemerge(geom)
        try:
            coords = np.asarray(m.coords)
        except Exception:
            return None, None
    if len(coords) < 2:
        return None, None
    a = (float(coords[0][0]), float(coords[0][1]))
    b = (float(coords[-1][0]), float(coords[-1][1]))
    return a, b


def _route_connected_by_streets(streets_m, seg_id_col, bundle_seg_ids, snap_tol=0.5):
    """
    True iff all street segments in bundle_seg_ids form a single connected component
    when nodes are segment endpoints (snapped by snap_tol meters), using streets_m.
    Mirrors route.py's graph construction.
    """
    sub = streets_m.loc[streets_m[seg_id_col].isin(bundle_seg_ids)]
    if sub.empty:
        return False

    def snap(pt):
        return (round(pt[0] / snap_tol) * snap_tol, round(pt[1] / snap_tol) * snap_tol)

    G = nx.Graph()
    for geom in sub.geometry:
        a, b = _endpoints_xy(geom)
        if a is None or b is None:
            continue
        G.add_edge(snap(a), snap(b))

    return G.number_of_edges() > 0 and nx.number_connected_components(G) == 1


def _route_style_connected(segments_gdf, bundle_id, snap_tol=0.5):
    """
    True iff all segments in this bundle form a single connected component
    when nodes are segment endpoints (snapped by snap_tol meters).
    Mirrors route.py’s graph.
    """
    sub = segments_gdf.loc[segments_gdf["bundle_id"] == bundle_id]
    if sub.empty:
        return False

    def snap(pt):
        return (round(pt[0] / snap_tol) * snap_tol, round(pt[1] / snap_tol) * snap_tol)

    G = nx.Graph()
    for geom in sub.geometry:
        a, b = _endpoints_xy(geom)
        if a is None or b is None:
            continue
        G.add_edge(snap(a), snap(b))
    return G.number_of_edges() > 0 and nx.number_connected_components(G) == 1


# Fallback candidates if filters.cpd.name_field isn't set in params.yaml
_CPD_NAME_CANDIDATES = ["COMMUNITY", "COMMPLAN", "CMNTY_PLN", "CPD", "CPD_NAME", "NAME", "COMM_PLAN"]


def _get_cpd_name_field(cpd_gdf, pr):
    namef = (pr or {}).get("filters", {}).get("cpd", {}).get("name_field", "")
    if namef and namef in cpd_gdf.columns:
        return namef
    for c in _CPD_NAME_CANDIDATES:
        if c in cpd_gdf.columns:
            return c
    raise SystemExit(
        "CPD filter requested but I can't find a name field in the CPD layer. "
        "Set filters.cpd.name_field in params.yaml."
    )


def _attach_bundle_cpd(bundles_m: gpd.GeoDataFrame, cpd_m: gpd.GeoDataFrame, name_field: str) -> gpd.GeoDataFrame:
    g = bundles_m.copy()
    # Assign each segment to a CPD by centroid ∈ CPD polygon; then give each bundle its modal CPD
    cent = g[["bundle_id", "geometry"]].copy()
    cent["geometry"] = cent.geometry.centroid
    joined = gpd.sjoin(cent, cpd_m[[name_field, "geometry"]], how="left", predicate="intersects")
    top = (
        joined.groupby("bundle_id")[name_field]
        .agg(lambda s: s.dropna().value_counts().index[0] if s.dropna().size else None)
        .rename("bundle_cpd")
    )
    return g.merge(top, left_on="bundle_id", right_index=True, how="left")


def _load_bundles(session: str, bundle_file: str | None = None) -> gpd.GeoDataFrame:
    """
    Load bundles for a session.

    Args:
        session: Session name (DH, D2DS)
        bundle_file: Optional path to shared bundle file.
                     If provided, all sessions use this file.
                     If None, load from outputs/bundles/{session}/bundles.parquet
    """
    root, _, out_root = paths()

    if bundle_file:
        # Use shared bundle file for all sessions
        bpath = Path(bundle_file)
        if not bpath.is_absolute():
            bpath = root / bundle_file
    else:
        # Use session-specific bundle file
        bpath = root / "outputs" / "bundles" / session.upper() / "bundles.parquet"

    if not bpath.exists():
        raise SystemExit(f"Missing bundles parquet: {bpath}")

    g = gpd.read_parquet(bpath)
    return g


def run_plan(
    date: str,
    interviewers: list[str] | tuple[str, ...] = ("A", "B", "C", "D"),
    tasks: list[str] | tuple[str, ...] = ("DH", "D2DS"),
    cpd: str | None = None,
    sfh_min: int | None = None,
    sfh_max: int | None = None,
    list_code: int = 30,
    seed: int = 42,
    out_csv: str | None = None,
    bundle_file: str | None = "outputs/bundles/DH/bundles_multibfs_regroup_filtered_length_3.parquet",
    is_week_1: bool = False,
):
    """
    Generate daily plan using conditional sampling based on historical pothole data.

    New Design (Single Layer Randomization):
    - Week 1 (is_week_1=True): 24 DH bundles (4 conditional + 20 random), NO D2DS
      - 6 interviewers × 4 bundles each = 24 total
    - Week 2+ (is_week_1=False): 6 DH (4 conditional + 2 random) + 6 D2DS (4 from DH conditional + 2 random)
      - Each interviewer gets 1 DH + 1 D2DS

    Conditional = bundle had at least one pothole in preceding week (based on latest data)

    Experiment Phases:
    - Pilot: 2025-12-27 to 2026-01-03 (separate bundle pool with within-phase without-replacement)
    - Official: 2026-01-10 onwards (separate bundle pool with within-phase without-replacement)

    Writes outputs/plans/bundles_plan_<date>.csv by default.

    Args:
        date: Date string (YYYY-MM-DD) for the plan
        interviewers: List of interviewer codes (default: A, B, C, D)
        tasks: List of task types (default: DH, D2DS)
        cpd: Optional CPD (Community Planning District) name filter
        sfh_min: Optional minimum SFH addresses per bundle
        sfh_max: Optional maximum SFH addresses per bundle
        list_code: List code for the plan (default: 30)
        seed: Random seed for reproducible bundle selection
        out_csv: Optional explicit output CSV path
        bundle_file: Optional path to shared bundle file for all tasks.
                     Default: "outputs/bundles/DH/bundles_multibfs_regroup_filtered_length_3.parquet"
                     This is the length-filtered version (bundles < 3 km total length).
                     Both DH and D2DS will use this file instead of task-specific bundles.
                     This ensures both tasks use the same filtered/validated bundle set
                     without overly long bundles that may be difficult to complete.
                     Set to None to use task-specific bundles from
                     outputs/bundles/{task}/bundles.parquet instead.
        is_week_1: Whether this is Week 1 (special sampling: 24 DH, no D2DS)
    """
    root, _, out_root = paths()
    rng = np.random.default_rng(int(seed))

    # ============================================================================
    # Determine experiment phase based on date
    # ============================================================================
    current_date = datetime.strptime(date, "%Y-%m-%d")

    # Define phase date ranges
    pilot_start = datetime.strptime("2025-12-27", "%Y-%m-%d")
    pilot_end = datetime.strptime("2026-01-03", "%Y-%m-%d")
    official_start = datetime.strptime("2026-01-10", "%Y-%m-%d")

    if pilot_start <= current_date <= pilot_end:
        experiment_phase = "pilot"
        phase_min_date = "2025-12-27"
        phase_max_date = "2026-01-03"
    elif current_date >= official_start:
        experiment_phase = "official"
        phase_min_date = "2026-01-10"
        phase_max_date = None  # No upper limit for official
    else:
        raise ValueError(
            f"Date {date} is not in a valid experiment phase. "
            f"Valid ranges: Pilot (2025-12-27 to 2026-01-03), Official (2026-01-10+)"
        )

    print(f"\n{'='*70}")
    print(f"GENERATING PLAN FOR {date}")
    print(f"Experiment Phase: {experiment_phase.upper()}")
    print(f"Phase Date Range: {phase_min_date} to {phase_max_date or 'ongoing'}")
    print(f"{'='*70}")

    # Parse date and load latest pothole activities
    current_date = datetime.strptime(date, "%Y-%m-%d")
    print(f"\n[Data Loading] Loading latest pothole activities...")
    activities = fetch_latest_notification_activities(use_local=False, download_if_missing=True)
    print(f"[Data Loading] Loaded {len(activities):,} pothole records")
    print(f"[Data Loading] Date range: {activities['date_reported'].min()} to {activities['date_reported'].max()}")

    # ============================================================================
    # BUNDLE TRACKING: Load historical bundle usage to ensure without-replacement
    # ============================================================================
    print(f"\n[Bundle Tracking] Checking historical bundle usage within {experiment_phase} phase...")
    print(f"  Phase tracking range: {phase_min_date} to {phase_max_date or 'ongoing'}")
    print(f"  (Only plans within this phase are tracked for without-replacement)")
    plan_dir = out_root / "plans"
    used_tracker = get_used_bundles(plan_dir, exclude_date=date, min_date=phase_min_date, max_date=phase_max_date)

    if used_tracker['all']:
        print_usage_summary(used_tracker)
    else:
        print("  No historical plans found - this appears to be the first plan")

    # Load sources/config (for CRS + optional CPD layer)
    streets, addrs, ds, pr = load_sources()
    work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))

    # Router uses street geometries; keep these handy for connectivity checks
    seg_id = pr["fields"]["streets_segment_id"]
    streets_m = project_to(streets, work_epsg)

    # Load filters layers once (for zoning + CPD + address filters)
    zoning_m, cpd_m_all, fcfg = load_filters_layers(ds, pr, work_epsg)

    # --- SFH-like addresses for mapping (same filters as sweep/emit) ---
    addrs_m = project_to(addrs, work_epsg).copy()
    addrs_f = apply_spatial_filters(addrs_m, zoning_m, cpd_m_all, fcfg, report={})

    # Drop non-addressable points before counting SFH addresses
    mask_addr = addressable_mask(addrs_f)
    addrs_f = addrs_f.loc[mask_addr].copy()

    # Compose address + __unit_blank__ flag (same helper as emit)
    _compose_address(addrs_f)

    # Keep only SFH-like addresses (no explicit unit string) if flag exists
    if "__unit_blank__" in addrs_f.columns:
        addrs_f = addrs_f.loc[addrs_f["__unit_blank__"]].copy()

    used_seg_ids: set[str] = set()

    if cpd:
        cpd_m = cpd_m_all
        if cpd_m is None or len(cpd_m) == 0:
            raise SystemExit(
                "CPD filter requested but no CPD layer is configured. "
                "Set datasources.yaml: cpd.path and params.yaml: filters.cpd.name_field."
            )
        name_field = _get_cpd_name_field(cpd_m, pr)

    # ============================================================================
    # NEW CONDITIONAL SAMPLING LOGIC
    # ============================================================================

    # Load DH bundles (shared for both DH and D2DS)
    print(f"\n[Bundle Loading] Loading bundles...")
    g_dh = _load_bundles("DH", bundle_file=bundle_file)

    # Standardize segment_id column
    seg_col_in_bundles = seg_id if seg_id in g_dh.columns else "segment_id"
    if seg_col_in_bundles != "segment_id":
        g_dh["segment_id"] = g_dh[seg_col_in_bundles].astype(str)
    else:
        g_dh["segment_id"] = g_dh["segment_id"].astype(str)

    # Get eligible bundles (had pothole in preceding week)
    print(f"\n[Eligibility Check] Determining eligible bundles based on preceding week...")

    # Calculate and display the actual date range being used
    from utils.sampling import get_week_start
    current_week_start = get_week_start(pd.Timestamp(current_date))
    preceding_week_start = current_week_start - timedelta(days=7)
    preceding_week_end = current_week_start - timedelta(days=1)
    print(f"  Current date: {date}")
    print(f"  Current week start (Saturday): {current_week_start.date()}")
    print(f"  Preceding week range: {preceding_week_start.date()} to {preceding_week_end.date()}")

    eligible_bundle_ids = get_eligible_bundles_for_date(
        current_date=current_date,
        activities_df=activities,
        bundles_df=g_dh,
        segment_col='segment_id'
    )
    print(f"[Eligibility Check] Found {len(eligible_bundle_ids)} eligible bundles (had potholes in preceding week)")

    # Apply filters (CPD, SFH, connectivity) to get candidate pools
    def _prepare_bundle_candidates(g, cpd_filter=None, sfh_min_val=None, sfh_max_val=None):
        """Apply filters and return candidate bundles"""
        # Optional CPD attach/filter
        if cpd_filter:
            g_m = project_to(g, work_epsg)
            g = _attach_bundle_cpd(g_m, cpd_m, name_field)
            g = g.loc[g["bundle_cpd"].astype(str) == str(cpd_filter)].copy()

        # Unique bundles with counts
        u = (
            g[["bundle_id", "bundle_addr_total"]]
            .drop_duplicates("bundle_id")
            .rename(columns={"bundle_addr_total": "sfh_bundle_total"})
        )

        # SFH range filter
        if sfh_min_val is not None:
            u = u[u["sfh_bundle_total"] >= int(sfh_min_val)]
        if sfh_max_val is not None:
            u = u[u["sfh_bundle_total"] <= int(sfh_max_val)]

        # Build segment lists per bundle
        seg_ids_by_bundle = (
            g[["bundle_id", "segment_id"]]
            .drop_duplicates()
            .groupby("bundle_id")["segment_id"]
            .agg(list)
            .rename("_seg_ids")
        )
        u = u.join(seg_ids_by_bundle, on="bundle_id")
        u = u.reset_index(drop=True)

        # No-overlap filter (segment-disjoint from already used)
        def _is_disjoint(seglist):
            if not isinstance(seglist, (list, tuple, set)) or not seglist:
                return False
            return {str(x) for x in seglist}.isdisjoint(used_seg_ids)

        mask = u["_seg_ids"].apply(_is_disjoint).to_numpy()
        u = u.loc[mask].copy()

        # Router-style connectivity check
        connected_mask = u["_seg_ids"].apply(
            lambda seglist: _route_connected_by_streets(streets_m, seg_id, set(seglist), snap_tol=0.5)
            if isinstance(seglist, (list, tuple, set)) and len(seglist) > 0
            else False
        )
        dropped = int((~connected_mask).sum())
        if dropped:
            print(f"  Skipped {dropped} non-route-connected bundle(s)")

        u = u.loc[connected_mask].copy()

        return u

    # Prepare candidate pools
    print(f"\n[Filter Application] Applying filters (CPD, SFH, connectivity)...")
    all_candidates = _prepare_bundle_candidates(g_dh, cpd, sfh_min, sfh_max)

    # Split into eligible and non-eligible
    eligible_candidates = all_candidates[all_candidates["bundle_id"].isin(eligible_bundle_ids)].copy()

    print(f"  Total candidates after filters: {len(all_candidates)}")
    print(f"  Eligible candidates (conditional pool): {len(eligible_candidates)}")

    # Get all bundle IDs for sampling
    all_bundle_ids_before_filter = set(all_candidates["bundle_id"].unique())

    # ============================================================================
    # IMPORTANT: Get previous week's DH conditional bundles BEFORE filtering
    # These bundles need to be exempt from without-replacement for D2DS reuse
    # ============================================================================
    print(f"\n[D2DS Exemption] Checking previous week's DH conditional for D2DS reuse...")
    prev_week_conditional = get_previous_week_conditional_bundles(
        plan_dir=plan_dir,
        current_date=date,
        activities_df=activities,
        bundles_df=g_dh,
        min_date=phase_min_date,  # Only look within current phase
        max_date=phase_max_date   # Ensure we stay within phase
    )

    # Filter out previously used bundles to ensure without-replacement
    # For DH sampling: strict without-replacement (no exemptions)
    all_bundle_ids = filter_available_bundles(all_bundle_ids_before_filter, used_tracker)

    # For D2DS sampling: exempt previous week's DH conditional bundles
    # This allows D2DS to reuse them despite the without-replacement rule
    all_bundle_ids_for_d2ds = filter_available_bundles(
        all_bundle_ids_before_filter,
        used_tracker,
        exempt_bundles=prev_week_conditional
    )

    # Also filter eligible bundles
    eligible_bundle_ids_filtered = set(eligible_candidates["bundle_id"].unique()) & all_bundle_ids

    # ============================================================================
    # Sample bundles using new conditional logic
    # ============================================================================
    rows = []
    dh_conditional_bundles = []  # Track DH conditional bundles for D2DS

    print(f"\n{'='*70}")
    print("SAMPLING BUNDLES (Conditional Design)")
    print(f"{'='*70}")

    # ============================================================================
    # DH Sampling
    # ============================================================================
    print(f"\n[DH Sampling] Using conditional sampling...")

    # Use sample_dh_bundles utility
    dh_sample_result = sample_dh_bundles(
        current_date=current_date,
        eligible_bundles=eligible_bundle_ids_filtered,
        all_bundles=all_bundle_ids,
        is_day_1=is_week_1,
        seed=rng.integers(0, 1e9)
    )

    sampled_dh_bundles = dh_sample_result['all_sampled']
    dh_conditional_bundles = dh_sample_result['conditional']
    dh_random_bundles = dh_sample_result['random']

    print(f"  Sampled {len(sampled_dh_bundles)} DH bundles:")
    print(f"    Conditional: {len(dh_conditional_bundles)}")
    print(f"    Random: {len(dh_random_bundles)}")

    # Get bundle details for DH bundles
    dh_details = all_candidates[all_candidates['bundle_id'].isin(sampled_dh_bundles)].copy()

    # ============================================================================
    # Optimize bundle-to-interviewer assignment using interviewer-aware bisection
    # ============================================================================
    print(f"\n[Assignment Optimization] Assigning DH bundles to interviewers using interviewer-aware bisection...")

    try:
        # Calculate bundles per interviewer
        bundles_per_interviewer = len(sampled_dh_bundles) // len(interviewers)

        # Use nearest-then-refine (best algorithm: 53.6% better than baseline)
        geocoded_file = root / "data" / "interviewers_geocoded.csv"

        dh_assignments = assign_bundles_for_date_nearest_then_refine(
            date=date,
            bundles=list(sampled_dh_bundles),
            bundles_gdf=g_dh,
            geocoded_file=str(geocoded_file) if geocoded_file.exists() else None,
            bundles_per_interviewer=bundles_per_interviewer,
            max_refine_iterations=50  # Iterative refinement to minimize max travel distance
        )

        print(f"[Assignment Optimization] Successfully assigned DH bundles using nearest-then-refine")

        # Print travel time summary
        max_travel_time = 0
        for name, (_, travel_time) in dh_assignments.items():
            print(f"  {name}: {travel_time:.2f} km")
            max_travel_time = max(max_travel_time, travel_time)
        print(f"  Maximum travel time: {max_travel_time:.2f} km")

        # Create name-to-code mapping (assignments returns real names, but we need codes A-F)
        real_names = get_interviewers_for_date(date)
        name_to_code = {real_names[i]: interviewers[i] for i in range(len(interviewers))}

        # Create rows from optimized assignments
        for interviewer_name, (bundle_ids, _) in dh_assignments.items():
            # Map real name back to code (A-F)
            interviewer_code = name_to_code.get(interviewer_name, interviewer_name)
            for bundle_id in bundle_ids:
                # Get bundle details
                bundle_row = dh_details[dh_details['bundle_id'] == bundle_id]
                if len(bundle_row) == 0:
                    print(f"  WARNING: Bundle {bundle_id} not found in candidates, skipping")
                    continue

                n_sfh = int(bundle_row['sfh_bundle_total'].iloc[0])
                seg_list = bundle_row['_seg_ids'].iloc[0]

                rows.append({
                    "date": date,
                    "interviewer": interviewer_code,
                    "task": "DH",
                    "bundle_id": int(bundle_id),
                    "list_code": int(list_code),
                    "sfh_bundle_total": n_sfh,
                })

                # Mark segments as used
                used_seg_ids.update({str(x) for x in (seg_list or [])})

    except Exception as e:
        print(f"[Assignment Optimization] Warning: Distance-based assignment failed: {e}")
        print(f"[Assignment Optimization] Falling back to simple round-robin assignment")

        # Fallback to simple assignment
        for i, bundle_id in enumerate(sampled_dh_bundles):
            # Get interviewer (cycle through if we have more bundles than interviewers)
            ivw = interviewers[i % len(interviewers)]

            # Get bundle details
            bundle_row = dh_details[dh_details['bundle_id'] == bundle_id]
            if len(bundle_row) == 0:
                print(f"  WARNING: Bundle {bundle_id} not found in candidates, skipping")
                continue

            n_sfh = int(bundle_row['sfh_bundle_total'].iloc[0])
            seg_list = bundle_row['_seg_ids'].iloc[0]

            rows.append({
                "date": date,
                "interviewer": ivw,
                "task": "DH",
                "bundle_id": int(bundle_id),
                "list_code": int(list_code),
                "sfh_bundle_total": n_sfh,
            })

            # Mark segments as used
            used_seg_ids.update({str(x) for x in (seg_list or [])})

    # ============================================================================
    # D2DS Sampling (only if not Week 1)
    # ============================================================================
    if not is_week_1 and "D2DS" in [t.upper() for t in tasks]:
        print(f"\n[D2DS Sampling] Selecting D2DS bundles...")

        # Update available bundles (exclude already used DH bundles)
        # Use all_bundle_ids_for_d2ds which includes exemption for prev week's DH conditional
        all_available_for_d2ds = all_bundle_ids_for_d2ds - set(sampled_dh_bundles)

        # For random D2DS, use bundles that had potholes in the ACTUAL previous field work week
        # For Jan 3 (Week 2), this should be Dec 27 - Jan 2 (when Dec 27's DH work happened)
        # This is the same as the "preceding week" of the current date
        print(f"  D2DS random sampling uses bundles with potholes in: {preceding_week_start.date()} to {preceding_week_end.date()}")
        eligible_for_d2ds_random = eligible_bundle_ids_filtered & all_available_for_d2ds

        print(f"  Eligible bundles for D2DS random (had potholes in previous field work week): {len(eligible_for_d2ds_random)}")

        # Use select_d2ds_bundles utility with PREVIOUS week's conditional bundles
        # and eligible bundles as random pool
        d2ds_selection = select_d2ds_bundles(
            conditional_bundles=prev_week_conditional,
            all_bundles=all_available_for_d2ds,
            bundles_df=g_dh,
            n_from_conditional=4,
            n_random=2,
            seed=rng.integers(0, 1e9),
            segment_col='segment_id',
            eligible_bundles=eligible_for_d2ds_random  # Random D2DS conditional on last week potholes
        )

        sampled_d2ds_bundles = d2ds_selection['d2ds_all']
        d2ds_conditional = d2ds_selection['d2ds_conditional']
        d2ds_random = d2ds_selection['d2ds_random']

        print(f"  Sampled {len(sampled_d2ds_bundles)} D2DS bundles:")
        print(f"    From DH conditional: {len(d2ds_conditional)}")
        print(f"    Random: {len(d2ds_random)}")

        # Mark DH bundles that are also used for D2DS
        for bundle_id in d2ds_conditional:
            for row in rows:
                if row['bundle_id'] == bundle_id and row['task'] == 'DH':
                    # Note: We don't add a separate D2DS row for these,
                    # they're already in DH and will do both tasks
                    pass

        # Get bundle details for D2DS bundles
        d2ds_details = all_candidates[all_candidates['bundle_id'].isin(sampled_d2ds_bundles)].copy()

        # ============================================================================
        # Optimize D2DS bundle-to-interviewer assignment using nearest-then-refine
        # ============================================================================
        print(f"\n[Assignment Optimization] Assigning D2DS bundles to interviewers using nearest-then-refine...")

        try:
            # Calculate bundles per interviewer
            bundles_per_interviewer_d2ds = len(sampled_d2ds_bundles) // len(interviewers)

            # Use nearest-then-refine (best algorithm: 53.6% better than baseline)
            geocoded_file = root / "data" / "interviewers_geocoded.csv"

            d2ds_assignments = assign_bundles_for_date_nearest_then_refine(
                date=date,
                bundles=list(sampled_d2ds_bundles),
                bundles_gdf=g_dh,
                geocoded_file=str(geocoded_file) if geocoded_file.exists() else None,
                bundles_per_interviewer=bundles_per_interviewer_d2ds,
                max_refine_iterations=50  # Iterative refinement to minimize max travel distance
            )

            print(f"[Assignment Optimization] Successfully assigned D2DS bundles using nearest-then-refine")

            # Print travel time summary
            max_travel_time_d2ds = 0
            for name, (_, travel_time) in d2ds_assignments.items():
                print(f"  {name}: {travel_time:.2f} km")
                max_travel_time_d2ds = max(max_travel_time_d2ds, travel_time)
            print(f"  Maximum travel time: {max_travel_time_d2ds:.2f} km")

            # Create name-to-code mapping (assignments returns real names, but we need codes A-F)
            real_names_d2ds = get_interviewers_for_date(date)
            name_to_code_d2ds = {real_names_d2ds[i]: interviewers[i] for i in range(len(interviewers))}

            # Create rows from optimized assignments
            for interviewer_name, (bundle_ids, _) in d2ds_assignments.items():
                # Map real name back to code (A-F)
                interviewer_code = name_to_code_d2ds.get(interviewer_name, interviewer_name)
                for bundle_id in bundle_ids:
                    # Get bundle details (from either d2ds_conditional or d2ds_random)
                    if bundle_id in d2ds_conditional:
                        bundle_row = all_candidates[all_candidates['bundle_id'] == bundle_id]
                    else:
                        bundle_row = d2ds_details[d2ds_details['bundle_id'] == bundle_id]

                    if len(bundle_row) == 0:
                        print(f"  WARNING: Bundle {bundle_id} not found in candidates, skipping")
                        continue

                    n_sfh = int(bundle_row['sfh_bundle_total'].iloc[0])
                    seg_list = bundle_row['_seg_ids'].iloc[0] if bundle_id in d2ds_random else []

                    rows.append({
                        "date": date,
                        "interviewer": interviewer_code,
                        "task": "D2DS",
                        "bundle_id": int(bundle_id),
                        "list_code": int(list_code),
                        "sfh_bundle_total": n_sfh,
                    })

                    # Mark segments as used (only for random bundles, conditional already marked)
                    if bundle_id in d2ds_random:
                        used_seg_ids.update({str(x) for x in (seg_list or [])})

        except Exception as e:
            print(f"[Assignment Optimization] Warning: Distance-based D2DS assignment failed: {e}")
            print(f"[Assignment Optimization] Falling back to simple round-robin assignment")

            # Fallback to simple assignment
            for i, bundle_id in enumerate(sampled_d2ds_bundles):
                ivw = interviewers[i % len(interviewers)]

                # Get bundle details (from either d2ds_conditional or d2ds_random)
                if bundle_id in d2ds_conditional:
                    bundle_row = all_candidates[all_candidates['bundle_id'] == bundle_id]
                else:
                    bundle_row = d2ds_details[d2ds_details['bundle_id'] == bundle_id]

                if len(bundle_row) == 0:
                    print(f"  WARNING: Bundle {bundle_id} not found in candidates, skipping")
                    continue

                n_sfh = int(bundle_row['sfh_bundle_total'].iloc[0])
                seg_list = bundle_row['_seg_ids'].iloc[0] if bundle_id in d2ds_random else []

                rows.append({
                    "date": date,
                    "interviewer": ivw,
                    "task": "D2DS",
                    "bundle_id": int(bundle_id),
                    "list_code": int(list_code),
                    "sfh_bundle_total": n_sfh,
                })

                # Mark segments as used (only for random bundles, conditional already marked)
                if bundle_id in d2ds_random:
                    used_seg_ids.update({str(x) for x in (seg_list or [])})

    elif is_week_1:
        print(f"\n[D2DS Sampling] Skipped (Week 1 has no D2DS)")

    plan = pd.DataFrame(rows).sort_values(["task", "interviewer"]).reset_index(drop=True)

    out_csv = out_csv or str(out_root / "plans" / f"bundles_plan_{date}.csv")
    ensure_dir(Path(out_csv).parent)
    plan.to_csv(out_csv, index=False)
    print(f"[plan] wrote {out_csv} ({len(plan)} rows)")

    # --- Small HTML map (red=DH, blue=D2DS) with SFH points saved next to the CSV ---
    try:
        import folium

        g_all_m = []
        for t in plan["task"].unique():
            # Use the same bundle file logic as in _load_bundles
            if bundle_file:
                pth = Path(bundle_file)
                if not pth.is_absolute():
                    pth = paths()[0] / bundle_file
            else:
                pth = paths()[0] / "outputs" / "bundles" / t / "bundles.parquet"
            gb = gpd.read_parquet(pth)
            ids = plan.loc[plan["task"] == t, "bundle_id"].tolist()
            gb = gb.loc[gb["bundle_id"].isin(ids)].copy()

            # Color by task
            gb["__color"] = "#d7191c" if t == "DH" else "#2c7bb6"  # red / blue

            # Ensure we have the segment-id column
            if seg_id not in gb.columns:
                for alt in ("segment_id", "SEGMENT_ID", "StreetSegID", "SEGMENTID", "SegmentID"):
                    if alt in gb.columns:
                        gb = gb.rename(columns={alt: seg_id})
                        break

            g_all_m.append(gb[["bundle_id", seg_id, "geometry", "__color"]])

        if g_all_m:
            # Selected segments in working CRS and WGS84
            g_sel_m = gpd.GeoDataFrame(
                pd.concat(g_all_m, ignore_index=True),
                geometry="geometry",
                crs=streets_m.crs,
            )
            g_sel = g_sel_m.to_crs(4326)

            # Snap SFH-like addresses to selected segments (same colors as segments)
            pts_sel = None
            try:
                buf = int(pr.get("addr_buffer_meters", 25))
                joined = gpd.sjoin_nearest(
                    addrs_f,
                    g_sel_m[[seg_id, "geometry", "__color"]],
                    how="inner",
                    max_distance=buf,
                    distance_col="_dist_m",
                )
                pts_sel = project_to(joined, 4326)
            except Exception as e:
                print(f"[plan] address points skipped (nearest join failed): {e}")

            m = folium.Map(location=[32.77, -117.15], zoom_start=13, tiles="cartodbpositron")

            # Segment layer
            for _, row in g_sel.iterrows():
                folium.GeoJson(
                    data=row["geometry"].__geo_interface__,
                    style_function=(lambda _r, c=row["__color"]: {"color": c, "weight": 3, "opacity": 0.9}),
                    tooltip=f"bundle_id={int(row['bundle_id'])}",
                ).add_to(m)

            # Address points layer: homes included in the plan (same color as segment)
            if pts_sel is not None and len(pts_sel):
                for r in pts_sel.itertuples():
                    geom = r.geometry
                    color = getattr(r, "__color", "#444444")
                    folium.CircleMarker(
                        location=[geom.y, geom.x],
                        radius=3,
                        weight=0,
                        fill=True,
                        fill_opacity=0.9,
                        color=color,
                        fill_color=color,
                    ).add_to(m)

            map_out = Path(out_csv).with_suffix(".html")
            m.save(str(map_out))
            print(f"[plan] wrote map: {map_out}")
        else:
            print("[plan] map skipped: no selected bundles to render.")
    except Exception as e:
        print("[plan] map skipped:", e)

    return out_csv
