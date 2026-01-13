# fieldprep/src/sd311_fieldprep/emit.py
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd

from sd311_fieldprep.utils import (
    load_sources,
    ensure_dir,
    paths,
    project_to,
    load_filters_layers,
    apply_spatial_filters,
)
from sd311_fieldprep.route import build_walk_order


def _pick_existing(df: pd.DataFrame, candidates):
    """Return the first candidate column name (case-insensitive) that exists in df."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _compose_address(addrs: pd.DataFrame) -> str:
    """
    Compose a human-readable address string using DataSD address fields, if present:
      addrnmbr, addrfrac, addrpdir, addrname, addrpostd, addrsfx, addrunit, addrzip
    Writes 'address' into `addrs` and returns the column name.
    """
    def col(name):  # case-insensitive getter
        m = {c.lower(): c for c in addrs.columns}
        return m.get(name.lower())

    num = col("addrnmbr")
    frac = col("addrfrac")
    pdir = col("addrpdir")
    name = col("addrname")
    postd = col("addrpostd")
    sfx = col("addrsfx")
    unit = col("addrunit")
    zcol = col("addrzip")

    # Helpers to clean pieces
    def as_str(s):
        return addrs[s].astype(str).where(~addrs[s].isna(), "").str.strip() if s else ""

    num_s   = as_str(num)
    # Treat '0' and '0000' as missing house numbers (DataSD sometimes encodes unknown as 0)
    if isinstance(num_s, pd.Series):
        num_s = num_s.mask(num_s.eq("0") | num_s.eq("0000"), "")


    frac_s  = as_str(frac)
    pdir_s  = as_str(pdir)
    name_s  = as_str(name)
    postd_s = as_str(postd)
    sfx_s   = as_str(sfx)
    unit_s  = as_str(unit)
    zip_s   = as_str(zcol)

    # Street suffix nice-case mapping
    sfx_map = {
        "ST": "St", "AVE": "Ave", "AV": "Ave", "AVENUE": "Ave", "BLVD": "Blvd", "CT": "Ct",
        "DR": "Dr", "LN": "Ln", "RD": "Rd", "PL": "Pl", "TER": "Ter", "WAY": "Way",
        "HWY": "Hwy", "PKWY": "Pkwy", "CIR": "Cir"
    }
    sfx_fmt = sfx_s.str.upper().map(sfx_map).fillna(sfx_s.str.title()) if isinstance(sfx_s, pd.Series) else ""

    # Build number + fraction
    numfrac = num_s
    if isinstance(frac_s, pd.Series):
        frac_clean = frac_s.mask(frac_s.eq(""))
        numfrac = numfrac.mask(numfrac.eq("")).fillna("")  # keep as string
        numfrac = numfrac + frac_clean.radd(" ", fill_value="").fillna("")

    # Proper-case street name
    name_fmt = name_s.str.title() if isinstance(name_s, pd.Series) else ""

    # Compose main line: pre-dir, name, suffix, post-dir
    parts = []
    if isinstance(pdir_s, pd.Series):   parts.append(pdir_s)
    if isinstance(name_fmt, pd.Series): parts.append(name_fmt)
    if isinstance(sfx_fmt, pd.Series):  parts.append(sfx_fmt)
    if isinstance(postd_s, pd.Series):  parts.append(postd_s)

    # Join street parts with spaces
    if parts:
        street = parts[0]
        for p in parts[1:]:
            street = street.str.cat(p, sep=" ").str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        street = pd.Series([""] * len(addrs), index=addrs.index)

    # Final address line: number + street
    addr_line = numfrac.mask(numfrac.eq("")).fillna("").str.cat(street, sep=" ").str.replace(r"\s+", " ", regex=True).str.strip()
    # If result is just "0" (defensive), blank it
    addr_line = addr_line.mask(addr_line.eq("0"), "")

    # Append ZIP if present (5-digit)
    if isinstance(zip_s, pd.Series) and not zip_s.empty:
        zip5 = zip_s.str.extract(r"(\d{5})", expand=False).fillna("")
        addr_line = np.where(zip5.eq(""), addr_line, addr_line + ", " + zip5)

    # Store to 'address'
    addrs["address"] = pd.Series(addr_line, index=addrs.index).fillna("").str.replace(r",\s*$", "", regex=True)

    # Flag blank unit (helps keep SFH-like addresses only later)
    if isinstance(unit_s, pd.Series):
        mask_blank = unit_s.fillna("").str.strip().eq("")
        addrs["__unit_blank__"] = mask_blank

    return "address"


def _stable_seed(date_str: str, interviewer: str, bundle_id: int) -> int:
    s = f"{date_str}|{interviewer}|{int(bundle_id)}|DH|partial"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2**32 - 1)


def _stable_seed_bundle_treatment(date_str: str, interviewer: str, bundle_id: int) -> int:
    """Deterministic seed for bundle-level control vs treatment assignment."""
    s = f"{date_str}|{interviewer}|{int(bundle_id)}|DH|treatment"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2**32 - 1)


def _stable_seed_segment(date_str: str, interviewer: str, seg_id: str | int) -> int:
    """Deterministic seed for per-segment partial DH address thinning."""
    s = f"{date_str}|{interviewer}|{str(seg_id)}|DH|partial"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2**32 - 1)


def _generate_segment_analysis(routes: pd.DataFrame, date: str, out_dir: Path, root: Path):
    """
    Generate comprehensive segment-level analysis CSV with pothole metrics.

    Outputs segment_analysis.csv with columns:
    - segment_id
    - bundle_id
    - date
    - treatment_status_dh (DH/empty)
    - survey_status_d2ds (D2DS/empty)

    This week pothole metrics:
    - has_pothole_this_week (1/0)
    - pothole_count_this_week
    - all_potholes_fixed_this_week (1/0)
    - share_fixed_this_week (0-1)
    - avg_days_to_fix_this_week
    - max_days_to_fix_this_week

    Last week pothole metrics:
    - has_pothole_last_week (1/0)
    - pothole_count_last_week
    - all_potholes_fixed_last_week (1/0)
    - share_fixed_last_week (0-1)
    - avg_days_to_fix_last_week
    - max_days_to_fix_last_week

    - bundle_pothole_count_excl_segment (this week's potholes in bundle excluding this segment)
    """
    print(f"[emit] Generating segment-level analysis...")

    try:
        # Load notification activities data
        from utils.data_fetcher import fetch_latest_notification_activities
        potholes = fetch_latest_notification_activities(use_local=True, download_if_missing=True)

        # Calculate days to fix
        potholes['days_to_fix'] = (
            potholes['date_closed'] - potholes['date_reported']
        ).dt.days

        potholes['week_start'] = pd.to_datetime(potholes['week_start'])

        # Convert date to datetime
        date_obj = pd.to_datetime(date)
        last_week_date = date_obj - pd.Timedelta(days=7)

        # Helper function to calculate weekly metrics
        def calc_week_metrics(week_date, suffix):
            week_data = potholes[potholes['week_start'] == week_date].copy()

            if week_data.empty:
                print(f"[emit][WARN] No pothole data for {suffix} ({week_date.date()})")
                return pd.DataFrame()

            # Count total potholes reported per segment (each row is one pothole)
            pothole_counts = week_data.groupby('segment_id').size().reset_index(name=f'pothole_count_{suffix}')

            # Count fixed potholes (those with date_closed not null)
            fixed_mask = week_data['date_closed'].notna()
            fixed_counts = week_data[fixed_mask].groupby('segment_id').size().reset_index(name=f'_fixed_count_{suffix}')

            # Merge counts
            metrics = pothole_counts.merge(fixed_counts, on='segment_id', how='left')
            metrics[f'_fixed_count_{suffix}'] = metrics[f'_fixed_count_{suffix}'].fillna(0).astype(int)

            # Calculate derived metrics
            metrics[f'has_pothole_{suffix}'] = (metrics[f'pothole_count_{suffix}'] > 0).astype(int)

            # Share fixed (0-1)
            metrics[f'share_fixed_{suffix}'] = (
                metrics[f'_fixed_count_{suffix}'] / metrics[f'pothole_count_{suffix}']
            ).fillna(0)

            # All potholes fixed (1/0)
            metrics[f'all_potholes_fixed_{suffix}'] = (
                (metrics[f'pothole_count_{suffix}'] > 0) &
                (metrics[f'_fixed_count_{suffix}'] == metrics[f'pothole_count_{suffix}'])
            ).astype(int)

            # Days to fix metrics (only for fixed potholes)
            fixed_data = week_data[week_data['date_closed'].notna()].copy()

            if not fixed_data.empty and 'days_to_fix' in fixed_data.columns:
                days_metrics = fixed_data.groupby('segment_id').agg({
                    'days_to_fix': ['mean', 'max']
                }).reset_index()

                if not days_metrics.empty:
                    days_metrics.columns = ['segment_id', f'avg_days_to_fix_{suffix}', f'max_days_to_fix_{suffix}']
                    metrics = metrics.merge(days_metrics, on='segment_id', how='left')
                else:
                    metrics[f'avg_days_to_fix_{suffix}'] = None
                    metrics[f'max_days_to_fix_{suffix}'] = None

                # Fill NaN for segments with potholes but none fixed (use 0 as placeholder)
                metrics[f'avg_days_to_fix_{suffix}'] = metrics[f'avg_days_to_fix_{suffix}'].fillna(0)
                metrics[f'max_days_to_fix_{suffix}'] = metrics[f'max_days_to_fix_{suffix}'].fillna(0)
            else:
                # No days_to_fix data available
                metrics[f'avg_days_to_fix_{suffix}'] = 0
                metrics[f'max_days_to_fix_{suffix}'] = 0

            # Drop intermediate columns
            metrics = metrics.drop(columns=[f'_fixed_count_{suffix}'])

            return metrics

        # Calculate metrics for this week and last week
        this_week_metrics = calc_week_metrics(date_obj, 'this_week')
        last_week_metrics = calc_week_metrics(last_week_date, 'last_week')

        # Merge with routes to get bundle_id and task info
        routes_seg = routes[['segment_id', 'bundle_id', 'task', 'date']].drop_duplicates()

        # Create DH and D2DS status columns
        routes_seg['treatment_status_dh'] = ''
        routes_seg['survey_status_d2ds'] = ''

        dh_mask = routes_seg['task'].str.upper() == 'DH'
        d2ds_mask = routes_seg['task'].str.upper() == 'D2DS'

        routes_seg.loc[dh_mask, 'treatment_status_dh'] = 'DH'
        routes_seg.loc[d2ds_mask, 'survey_status_d2ds'] = 'D2DS'

        # Merge with pothole metrics
        analysis = routes_seg.copy()

        if not this_week_metrics.empty:
            analysis = analysis.merge(this_week_metrics, on='segment_id', how='left')
        else:
            # Add empty columns if no data
            for col in ['has_pothole_this_week', 'pothole_count_this_week',
                       'all_potholes_fixed_this_week', 'share_fixed_this_week',
                       'avg_days_to_fix_this_week', 'max_days_to_fix_this_week']:
                analysis[col] = None

        if not last_week_metrics.empty:
            analysis = analysis.merge(last_week_metrics, on='segment_id', how='left')
        else:
            # Add empty columns if no data
            for col in ['has_pothole_last_week', 'pothole_count_last_week',
                       'all_potholes_fixed_last_week', 'share_fixed_last_week',
                       'avg_days_to_fix_last_week', 'max_days_to_fix_last_week']:
                analysis[col] = None

        # Fill NaN values for segments without potholes
        # Note: avg/max_days_to_fix will be None for all segments if t_repair field doesn't exist
        # This week
        analysis['has_pothole_this_week'] = analysis['has_pothole_this_week'].fillna(0).astype(int)
        analysis['pothole_count_this_week'] = analysis['pothole_count_this_week'].fillna(0).astype(int)
        analysis['all_potholes_fixed_this_week'] = analysis['all_potholes_fixed_this_week'].fillna(0).astype(int)
        analysis['share_fixed_this_week'] = analysis['share_fixed_this_week'].fillna(0)
        # For days metrics: keep as None if field missing, otherwise fill with 0
        if analysis['avg_days_to_fix_this_week'].notna().any():
            analysis['avg_days_to_fix_this_week'] = analysis['avg_days_to_fix_this_week'].fillna(0)
            analysis['max_days_to_fix_this_week'] = analysis['max_days_to_fix_this_week'].fillna(0)

        # Last week
        analysis['has_pothole_last_week'] = analysis['has_pothole_last_week'].fillna(0).astype(int)
        analysis['pothole_count_last_week'] = analysis['pothole_count_last_week'].fillna(0).astype(int)
        analysis['all_potholes_fixed_last_week'] = analysis['all_potholes_fixed_last_week'].fillna(0).astype(int)
        analysis['share_fixed_last_week'] = analysis['share_fixed_last_week'].fillna(0)
        # For days metrics: keep as None if field missing, otherwise fill with 0
        if analysis['avg_days_to_fix_last_week'].notna().any():
            analysis['avg_days_to_fix_last_week'] = analysis['avg_days_to_fix_last_week'].fillna(0)
            analysis['max_days_to_fix_last_week'] = analysis['max_days_to_fix_last_week'].fillna(0)

        # Calculate bundle-level pothole count excluding each segment (this week only)
        bundle_totals = analysis.groupby('bundle_id')['pothole_count_this_week'].sum().to_dict()
        analysis['bundle_pothole_count_excl_segment'] = (
            analysis['bundle_id'].map(bundle_totals) - analysis['pothole_count_this_week']
        )

        # Select final columns
        output_cols = [
            'segment_id',
            'bundle_id',
            'date',
            'treatment_status_dh',
            'survey_status_d2ds',
            'has_pothole_this_week',
            'pothole_count_this_week',
            'all_potholes_fixed_this_week',
            'share_fixed_this_week',
            'avg_days_to_fix_this_week',
            'max_days_to_fix_this_week',
            'has_pothole_last_week',
            'pothole_count_last_week',
            'all_potholes_fixed_last_week',
            'share_fixed_last_week',
            'avg_days_to_fix_last_week',
            'max_days_to_fix_last_week',
            'bundle_pothole_count_excl_segment'
        ]

        # Write to CSV in daily output directory
        output_file = out_dir / "segment_analysis.csv"
        analysis[output_cols].to_csv(output_file, index=False)

        # Also save to routing folder
        routing_dir = root / "outputs" / "routing" / "segment_analysis" / date
        routing_dir.mkdir(parents=True, exist_ok=True)
        routing_file = routing_dir / "segment_analysis.csv"
        analysis[output_cols].to_csv(routing_file, index=False)

        print(f"[emit] Wrote segment analysis to {output_file}")
        print(f"[emit] Wrote segment analysis to {routing_file}")
        print(f"[emit]   {len(analysis)} segments analyzed")
        print(f"[emit]   This week: {analysis['has_pothole_this_week'].sum()} segments with potholes")
        print(f"[emit]   Last week: {analysis['has_pothole_last_week'].sum()} segments with potholes")

    except Exception as e:
        print(f"[emit][ERROR] Failed to generate segment analysis: {e}")
        import traceback
        traceback.print_exc()


def run_emit(date: str, plan_csv: str, bundle_file: str | None = None, addr_assignment_file: str | None = None):
    """
    Generate daily work files (routes, segments, addresses) from a plan.

    Args:
        date: Date string (YYYY-MM-DD) for the plan
        plan_csv: Path to the plan CSV file
        bundle_file: Optional path to shared bundle file for all tasks.
                     Example: "outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet"
                     If provided, all tasks (DH, D2DS) will use this file.
                     If None (default), each task uses its own bundle file from
                     outputs/bundles/{task}/bundles.parquet
        addr_assignment_file: Optional path to segment-to-address assignment file from sweep.
                              Example: "outputs/sweep/locked/segment_addresses_b40_m2.parquet"
                              If provided, uses the exact address assignments from sweep instead
                              of re-running spatial join. This ensures consistency between
                              sweep → bundle → plan → emit.
                              If None (default), performs spatial join (old behavior).
    """
    root, cfg, out_root = paths()

    # ---- Plan ----
    plan = pd.read_csv(plan_csv, dtype={"interviewer": str, "task": str})
    plan = plan.loc[plan["date"] == date].copy()
    if plan.empty:
        raise SystemExit(f"No rows in plan for date={date}")
    # Ignore any plan-time dh_saturation; we randomize per-segment below

    # ---- Bundles ----
    bundles = {}
    if bundle_file:
        # Use shared bundle file for all sessions
        bpath = Path(bundle_file)
        if not bpath.is_absolute():
            bpath = root / bundle_file
        if not bpath.exists():
            raise SystemExit(f"Shared bundle file not found: {bpath}")

        # Load once and use for all sessions
        shared_bundles = gpd.read_parquet(bpath)
        for session in plan["task"].str.upper().unique():
            bundles[session] = shared_bundles
    else:
        # Use session-specific bundle files (original behavior)
        for session in plan["task"].str.upper().unique():
            bpath = root / "outputs" / "bundles" / session / "bundles.parquet"
            if not bpath.exists():
                raise SystemExit(f"Missing bundles parquet for session {session}: {bpath}")
            bundles[session] = gpd.read_parquet(bpath)

    # ---- Outputs dir ----
    out_dir = out_root / "incoming" / "daily" / date
    ensure_dir(out_dir)

    # ---- Sources & config fields ----
    streets, addrs, ds, pr = load_sources()
    fields       = (pr.get("fields", {}) or {})
    seg_id_pref  = fields.get("streets_segment_id") or pr.get("segment_id_field") or "segment_id"
    addr_id_name = fields.get("addresses_id", "addr_id")  # unified alias from utils
    w_m          = int((pr.get("crs", {}) or {}).get("working_meters", 26911))
    w_wgs        = int((pr.get("crs", {}) or {}).get("output_wgs84", 4326))
    buf          = int(pr.get("addr_buffer_meters", 25))

    streets_m = project_to(streets, w_m)
    addrs_m   = project_to(addrs,   w_m)

    # Compose addresses from DataSD fields (and mark unit blankness)
    _compose_address(addrs_m)

    # ---- Build ordered routes and collect used segments ----
    used_segments = set()
    routes_rows = []

    # keep last-detected streets_key for segments.geojson
    streets_key_global = None

    for row in plan.itertuples(index=False):
        session     = str(row.task).upper()
        interviewer = str(row.interviewer)
        bundle_id   = int(row.bundle_id)
        list_code   = int(row.list_code)

        bund = bundles[session]

        # Detect bundles' segment id column (does NOT assume 'segment_id' exists)
        bund_seg_key = _pick_existing(bund, [seg_id_pref, "iamfloc", "segment_id", "streetsegid", "segmentid", "street_seg", "OBJECTID", "roadsegid"])
        if bund_seg_key is None:
            raise SystemExit(f"[emit:{session} {interviewer}] bundles file lacks a recognizable segment-id column. Columns: {list(bund.columns)[:12]}...")

        sub = bund.loc[bund["bundle_id"] == bundle_id, [bund_seg_key, "bundle_id"]].copy()
        if sub.empty:
            raise SystemExit(f"Bundle {bundle_id} not found in session {session}")

        # Choose a common join key between streets and bundles
        streets_key = _pick_existing(streets_m, [seg_id_pref, bund_seg_key, "iamfloc", "segment_id", "streetsegid", "segmentid", "street_seg", "OBJECTID", "roadsegid"])
        if streets_key is None:
            raise SystemExit(f"[emit:{session} {interviewer}] No segment-id column found in streets. Tried: {seg_id_pref}, {bund_seg_key}, 'iamfloc', 'segment_id', ...")
        streets_key_global = streets_key

        streets_m["_seg_str"] = streets_m[streets_key].astype(str)
        sub["_seg_str"]       = sub[bund_seg_key].astype(str)

        bsegs = streets_m.merge(sub[["_seg_str", "bundle_id"]], on="_seg_str", how="inner")
        if bsegs.empty:
            left_sample  = streets_m[[streets_key]].head(3).astype(str).to_dict(orient="list")
            right_sample = sub[[bund_seg_key]].head(3).astype(str).to_dict(orient="list")
            raise SystemExit(
                f"[emit:{session} {interviewer}] bundle_id={bundle_id}: Join produced 0 rows. "
                f"Tried streets key='{streets_key}' vs bundles key='{bund_seg_key}'. "
                f"Sample streets keys: {left_sample}; bundle keys: {right_sample}."
            )

        # Route order (route expects column named 'segment_id')
        ordered = build_walk_order(bsegs.rename(columns={streets_key: "segment_id"}))

        for i, (seg_id_val, oriented) in enumerate(ordered, start=1):
            routes_rows.append(
                {
                    "date": date,
                    "interviewer": interviewer,
                    "task": session,
                    "seq": i,
                    "segment_id": seg_id_val,
                    "start_flag": 1 if i == 1 else 0,
                    "list_code": list_code,
                    "bundle_id": bundle_id,
                    "dh_saturation": "",  # No longer used at segment level; now address-level allocation
                }
            )
            used_segments.add(str(seg_id_val))

    # Write routes.csv
    routes = pd.DataFrame(routes_rows)
    routes.to_csv(out_dir / "routes.csv", index=False)

    # segments.geojson (WGS84) — use last streets_key encountered
    if streets_key_global is None:
        raise SystemExit("Could not determine a segment-id column to write segments.geojson.")
    segs_use = streets.loc[streets[streets_key_global].astype(str).isin(used_segments)].copy().to_crs(4326)
    segs_use[[streets_key_global, "geometry"]].rename(columns={streets_key_global: "segment_id"}).to_file(out_dir / "segments.geojson", driver="GeoJSON")

    # ---- Build sfh_points.csv ----
    zoning_m, cpd_m, fcfg = load_filters_layers(ds, pr, w_m)
    addrs_m_filt = apply_spatial_filters(addrs_m, zoning_m, cpd_m, fcfg)

    # Drop non-addressable addresses so enumerators don't see blank entries
    try:
        from sd311_fieldprep.utils import addressable_mask
        mask_addr = addressable_mask(addrs_m_filt)
        dropped = int((~mask_addr).sum())
        if dropped:
            print(f"[emit] dropping {dropped} non-addressable addresses (missing number/name)")
        addrs_m_filt = addrs_m_filt.loc[mask_addr].copy()
    except Exception:
        pass

    # Drop addresses with explicit unit strings (keep SFH-like)
    if "__unit_blank__" in addrs_m_filt.columns:
        addrs_m_filt = addrs_m_filt.loc[addrs_m_filt["__unit_blank__"]].copy()

    used_set = set(used_segments)

    # ---- Match addresses to segments ----
    # Use precomputed assignment file if provided (ensures consistency with sweep)
    # Otherwise fall back to spatial join (old behavior)

    if addr_assignment_file:
        # Use exact assignments from sweep
        print(f"[emit] Using precomputed address assignments from {addr_assignment_file}")

        # Read assignment file
        assign_path = Path(addr_assignment_file)
        if not assign_path.is_absolute():
            assign_path = root / addr_assignment_file
        if not assign_path.exists():
            raise SystemExit(f"Address assignment file not found: {assign_path}")

        addr_assign = pd.read_parquet(assign_path)
        print(f"[emit] Loaded {len(addr_assign)} address assignments")

        # Filter to only segments in current plan
        addr_assign["segment_id"] = addr_assign["segment_id"].astype(str)
        plan_assignments = addr_assign[addr_assign["segment_id"].isin(used_set)].copy()
        print(f"[emit] Filtered to {len(plan_assignments)} addresses for {len(used_set)} segments in plan")

        # Join with filtered addresses to get geometry and address details
        pts_all = addrs_m_filt.merge(
            plan_assignments[["addr_id", "segment_id"]],
            on="addr_id",
            how="inner"
        )

        if pts_all.empty:
            raise SystemExit(f"No addresses matched after applying precomputed assignments. Check that addr_id field matches.")

        print(f"[emit] Matched {len(pts_all)} addresses using precomputed assignments")

    else:
        # Fall back to spatial join (old behavior)
        segs_m = streets_m.loc[streets_m[streets_key_global].astype(str).isin(used_set), [streets_key_global, "geometry"]].copy()

        try:
            joined = gpd.sjoin_nearest(addrs_m_filt, segs_m, how="inner", max_distance=buf)
        except Exception as e:
            raise RuntimeError("Nearest join requires a spatial index (rtree). Try: pip install rtree") from e

        if joined.empty:
            raise SystemExit(f"No address points matched to the used segments within {buf}m buffer.")

        pts_all = addrs_m_filt.loc[joined.index].copy()
        pts_all["segment_id"] = joined[streets_key_global].astype(str).values

    # Project to WGS84 and prepare columns
    pts_wgs = project_to(pts_all, w_wgs).copy()
    pts_wgs["lat"] = pts_wgs.geometry.y.values
    pts_wgs["lon"] = pts_wgs.geometry.x.values

    # Ensure unified 'addr_id' column exists (alias if needed)
    if "addr_id" not in pts_wgs.columns and addr_id_name in pts_wgs.columns:
        pts_wgs.rename(columns={addr_id_name: "addr_id"}, inplace=True)

    # Ensure 'address' column exists
    if "address" not in pts_wgs.columns:
        pts_wgs["address"] = ""

    # ---- Merge addresses with task/bundle info ----
    task_segs = routes[["interviewer", "task", "segment_id", "bundle_id"]].drop_duplicates()
    pts_task = pts_wgs.merge(task_segs, on="segment_id", how="left")

    # ---- Address-level DH treatment allocation ----
    # For DH tasks: allocate addresses within each bundle to 50% control, 25% full, 25% partial
    pts_task["dh_treatment"] = ""  # Will be: control/full/partial for DH addresses
    pts_task["dh_selected"] = True  # Whether to include in final output

    is_dh = pts_task["task"].astype(str).str.upper().eq("DH")

    if is_dh.any():
        print(f"[emit] Allocating DH addresses to treatment groups (50% control, 25% full, 25% partial)...")

        # Group DH addresses by (interviewer, bundle_id)
        for (ivw, bid), g in pts_task.loc[is_dh].groupby(["interviewer", "bundle_id"], dropna=False):
            n = len(g)
            idx = g.index.to_numpy()

            # Round down to multiple of 4 for clean 50%/25%/25% split
            n_usable = (n // 4) * 4
            n_control = n_usable // 2
            n_full = n_usable // 4
            n_partial = n_usable // 4

            # Create seed for this bundle's randomization
            seed = _stable_seed_bundle_treatment(date, ivw, bid)
            rng = np.random.default_rng(seed)

            # Shuffle indices
            shuffled = rng.permutation(idx)

            # Assign first n_usable addresses
            control_idx = shuffled[:n_control]
            full_idx = shuffled[n_control:n_control + n_full]
            partial_idx = shuffled[n_control + n_full:n_usable]

            # Assign remainder to keep full and partial balanced
            # Priority: control gets more (50%), then balance full/partial (25% each)
            if n > n_usable:
                remainder_idx = shuffled[n_usable:]
                n_remainder = len(remainder_idx)

                if n_remainder == 1:
                    # Give to control (maintains 50% ratio)
                    control_idx = np.append(control_idx, remainder_idx[0])
                elif n_remainder == 2:
                    # Give 1 to control, 1 randomly to full or partial (keeps them balanced)
                    control_idx = np.append(control_idx, remainder_idx[0])
                    if rng.random() < 0.5:
                        full_idx = np.append(full_idx, remainder_idx[1])
                    else:
                        partial_idx = np.append(partial_idx, remainder_idx[1])
                elif n_remainder == 3:
                    # Give 1 to each group (keeps ratios perfect)
                    control_idx = np.append(control_idx, remainder_idx[0])
                    full_idx = np.append(full_idx, remainder_idx[1])
                    partial_idx = np.append(partial_idx, remainder_idx[2])

            # Mark treatment assignments
            pts_task.loc[control_idx, "dh_treatment"] = "control"
            pts_task.loc[full_idx, "dh_treatment"] = "full"
            pts_task.loc[partial_idx, "dh_treatment"] = "partial"

            # Control addresses not selected (not visited)
            pts_task.loc[control_idx, "dh_selected"] = False

            # Full addresses all selected (100% visited)
            pts_task.loc[full_idx, "dh_selected"] = True

            # Partial addresses: randomly select 50%
            if len(partial_idx) > 0:
                k = len(partial_idx) // 2
                if k > 0:
                    seed_partial = _stable_seed(date, ivw, bid)
                    rng_partial = np.random.default_rng(seed_partial)
                    keep_partial = rng_partial.choice(partial_idx, size=k, replace=False)
                    pts_task.loc[partial_idx, "dh_selected"] = False
                    pts_task.loc[keep_partial, "dh_selected"] = True

            print(f"[emit]   {ivw} bundle {bid}: {len(control_idx)} control, {len(full_idx)} full, {len(partial_idx)} partial (total {n})")

    # --- Keep only the addresses we actually want crews to visit ---
    is_dh = pts_task["task"].astype(str).str.upper().eq("DH")
    pts_task = pts_task.loc[~is_dh | (is_dh & pts_task["dh_selected"])].copy()

    # --- Unified sfh_points.csv: one file for all tasks, no dh_selected column ---
    out_cols = ["interviewer", "task", "segment_id", "lat", "lon", "addr_id", "address"]
    if "addr_id" not in pts_task.columns:
        pts_task["addr_id"] = np.arange(len(pts_task))  # fallback deterministic id
    pts_task[out_cols].to_csv(out_dir / "sfh_points.csv", index=False)

    # --- Post-join diagnostic for zero-target tasks ---
    summary = (
        pts_task.groupby(["interviewer","task"], dropna=False)
                .size().rename("post_pts").reset_index()
    )
    zeros = summary.loc[summary["post_pts"].eq(0)]
    if len(zeros):
        print("[emit][WARN] Zero-target interviewer-task(s) detected:")
        print(zeros.to_string(index=False))

        # Segment-level detail, including route segments that got 0 points
        route_segs = routes[["interviewer","task","segment_id"]].drop_duplicates()
        seg_post = (pts_task.groupby(["interviewer","task","segment_id"], dropna=False)
                            .size().rename("post_pts").reset_index())
        dbg = zeros.merge(route_segs, on=["interviewer","task"], how="left")                    .merge(seg_post, on=["interviewer","task","segment_id"], how="left")                    .fillna({"post_pts": 0}).astype({"post_pts": int})
        dbg_path = out_dir / "debug_zero_tasks.csv"
        dbg.to_csv(dbg_path, index=False)
        print(f"[emit][WARN] Wrote segment-level debug to {dbg_path}")

    # ---- starts.csv (Per-task counts from the filtered points) ----
    sfh_counts = (
        pts_task.groupby(["interviewer", "task"], as_index=False)
                .size()
                .rename(columns={"size": "n_sfh_targets"})
    )

    starts = routes.loc[routes["start_flag"] == 1, ["date", "interviewer", "task", "segment_id"]]
    starts = starts.merge(sfh_counts, on=["interviewer", "task"], how="left")
    starts["n_sfh_targets"] = starts["n_sfh_targets"].fillna(0).astype(int)

    # Warn if any interviewer-task ended with zero targets
    zero_rows = starts.loc[starts["n_sfh_targets"] == 0]
    if not zero_rows.empty:
        print("[emit][WARN] The following interviewer-task(s) have zero valid SFH targets:")
        print(zero_rows[["interviewer", "task", "segment_id"]].to_string(index=False))

    # Find starting address: use segment endpoint and find nearest address
    # Get segment geometries for start segments
    start_seg_ids = starts["segment_id"].astype(str).values
    segs_start = streets_m.loc[
        streets_m[streets_key_global].astype(str).isin(start_seg_ids),
        [streets_key_global, "geometry"]
    ].copy()
    segs_start.rename(columns={streets_key_global: "segment_id"}, inplace=True)
    segs_start["segment_id"] = segs_start["segment_id"].astype(str)

    # Extract first endpoint (start point) of each segment
    def get_segment_start_point(geom):
        """Extract the first coordinate (start point) from a LineString."""
        from shapely.geometry import Point
        if geom.geom_type == 'LineString':
            return Point(geom.coords[0])
        elif geom.geom_type == 'MultiLineString':
            # For MultiLineString, use first point of first line
            return Point(geom.geoms[0].coords[0])
        else:
            # Fallback: use centroid
            return geom.centroid

    segs_start["start_point"] = segs_start["geometry"].apply(get_segment_start_point)

    # Merge start points with starts table
    starts = starts.merge(
        segs_start[["segment_id", "start_point"]],
        on="segment_id",
        how="left"
    )

    # For each start point, find nearest address from pts_task
    # Need to use projected CRS (meters) for accurate distance calculation
    # Get pts_task in meters CRS
    pts_task_m = pts_task.copy()
    if pts_task_m.crs and pts_task_m.crs != w_m:
        pts_task_m = pts_task_m.to_crs(w_m)

    start_addresses = []
    for idx, row in starts.iterrows():
        ivw, task = row["interviewer"], row["task"]
        start_pt = row["start_point"]  # Already in meters (from streets_m)

        # Get addresses for this interviewer-task (in meters CRS)
        task_addrs = pts_task_m[
            (pts_task_m["interviewer"] == ivw) &
            (pts_task_m["task"] == task)
        ].copy()

        if task_addrs.empty or start_pt is None:
            start_addresses.append("")
            continue

        # Calculate distance from start point to each address (in meters)
        task_addrs["_dist"] = task_addrs.geometry.distance(start_pt)

        # Get closest address
        closest_idx = task_addrs["_dist"].idxmin()
        # Get address from original pts_task (which has the formatted address)
        closest_address = pts_task.loc[closest_idx, "address"]
        start_addresses.append(closest_address)

    starts["address"] = start_addresses
    starts.drop(columns=["start_point"], inplace=True)
    starts.to_csv(out_dir / "starts.csv", index=False)

    # ---- Generate segment-level comprehensive CSV ----
    _generate_segment_analysis(routes, date, out_dir, root)

    print(f"[emit] wrote {out_dir}")
