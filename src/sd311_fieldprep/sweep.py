from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import folium

from sd311_fieldprep.utils import (
    load_sources, ensure_dir, project_to, paths,
    load_filters_layers, apply_spatial_filters, load_city_boundary
)
from sd311_fieldprep.emit import _compose_address

def _compute_seg_length(streets_m):
    # handle MultiLineString by length sum
    return streets_m.geometry.length

def _nearest_join(addrs_m, streets_m, max_dist_m, seg_id_col):
    """
    Join addresses to street segments with name-based matching priority.

    Strategy:
    1. Find all candidate segments within max_dist_m buffer
    2. For addresses with multiple candidates, prefer name matches
    3. Fall back to distance-based selection
    """
    try:
        # Get all columns needed for name matching
        street_cols = [seg_id_col, "geometry"]
        if "rd20full" in streets_m.columns:
            street_cols.append("rd20full")

        # Find ALL candidates within buffer (not just nearest)
        joined = gpd.sjoin_nearest(addrs_m, streets_m[street_cols],
                                   how="inner", max_distance=max_dist_m, distance_col="_dist")
    except Exception as e:
        raise RuntimeError("Nearest join requires a spatial index (rtree). Try: pip install rtree") from e

    # If no name fields available, return as-is
    if "rd20full" not in streets_m.columns or "addrname" not in addrs_m.columns:
        print("[sweep] Warning: Street name matching unavailable (missing rd20full or addrname)")
        return joined

    # For addresses with multiple candidates, apply name matching
    addr_index = addrs_m.index.name or "index"
    if addr_index == "index":
        joined["_addr_idx"] = joined.index
    else:
        joined["_addr_idx"] = joined.index

    # Count candidates per address
    candidates_count = joined.groupby("_addr_idx").size()
    multi_candidate_addrs = candidates_count[candidates_count > 1].index

    if len(multi_candidate_addrs) == 0:
        # No ambiguous cases, return as-is
        return joined.drop(columns=["_addr_idx"])

    print(f"[sweep] Name-matching: {len(multi_candidate_addrs)} addresses have multiple candidates")

    # Normalize street names for comparison
    def normalize_street_name(name):
        if pd.isna(name) or name == "":
            return ""
        name = str(name).upper().strip()
        # Remove common abbreviations and punctuation
        name = name.replace("STREET", "ST").replace("AVENUE", "AVE").replace("BOULEVARD", "BLVD")
        name = name.replace("ROAD", "RD").replace("DRIVE", "DR").replace("LANE", "LN")
        name = name.replace("PLACE", "PL").replace("COURT", "CT").replace("CIRCLE", "CIR")
        name = name.replace(".", "").replace(",", "")
        return name

    joined["_street_norm"] = joined["rd20full"].apply(normalize_street_name)
    joined["_addr_norm"] = joined["addrname"].apply(normalize_street_name)

    # Mark name matches - check if either contains the other
    def check_name_match(row):
        street = row["_street_norm"]
        addr = row["_addr_norm"]
        if not street or not addr:
            return False
        return addr in street or street in addr

    joined["_name_match"] = joined.apply(check_name_match, axis=1)

    # For multi-candidate addresses, prefer name matches
    selected_indices = []
    name_matched_count = 0
    for addr_idx in multi_candidate_addrs:
        addr_candidates = joined[joined["_addr_idx"] == addr_idx]

        # First try: name matches
        name_matches = addr_candidates[addr_candidates["_name_match"]]
        if len(name_matches) > 0:
            # If multiple name matches, pick closest
            best_idx = name_matches["_dist"].idxmin()
            selected_indices.append(best_idx)
            name_matched_count += 1
        else:
            # No name match, use nearest
            best_idx = addr_candidates["_dist"].idxmin()
            selected_indices.append(best_idx)

    # Keep single-candidate addresses as-is
    single_candidate = joined[~joined["_addr_idx"].isin(multi_candidate_addrs)]

    # Combine results - select the chosen rows for multi-candidate addresses
    if len(selected_indices) > 0:
        multi_selected = joined.loc[selected_indices]
        result = pd.concat([single_candidate, multi_selected], ignore_index=False)
    else:
        result = single_candidate

    # Cleanup helper columns
    result = result.drop(columns=["_addr_idx", "_street_norm", "_addr_norm", "_name_match"])

    print(f"[sweep] Name-matching: {name_matched_count}/{len(multi_candidate_addrs)} resolved by name, {len(multi_candidate_addrs)-name_matched_count} by distance")

    return result

def build_layered_map(
    streets_all_wgs: gpd.GeoDataFrame,
    eligible_wgs: gpd.GeoDataFrame,
    zoning_wgs: gpd.GeoDataFrame | None,
    fcfg: dict,
    city_wgs: gpd.GeoDataFrame | None,
    seg_id_col: str,
    zoom_start: int = 12
) -> folium.Map:
    # Center on bounds of either eligible or all streets
    center = [32.7157, -117.1611]
    base = eligible_wgs if len(eligible_wgs) else streets_all_wgs
    if len(base):
        minx, miny, maxx, maxy = base.total_bounds
        center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    # City boundary outline
    if city_wgs is not None and len(city_wgs):
        folium.GeoJson(
            city_wgs,
            name="City boundary",
            style_function=lambda _f: {"color": "#000000", "weight": 2, "opacity": 0.8, "fill": False},
        ).add_to(m)

    # Zoning shading
    zmode = fcfg.get("zoning", {}).get("mode", "off")
    zcodes = fcfg.get("zoning", {}).get("codes", set())
    zcode_field = fcfg.get("zoning", {}).get("code_field") or None
    if zmode in ("include", "exclude") and zoning_wgs is not None and len(zoning_wgs):
        if not zcode_field:
            for cand in ["ZONE","ZONING","BASE_ZONE","ZONING_CODE","ZONE_CODE","ZONING_TY","ZONINGTYPE"]:
                if cand in zoning_wgs.columns:
                    zcode_field = cand
                    break
        if zcode_field and zcode_field in zoning_wgs.columns:
            if zmode == "include":
                inc = zoning_wgs[zoning_wgs[zcode_field].astype(str).isin(zcodes)]
                exc = zoning_wgs[~zoning_wgs[zcode_field].astype(str).isin(zcodes)]
            else:
                exc = zoning_wgs[zoning_wgs[zcode_field].astype(str).isin(zcodes)]
                inc = zoning_wgs[~zoning_wgs[zcode_field].astype(str).isin(zcodes)]
            if len(inc):
                folium.GeoJson(
                    inc,
                    name="Zoning (included)",
                    style_function=lambda _f: {"color": "#2ca25f", "weight": 1, "opacity": 0.6,
                                              "fill": True, "fillColor": "#2ca25f", "fillOpacity": 0.3},
                ).add_to(m)
            if len(exc):
                folium.GeoJson(
                    exc,
                    name="Zoning (excluded)",
                    style_function=lambda _f: {"color": "#de2d26", "weight": 1, "opacity": 0.9,
                                              "fill": True, "fillColor": "#de2d26", "fillOpacity": 0.7},
                ).add_to(m)

    # Streets excluded vs eligible
    eligible_ids = set(eligible_wgs[seg_id_col].astype(str)) if seg_id_col in eligible_wgs.columns else set()
    if seg_id_col in streets_all_wgs.columns:
        streets_all_wgs["__id_str"] = streets_all_wgs[seg_id_col].astype(str)
        excl = streets_all_wgs[~streets_all_wgs["__id_str"].isin(eligible_ids)]
        if len(excl):
            folium.GeoJson(
                excl[[seg_id_col,"geometry"]],
                name="Segments (excluded)",
                tooltip=folium.GeoJsonTooltip(fields=[seg_id_col]),
                style_function=lambda _f: {"color": "#c7c7c7", "weight": 2, "opacity": 0.8},
            ).add_to(m)
    if len(eligible_wgs):
        folium.GeoJson(
            eligible_wgs[[seg_id_col,"sfh_addr_count","addr_density_per_100m","geometry"]],
            name="Segments (eligible)",
            tooltip=folium.GeoJsonTooltip(fields=[seg_id_col,"sfh_addr_count","addr_density_per_100m"]),
            style_function=lambda _f: {"color": "#1f78b4", "weight": 3, "opacity": 0.9},
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

def run_sweep(buffers, mins, tag="default"):
    streets, addrs, ds, pr = load_sources()
    seg_id = pr["fields"]["streets_segment_id"]

    work = int(pr.get("crs", {}).get("working_meters", 26911))
    out_wgs = int(pr.get("crs", {}).get("output_wgs84", 4326))
    zoom = int(pr.get("map", {}).get("zoom_start", 12))

    root, cfg, out_root = paths()
    sweep_dir = out_root / "sweep" / tag
    ensure_dir(sweep_dir)

    streets_m = project_to(streets, work).copy()
    addrs_m   = project_to(addrs,   work).copy()
    streets_m["seg_length_m"] = _compute_seg_length(streets_m)

    zoning_m, cpd_m, fcfg = load_filters_layers(ds, pr, work)
    filt_report = {}
    addrs_pref = apply_spatial_filters(addrs_m, zoning_m, cpd_m, fcfg, report=filt_report)

    # Drop non-addressable points before counting SFH addresses
    from sd311_fieldprep.utils import addressable_mask
    mask_addr = addressable_mask(addrs_pref)
    dropped = int((~mask_addr).sum())
    if dropped:
        print(f"[sweep] dropping {dropped} non-addressable addresses (missing number/name)")
    addrs_pref = addrs_pref.loc[mask_addr].copy()

    # Compose addresses and filter to SFH-only (no unit numbers)
    _compose_address(addrs_pref)
    if "__unit_blank__" in addrs_pref.columns:
        before_unit_filter = len(addrs_pref)
        addrs_pref = addrs_pref.loc[addrs_pref["__unit_blank__"]].copy()
        dropped_units = before_unit_filter - len(addrs_pref)
        if dropped_units:
            print(f"[sweep] dropping {dropped_units} multi-unit addresses (keeping SFH-only with blank unit field)")

    rows = []
    for B in buffers:
        for M in mins:
            join = _nearest_join(addrs_pref, streets_m, B, seg_id)
            counts = join.groupby(seg_id).size().rename("sfh_addr_count").to_frame()
            g = streets_m.merge(counts, left_on=seg_id, right_index=True, how="left").fillna({"sfh_addr_count":0})
            g["addr_density_per_100m"] = (g["sfh_addr_count"] / g["seg_length_m"].replace(0, np.nan)) * 100.0
            eligible = g.loc[g["sfh_addr_count"] >= M].copy()

            scen = f"b{int(B)}_m{int(M)}"
            out_parquet = sweep_dir / f"eligible_{scen}.parquet"
            eligible.to_parquet(out_parquet, index=False)

            # ---- Save address-to-segment assignment for emit to use ----
            # Get address ID field name
            addr_id_field = pr.get("fields", {}).get("addresses_id", "addr_id")

            # Filter join to only include eligible segments
            eligible_seg_ids = set(eligible[seg_id].astype(str))
            addr_assignment = join.loc[join[seg_id].astype(str).isin(eligible_seg_ids)].copy()

            # Keep only segment_id and addr_id columns
            addr_mapping = addr_assignment[[seg_id, addr_id_field]].copy()
            addr_mapping.columns = ["segment_id", "addr_id"]

            # Save the mapping
            addr_map_file = sweep_dir / f"segment_addresses_{scen}.parquet"
            addr_mapping.to_parquet(addr_map_file, index=False)
            print(f"[sweep] saved {len(addr_mapping)} address assignments to {addr_map_file.name}")

            row = {
                "scenario": scen,
                "addr_buffer_m": int(B),
                "sfh_min": int(M),
                "n_segments": int(len(eligible)),
                "sum_sfh_addresses": int(eligible["sfh_addr_count"].sum()),
                "median_seg_len_m": float(eligible["seg_length_m"].median() if len(eligible) else 0.0),
                "median_addr_density_per_100m": float(eligible["addr_density_per_100m"].median() if len(eligible) else 0.0),
                "addresses_initial": int(filt_report.get("addresses_initial", 0)),
                "addresses_after_zoning": int(filt_report.get("addresses_after_zoning", filt_report.get("addresses_initial", 0))),
                "addresses_after_cpd": int(filt_report.get("addresses_after_cpd", filt_report.get("addresses_after_zoning", 0))),
                "addresses_final": int(filt_report.get("addresses_final", 0)),
                "addresses_filtered": int(filt_report.get("addresses_filtered", 0)),
                "zoning_mode": fcfg.get("zoning",{}).get("mode","off"),
                "zoning_codes_n": int(len(fcfg.get("zoning",{}).get("codes", []))),
                "cpd_exclude_n": int(len(fcfg.get("cpd",{}).get("exclude", []))),
            }
            rows.append(row)

            try:
                streets_all_wgs = streets_m.to_crs(out_wgs)[[seg_id, "geometry"]].copy()
                eligible_wgs = eligible.to_crs(out_wgs).copy()
                city_m = load_city_boundary(ds, work)
                city_wgs = city_m.to_crs(out_wgs) if city_m is not None else None
                zoning_m2, _, fcfg_map = load_filters_layers(ds, pr, work)
                zoning_wgs = zoning_m2.to_crs(out_wgs) if zoning_m2 is not None else None
                m = build_layered_map(
                    streets_all_wgs=streets_all_wgs,
                    eligible_wgs=eligible_wgs,
                    zoning_wgs=zoning_wgs,
                    fcfg=fcfg_map,
                    city_wgs=city_wgs,
                    seg_id_col=seg_id,
                    zoom_start=zoom,
                )
                html = sweep_dir / f"eligible_{scen}_map.html"
                m.save(str(html))
            except Exception as e:
                print(f"[sweep] map skipped for {scen}: {e}")

    pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)
    print(f"[sweep] wrote {sweep_dir}")
