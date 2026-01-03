from pathlib import Path
from typing import Optional

import folium  # make sure this is in requirements.txt

from sd311_fieldprep.utils import (
    load_sources,
    project_to,
    load_filters_layers,
    apply_spatial_filters,
    addressable_mask,
)
from sd311_fieldprep.emit import _compose_address


def export_address_crosswalk(
    out_csv: str,
    map_html: Optional[str] = None,
):
    """
    Generate a 2-column CSV: addr_id, address, restricted to the same
    zoning/CPD + addressability + SFH-like filters used in the main pipeline.

    Optionally also export an HTML map with all included addresses.
    """
    # load_sources() returns (streets, addrs, datasources, params)
    streets, addrs, ds, pr = load_sources()

    work_epsg = int((pr.get("crs", {}) or {}).get("working_meters", 26911))

    # Project to working CRS
    addrs_m = project_to(addrs, work_epsg).copy()

    # --- Apply zoning + CPD filters ---
    zoning_m, cpd_m, fcfg = load_filters_layers(ds, pr, work_epsg)
    addrs_f = apply_spatial_filters(addrs_m, zoning_m, cpd_m, fcfg)

    # Drop non-addressable
    mask_addr = addressable_mask(addrs_f)
    addrs_f = addrs_f.loc[mask_addr].copy()

    # Compose address + __unit_blank__ flag (same helper as emit)
    _compose_address(addrs_f)

    # Keep only SFH-like addresses (no explicit unit string) if flag exists
    if "__unit_blank__" in addrs_f.columns:
        addrs_f = addrs_f.loc[addrs_f["__unit_blank__"]].copy()

    # ---------- Crosswalk ----------
    xw = (
        addrs_f[["addr_id", "address"]]
        .drop_duplicates(subset=["addr_id"])
        .reset_index(drop=True)
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    xw.to_csv(out_csv, index=False)
    print(f"[crosswalk] wrote: {out_csv}  (rows={len(xw)})")

    # ---------- Optional HTML map ----------
    if map_html is not None:
        # Reproject to WGS84 for web mapping
        addrs_ll = project_to(addrs_f, 4326).copy()

        # Center map on mean lat/lon
        center = [
            float(addrs_ll.geometry.y.mean()),
            float(addrs_ll.geometry.x.mean()),
        ]

        m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

        # One marker per address, popup with addr_id + address
        for row in addrs_ll.itertuples():
            geom = row.geometry
            popup_html = f"{getattr(row, 'addr_id', '')} — {getattr(row, 'address', '')}"
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=3,
                popup=popup_html,
                weight=0,
                fill=True,
                fill_opacity=0.7,
            ).add_to(m)

        map_html = Path(map_html)
        map_html.parent.mkdir(parents=True, exist_ok=True)
        m.save(map_html)
        print(f"[crosswalk] wrote map: {map_html}  (points={len(addrs_ll)})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="Path to output CSV, e.g. outputs/crosswalk/address_crosswalk.csv",
    )
    ap.add_argument(
        "--map-html",
        help="Optional HTML map output path, e.g. outputs/crosswalk/address_crosswalk_map.html",
    )

    args = ap.parse_args()
    export_address_crosswalk(
        out_csv=args.out,
        map_html=args.map_html,
    )
