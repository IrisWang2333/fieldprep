import geopandas as gpd
from sd311_fieldprep.utils import (
    load_sources, project_to, load_filters_layers, apply_spatial_filters, paths
)

def run_doctor(buffer_m=40, mins=5):
    streets, addrs, ds, pr = load_sources()
    seg_id = pr["fields"]["streets_segment_id"]
    addr_id = pr["fields"]["addresses_id"]

    w_m = int(pr.get("crs", {}).get("working_meters", 26911))
    streets_m = project_to(streets, w_m)
    addrs_m   = project_to(addrs,   w_m)

    print("== CRS ==")
    print("streets:", streets.crs)
    print("addrs  :", addrs.crs)
    print("work EPSG:", w_m)

    print("\n== Bounds (meters) ==")
    print("streets:", streets_m.total_bounds)
    print("addrs  :", addrs_m.total_bounds)

    print("\n== Raw counts ==")
    print("streets segments:", len(streets))
    print("addresses       :", len(addrs))

    # Filters
    zoning_m, cpd_m, fcfg = load_filters_layers(ds, pr, w_m)
    addrs_f = apply_spatial_filters(addrs_m, zoning_m, cpd_m, fcfg, report={})
    print("\n== After filters ==")
    print("addresses_kept  :", len(addrs_f))

    # Nearest join diagnostic
    try:
        j = gpd.sjoin_nearest(addrs_f, streets_m[[seg_id, "geometry"]], how="inner", max_distance=buffer_m)
        print("\n== Nearest-join ==")
        print("matched addresses:", len(j))
        seg_counts = j.groupby(seg_id).size()
        elig = seg_counts[seg_counts >= mins]
        print("eligible segments (mins={}): {}".format(mins, len(elig)))
        print("sum SFH addresses on eligible:", int(elig.sum()))
    except Exception as e:
        print("Nearest-join failed:", repr(e))
