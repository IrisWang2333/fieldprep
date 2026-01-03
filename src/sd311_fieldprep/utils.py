# fieldprep/src/sd311_fieldprep/utils.py
from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd

def project_to(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None or (gdf.crs.to_epsg() or 0) != epsg:
        return gdf.to_crs(epsg)
    return gdf

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def paths():
    root = Path(__file__).resolve().parents[2]  # .../fieldprep
    cfg = root / "config"
    out = root / "outputs"
    return root, cfg, out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_layer(entry: dict):
    g = gpd.read_file(entry["path"])
    if entry.get("crs_input"):
        g.set_crs(epsg=int(entry["crs_input"]), inplace=True, allow_override=True)
    return g

def _ensure_col(gdf, want_col, candidates):
    if want_col and want_col in gdf.columns:
        return want_col
    for c in candidates:
        if c in gdf.columns:
            gdf.rename(columns={c: want_col or c}, inplace=True)
            return want_col or c
    # synthesize if none found
    newcol = want_col or "tmp_id"
    gdf[newcol] = range(1, len(gdf) + 1)
    return newcol

def load_sources():
    _, cfg, _ = paths()
    ds = load_yaml(cfg / "datasources.yaml")
    pr = load_yaml(cfg / "params.yaml") or {}

    streets_entry = ds.get("streets") or ds.get("streets_segments")
    addrs_entry   = ds.get("addresses") or ds.get("address_points")
    if not streets_entry or not addrs_entry:
        raise SystemExit("datasources.yaml must define streets/streets_segments AND addresses/address_points")

    streets = _read_layer(streets_entry)
    addrs   = _read_layer(addrs_entry)

    # Resolve desired field names (from config) ------------------------------
    seg_id_want  = streets_entry.get("id_field") or pr.get("fields", {}).get("streets_segment_id") or "segment_id"
    addr_id_want = addrs_entry.get("id_field")   or pr.get("fields", {}).get("addresses_id")       or "OBJECTID"
    addr_tx      = addrs_entry.get("text_field") or pr.get("fields", {}).get("addresses_text")     or "address"

    # Streets: keep your existing normalization via _ensure_col --------------
    seg_id = _ensure_col(
        streets, seg_id_want,
        ["segment_id","SEGMENT_ID","StreetSegID","SEGMENTID","SegmentID"]
    )

    # Addresses: do NOT rename the raw id; instead, create a stable alias ----
    lower_map = {c.lower(): c for c in addrs.columns}

    # Find the raw id column case-insensitively (prefer config if present)
    raw_addr_col = lower_map.get(str(addr_id_want).lower())
    if not raw_addr_col:
        for cand in ("objectid","addressid","address_id","addr_id"):
            if cand in lower_map:
                raw_addr_col = lower_map[cand]
                break
    if not raw_addr_col:
        raise SystemExit("Could not find an address ID column (e.g. OBJECTID) in the address layer.")

    # Create the canonical alias column (preserve original raw column)
    addrs["addr_id"] = addrs[raw_addr_col]
    addr_id = "addr_id"  # <- unified name used by all downstream code

    # Prepare fields in pr so callers read consistent keys -------------------
    pr.setdefault("fields", {})
    pr["fields"]["streets_segment_id"] = seg_id
    pr["fields"]["addresses_id"]       = addr_id
    if addr_tx and addr_tx in addrs.columns:
        pr["fields"]["addresses_text"] = addr_tx


    return streets, addrs, ds, pr

# -------- filters support --------

_ZONING_CODE_CANDIDATES = ["ZONE","ZONING","BASE_ZONE","ZONING_CODE","ZONE_CODE","ZONING_TY","ZONINGTYPE"]
_CPD_NAME_CANDIDATES    = ["COMMUNITY","COMMPLAN","CMNTY_PLN","CPD","CPD_NAME","NAME","COMM_PLAN"]

def _read_list_from_csv(csv_path: str, colname: str):
    df = pd.read_csv(csv_path)
    if colname not in df.columns:
        # try to auto-find a one-column CSV
        if df.shape[1] == 1:
            colname = df.columns[0]
        else:
            raise ValueError(f"{csv_path} must contain column '{colname}'")
    return set(str(x).strip() for x in df[colname].dropna().unique())

def load_filters_layers(ds, pr, target_epsg: int):
    """Load optional zoning & cpd layers, reproject to target_epsg, and parse filters config."""
    zoning_gdf = None
    cpd_gdf = None
    if ds.get("zoning") and ds["zoning"].get("path"):
        zoning_gdf = _read_layer(ds["zoning"])
        zoning_gdf = project_to(zoning_gdf, target_epsg)
    if ds.get("cpd") and ds["cpd"].get("path"):
        cpd_gdf = _read_layer(ds["cpd"])
        cpd_gdf = project_to(cpd_gdf, target_epsg)

    cfg = pr.get("filters", {}) if pr else {}
    zcfg = cfg.get("zoning", {}) if cfg else {}
    ccfg = cfg.get("cpd", {}) if cfg else {}

    # zoning codes set
    zoning_codes = set()
    if zcfg.get("codes_csv"):
        zoning_codes = _read_list_from_csv(zcfg["codes_csv"], "code")
    elif zcfg.get("codes_list"):
        zoning_codes = set(str(x).strip() for x in zcfg["codes_list"])

    # cpd exclude set
    cpd_excl = set()
    if ccfg.get("exclude_csv"):
        cpd_excl = _read_list_from_csv(ccfg["exclude_csv"], "name")
    elif ccfg.get("exclude_list"):
        cpd_excl = set(str(x).strip() for x in ccfg["exclude_list"])

    return zoning_gdf, cpd_gdf, {
        "zoning": {
            "mode": zcfg.get("mode","off"),
            "codes": zoning_codes,
            "code_field": zcfg.get("code_field",""),
            "drop_if_no_zoning": bool(zcfg.get("drop_if_no_zoning", False)),
        },
        "cpd": {
            "exclude": cpd_excl,
            "name_field": ccfg.get("name_field",""),
            "drop_if_no_cpd": bool(ccfg.get("drop_if_no_cpd", False)),
        }
    }

def apply_spatial_filters(addrs_m: gpd.GeoDataFrame,
                          zoning_m: gpd.GeoDataFrame | None,
                          cpd_m: gpd.GeoDataFrame | None,
                          fcfg: dict,
                          report: dict | None = None) -> gpd.GeoDataFrame:
    """Return a filtered copy of addrs_m according to zoning/CPD config. Uses point-in-polygon sjoins."""
    if report is None:
        report = {}
    g = addrs_m.copy()
    n0 = len(g)
    report["addresses_initial"] = n0

    # ZONING
    zmode = fcfg.get("zoning", {}).get("mode", "off")
    zcodes = fcfg.get("zoning", {}).get("codes", set())
    drop_no_z = fcfg.get("zoning", {}).get("drop_if_no_zoning", False)
    if zmode in ("include","exclude") and zoning_m is not None and len(zoning_m):
        # choose code field
        zcf = fcfg.get("zoning", {}).get("code_field") or next((c for c in _ZONING_CODE_CANDIDATES if c in zoning_m.columns), None)
        if not zcf:
            raise SystemExit("Zoning layer present but no code field found. Set filters.zoning.code_field in params.yaml.")
        zjoin = gpd.sjoin(g, zoning_m[[zcf,"geometry"]], how="left", predicate="intersects")
        zcodes_series = zjoin[zcf].astype(str)
        if zmode == "include":
            mask = zcodes_series.isin(zcodes)
            if drop_no_z:
                mask = mask & zcodes_series.notna()
        else:  # exclude
            mask = ~zcodes_series.isin(zcodes)
            if drop_no_z:
                mask = mask & zcodes_series.notna()
        g = zjoin.loc[mask, g.columns]
        report["zoning_mode"] = zmode
        report["zoning_codes_n"] = len(zcodes)
        report["addresses_after_zoning"] = len(g)

    # CPD
    cex = fcfg.get("cpd", {}).get("exclude", set())
    drop_no_c = fcfg.get("cpd", {}).get("drop_if_no_cpd", False)
    if cpd_m is not None and len(cex) and len(cpd_m):
        cnf = fcfg.get("cpd", {}).get("name_field") or next((c for c in _CPD_NAME_CANDIDATES if c in cpd_m.columns), None)
        if not cnf:
            raise SystemExit("CPD layer present but no name field found. Set filters.cpd.name_field in params.yaml.")
        cjoin = gpd.sjoin(g, cpd_m[[cnf,"geometry"]], how="left", predicate="intersects")
        cseries = cjoin[cnf].astype(str)
        mask = ~cseries.isin(cex)
        if drop_no_c:
            mask = mask & cseries.notna()
        g = cjoin.loc[mask, g.columns]
        report["cpd_exclude_n"] = len(cex)
        report["addresses_after_cpd"] = len(g)

    report["addresses_final"] = len(g)
    report["addresses_filtered"] = n0 - len(g)
    return g

def addressable_mask(addrs_df):
    """
    True for rows that have usable address info (nonzero number AND nonblank street name).
    Drop when street name is blank OR number is in {0, 0000, ""}.
    """
    import pandas as pd

    # case-insensitive column lookup
    cols = {c.lower(): c for c in addrs_df.columns}
    namecol = cols.get("addrname")
    numcol  = cols.get("addrnmbr")

    # If we can't even find a street name column, fail open (keep all)
    if not namecol:
        return pd.Series(True, index=addrs_df.index)

    name = addrs_df[namecol].astype(str).str.strip()
    name_blank = name.eq("")

    if numcol:
        num = addrs_df[numcol].astype(str).str.strip()
        # DataSD uses 0/0000/"" as "unknown" numbers
        num_missing = num.isin(["", "0", "0000"])
    else:
        # If no house-number column is present, treat as missing to be conservative
        num_missing = pd.Series(True, index=addrs_df.index)

    # Strict policy: drop if name is blank OR number missing/zero
    drop = name_blank | num_missing
    return ~drop



def folium_map(gdf, color_col=None, tooltip_cols=None, zoom_start=12):
    import folium
    g = gdf.to_crs(4326)

    # Default center: downtown San Diego (fallback if layer empty)
    default_center = [32.7157, -117.1611]
    center = default_center

    # Use total_bounds (no centroid warning; works for Lines/Polys/Points)
    if len(g) and not g.geometry.is_empty.all():
        minx, miny, maxx, maxy = g.total_bounds
        if all(map(lambda v: v == v, [minx, miny, maxx, maxy])):  # not NaN
            center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    # If empty, just return a basemap
    if len(g) == 0:
        return m

    # Style
    if color_col and color_col in g.columns:
        keys = sorted([k for k in g[color_col].dropna().unique().tolist()])
        palette = {
            k: "#{:02x}{:02x}{:02x}".format(
                (hash((k, 1)) & 255), (hash((k, 2)) & 255), (hash((k, 3)) & 255)
            )
            for k in keys
        }

        def style_func(feat):
            k = feat["properties"].get(color_col)
            return {"color": palette.get(k, "#333333"), "weight": 3, "opacity": 0.8}
    else:
        def style_func(_):
            return {"color": "#1f78b4", "weight": 3, "opacity": 0.8}

    folium.GeoJson(
        g,
        tooltip=folium.GeoJsonTooltip(fields=[c for c in (tooltip_cols or []) if c in g.columns]),
        style_function=style_func,
    ).add_to(m)
    return m

def load_city_boundary(ds, target_epsg: int):
    """Optionally load a city boundary polygon and project it."""
    ent = ds.get("city_boundary")
    if not ent or not ent.get("path"):
        return None
    try:
        g = _read_layer(ent)
        return project_to(g, target_epsg)
    except FileNotFoundError:
        print("[map] city_boundary path not found; skipping boundary layer.")
        return None
    except Exception as e:
        print(f"[map] city_boundary load failed: {e}; skipping boundary layer.")
        return None
