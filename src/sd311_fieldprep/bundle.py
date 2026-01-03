# fieldprep/src/sd311_fieldprep/bundle.py
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.ops import linemerge
from shapely import get_coordinates
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree


from sd311_fieldprep.utils import load_sources, ensure_dir, project_to, paths


# ---------- Geometry helpers ----------

def _snap_xy(pt, tol):
    return (round(pt[0]/tol)*tol, round(pt[1]/tol)*tol)

def _segment_endpoints(geom, snap_tol=0.5):
    # robust to MultiLineString
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

def _build_bundle_endpoint_index(g, snap_tol=0.5):
    """
    For each bundle_id, collect the set of snapped endpoints of its segments.
    Returns dict[bundle_id] -> set[(x,y)]
    """
    idx = {}
    for bid, sub in g.groupby("bundle_id"):
        S = set()
        for geom in sub.geometry:
            a, b = _segment_endpoints(geom, snap_tol)
            if a and b:
                S.add(_snap_xy(a, snap_tol))
                S.add(_snap_xy(b, snap_tol))
        idx[bid] = S
    return idx

def _endpoints_safe(geom):
    """Return (x0,y0),(x1,y1) for LineString or MultiLineString (longest part)."""
    if geom is None or geom.is_empty:
        return None
    g = geom
    if g.geom_type == "LineString":
        coords = list(g.coords)
        if len(coords) < 2:
            return None
        return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])
    elif g.geom_type == "MultiLineString":
        merged = linemerge(g)
        if merged.geom_type == "LineString":
            return _endpoints_safe(merged)
        parts = list(getattr(merged, "geoms", getattr(g, "geoms", [])))
        if not parts:
            return None
        longest = max(parts, key=lambda p: getattr(p, "length", 0.0))
        return _endpoints_safe(longest)
    return None


def _snap_point(pt, tol):
    """Snap a point to a square grid of size `tol` (meters)."""
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


# ---------- Graph/adjacency construction ----------

def _build_segment_neighbor_graph(segs_m: gpd.GeoDataFrame, join_tol_m: float):
    """
    Build segment neighbors via buffered intersections in meters.
    Only connect segments that BOTH have sfh_addr_count > 0 to avoid 'bridges'.
    Returns:
        nbrs: dict[int -> set[int]]
        keep_idx: list[int] (indices retained in 'g')
        nodes_by_seg: dict[int -> set[int]] (kept for API parity)
    """
    keep_idx = [i for i, geom in enumerate(segs_m.geometry) if geom is not None and not geom.is_empty]
    if not keep_idx:
        raise SystemExit("No valid line geometries found for bundling.")
    g = segs_m.iloc[keep_idx].copy()

    g["__res"] = g["sfh_addr_count"].fillna(0).astype(int) > 0

    bufs = [geom.buffer(float(join_tol_m)) for geom in g.geometry]
    tree = STRtree(bufs)

    nbrs = {g.index[i]: set() for i in range(len(g))}
    for i, b in enumerate(bufs):
        if not g.iloc[i]["__res"]:
            continue
        me = g.index[i]
        for j in tree.query(b):
            if j == i or not g.iloc[j]["__res"]:
                continue
            other = g.index[j]
            if bufs[i].intersects(bufs[j]):
                nbrs[me].add(other)

    # Limit to residential indices for downstream steps
    res_idx = [idx for idx in nbrs.keys()]
    g = g.loc[res_idx].copy()

    nodes_by_seg = {idx: set() for idx in g.index}
    return nbrs, list(g.index), nodes_by_seg


def _components_from_neighbors(nbrs: dict[int, set[int]]):
    """Return comp_id per segment index using BFS on the neighbor graph."""
    comp_of = {}
    comp_id = 0
    remaining = set(nbrs.keys())
    while remaining:
        root = remaining.pop()
        q = deque([root])
        comp_nodes = {root}
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if v in remaining:
                    remaining.remove(v)
                    comp_nodes.add(v)
                    q.append(v)
        for i in comp_nodes:
            comp_of[i] = comp_id
        comp_id += 1
    return comp_of


# ---------- Bundling within components ----------

def _grow_bundles_in_component(g, nbrs, comp_indices, target_addrs, rng):
    """
    Greedy BFS growth to target_addrs within one component.
    Returns dict seg_idx -> bundle_id (local to this component; caller offsets IDs).
    """
    remaining = set(comp_indices)
    order = list(comp_indices)
    rng.shuffle(order)

    out = {}
    next_bid = 1
    while remaining:
        # seed
        seed_idx = next(i for i in order if i in remaining)
        q, cur, total = deque([seed_idx]), [], 0
        remaining.remove(seed_idx)

        while q and total < target_addrs:
            u = q.popleft()
            cur.append(u)
            total += int(g.loc[u, "sfh_addr_count"])

            # prefer high-address neighbors to hit target with fewer segments
            for v in sorted(nbrs[u] & remaining,
                            key=lambda j: g.loc[j, "sfh_addr_count"],
                            reverse=True):
                remaining.remove(v)
                q.append(v)

        for i in cur:
            out[i] = next_bid
        next_bid += 1
    return out, next_bid


def _merge_tiny_bundles(g, min_bundle_sfh=None):
    """
    Merge tiny/singleton bundles into the nearest bundle *within the same component*.
    Falls back to global nearest; if no candidates exist, keep the tiny bundle.
    Mutates g['bundle_id'] in place and recomputes totals. Repeats until stable.
    """
    if "component_id" not in g.columns:
        g["component_id"] = 0

    changed = True
    max_iter = 10
    it = 0

    while changed and it < max_iter:
        it += 1
        changed = False

        # recompute per-bundle stats
        totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
        totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
        g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
        g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

        # centroids in meters
        geom_cent = g.geometry.centroid

        # Tiny if 1 segment or below threshold
        tiny_mask = (g["bundle_seg_count"] <= 1)
        if min_bundle_sfh is not None:
            tiny_mask = tiny_mask | (g["bundle_addr_total"] < int(min_bundle_sfh))

        tiny_bundles = g.loc[tiny_mask, ["bundle_id", "component_id"]].drop_duplicates()
        if tiny_bundles.empty:
            break

        # Precompute representative centroid per bundle
        sub_all = g[["bundle_id"]].copy()
        sub_all["_cent"] = geom_cent
        # Some groups can be empty if NaNs—guard with list() and skip if needed
        bundle_cent = sub_all.groupby("bundle_id")["_cent"].apply(
            lambda s: unary_union(list(s.dropna())).centroid if len(s.dropna()) else None
        )

        for bid, comp in tiny_bundles.itertuples(index=False, name=None):
            # candidate bundles in same component
            same_comp_bids = g.loc[(g["component_id"] == comp) & (g["bundle_id"] != bid), "bundle_id"].dropna().unique().tolist()

            # fallbacks
            cand = same_comp_bids if same_comp_bids else [b for b in bundle_cent.index if b != bid]

            # filter out candidates without a centroid
            cand = [b for b in cand if (b in bundle_cent.index and bundle_cent[b] is not None)]
            if not cand:
                # nothing to merge into -> keep this tiny bundle
                continue

            # centroid of the tiny bundle (union of its segment centroids)
            tiny_centroids = geom_cent[g["bundle_id"] == bid]
            if len(tiny_centroids.dropna()) == 0:
                # no geometry? skip
                continue
            src_cent = unary_union(list(tiny_centroids.dropna())).centroid

            # pick nearest candidate
            nearest_bid = min(cand, key=lambda b: src_cent.distance(bundle_cent[b]))

            # reassign all segments of tiny bundle to nearest
            g.loc[g["bundle_id"] == bid, "bundle_id"] = nearest_bid
            changed = True

    # Final recompute after merges
    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")
    return g


from shapely.ops import unary_union  # make sure this import exists at top

def _sweep_attach_residuals(g: gpd.GeoDataFrame,
                            soft_max_bundle_sfh: int | None = None,
                            snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """
    Attach unassigned residential segments to the nearest bundle in the SAME component,
    BUT ONLY if the segment shares an endpoint (snapped by snap_tol meters) with that bundle.
    This preserves route-style contiguity.
    """
    if "bundle_id" not in g.columns:
        return g

    unassigned = g["bundle_id"].isna()
    if not unassigned.any():
        return g

    # Precompute bundle centroids & totals among assigned
    cent = g.geometry.centroid
    assigned = g.loc[~g["bundle_id"].isna()].copy()
    if assigned.empty:
        return g  # nothing to attach to

    assigned["_cent"] = cent[~g["bundle_id"].isna()]
    bundle_cent = assigned.groupby("bundle_id")["_cent"].apply(lambda s: unary_union(list(s)).centroid)
    totals = assigned.groupby("bundle_id")["sfh_addr_count"].sum(min_count=1).to_dict()

    # Endpoint index per bundle (route-style)
    endpoint_idx = _build_bundle_endpoint_index(assigned, snap_tol=snap_tol)

    def can_accept(bid, add):
        if soft_max_bundle_sfh is None:
            return True
        return (totals.get(bid, 0) + add) <= soft_max_bundle_sfh * 1.10  # soft 10% overflow

    # Attach each residual segment if it shares an endpoint with a bundle
    for idx in g.index[unassigned]:
        # skip non-residential
        sfh_i = int(g.at[idx, "sfh_addr_count"]) if "sfh_addr_count" in g.columns else 0
        if sfh_i <= 0:
            continue

        comp = g.at[idx, "component_id"] if "component_id" in g.columns else None
        # candidate bundles: prefer same component
        cand = list(bundle_cent.index)
        if comp is not None and "component_id" in g.columns:
            same = g.loc[(~g["bundle_id"].isna()) & (g["component_id"] == comp), "bundle_id"].unique().tolist()
            if same:
                cand = [b for b in cand if b in same]

        # endpoints of this segment (snapped)
        a, b = _segment_endpoints(g.at[idx, "geometry"], snap_tol)
        if not a or not b:
            continue
        a = _snap_xy(a, snap_tol); b = _snap_xy(b, snap_tol)

        # filter candidates to those sharing an endpoint
        cand_touching = [bID for bID in cand if (a in endpoint_idx[bID] or b in endpoint_idx[bID])]
        if not cand_touching:
            continue  # do NOT attach if no endpoint match; keep it unassigned (avoids non-contiguity)

        # among touching candidates, choose nearest (by centroid)
        c0 = cent[idx]
        ranked = sorted(cand_touching, key=lambda bID: c0.distance(bundle_cent[bID]))
        chosen = None
        for bID in ranked:
            if can_accept(bID, sfh_i):
                chosen = bID
                break
        if chosen is None:
            chosen = ranked[0]  # last resort within touching set

        g.at[idx, "bundle_id"] = chosen
        totals[chosen] = totals.get(chosen, 0) + sfh_i
        # update endpoint index for chosen bundle
        endpoint_idx[chosen].add(a); endpoint_idx[chosen].add(b)

    # Recompute bundle totals
    totals2 = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals2 = totals2.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals2, left_on="bundle_id", right_index=True, how="left")
    return g


def _enforce_endpoint_contiguity(g: gpd.GeoDataFrame, snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """
    For each bundle_id, if its segments form >1 endpoint-connected component,
    split into separate bundle_ids (route-style contiguity). Recompute totals after.
    """
    def snap_xy(pt):
        return (round(pt[0]/snap_tol)*snap_tol, round(pt[1]/snap_tol)*snap_tol)

    next_bid = int(g["bundle_id"].max()) + 1 if g["bundle_id"].notna().any() else 1
    changed = False

    for bid, sub in g.groupby("bundle_id"):
        if pd.isna(bid) or sub.empty:
            continue

        # Build endpoint graph for this bundle (edges carry the segment idx)
        G = nx.Graph()
        edge_rows = []
        for idx, geom in zip(sub.index, sub.geometry):
            # robust endpoints (works for LineString / MultiLineString)
            a = b = None
            try:
                coords = np.asarray(geom.coords)
                if len(coords) >= 2:
                    a = (float(coords[0][0]), float(coords[0][1]))
                    b = (float(coords[-1][0]), float(coords[-1][1]))
            except Exception:
                from shapely.ops import linemerge
                m = linemerge(geom)
                try:
                    coords = np.asarray(m.coords)
                    if len(coords) >= 2:
                        a = (float(coords[0][0]), float(coords[0][1]))
                        b = (float(coords[-1][0]), float(coords[-1][1]))
                except Exception:
                    pass
            if a is None or b is None:
                continue
            a = snap_xy(a); b = snap_xy(b)
            G.add_edge(a, b, idx=int(idx))
            edge_rows.append(idx)

        # Nothing to do if no edges or already one component
        if G.number_of_edges() == 0 or nx.number_connected_components(G) <= 1:
            continue

        # Collect edge indices (segment ids) per connected component
        comps = []
        for comp_nodes in nx.connected_components(G):
            seg_idxs = [edata["idx"] for u, v, edata in G.edges(data=True)
                        if (u in comp_nodes and v in comp_nodes) and "idx" in edata]
            if seg_idxs:
                comps.append(seg_idxs)

        if len(comps) <= 1:
            continue

        # Keep the largest component on the original bundle id; split the rest
        comps.sort(key=len, reverse=True)
        largest = set(comps[0])
        for comp_seg_idxs in comps[1:]:
            g.loc[comp_seg_idxs, "bundle_id"] = next_bid
            next_bid += 1
            changed = True

    if changed:
        # Recompute totals safely
        totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
        totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
        g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
        g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    return g


def _build_connected_bundles(segs_m: gpd.GeoDataFrame, seg_id_col: str,
                             target_addrs: int, join_tol_m: float = 15.0,
                             seed: int = 42, min_bundle_sfh: int | None = None):
    """
    End-to-end:
      1) Build segment neighbor graph via endpoint snapping (meters).
      2) Find connected components and bundle *within* each component.
      3) Merge tiny/singleton bundles into nearest bundle in the same component.
      4) Sweep up any still-unassigned residential segments so we don't lose eligible edges.
    Returns a GeoDataFrame with per-segment bundle assignments and bundle totals.
    """
    # Build neighbor graph (segment-level)
    nbrs, keep_idx, _ = _build_segment_neighbor_graph(segs_m, join_tol_m)
    g = segs_m.iloc[keep_idx].copy()

    # Compute component ids (segment-level)
    comp_of = _components_from_neighbors(nbrs)
    g["component_id"] = g.index.map(comp_of.get)

    # Grow bundles within each component
    rng = np.random.default_rng(int(seed))
    bundle_id_global = 1
    out_ids = {}

    for comp in sorted(set(comp_of.values())):
        comp_indices = [i for i, c in comp_of.items() if c == comp]
        local_map, next_local = _grow_bundles_in_component(
            g, nbrs, comp_indices, target_addrs, rng
        )
        # offset local bundle ids to global space
        for i, local_bid in local_map.items():
            out_ids[i] = bundle_id_global + (local_bid - 1)
        bundle_id_global += (next_local - 1)

    g["bundle_id"] = g.index.map(out_ids.get)

    # Merge tiny/singletons (per component)
    g = _merge_tiny_bundles(g, min_bundle_sfh=min_bundle_sfh)

    # Sweep up any still-unassigned residential segments so we don't lose eligible edges
    g = _sweep_attach_residuals(g, soft_max_bundle_sfh=target_addrs)

    # NEW: route-style contiguity guarantee (split if needed)
    g = _enforce_endpoint_contiguity(g, snap_tol=0.5)

    # Order/columns
    cols = [seg_id_col, "sfh_addr_count", "bundle_id", "bundle_addr_total",
            "bundle_seg_count", "component_id", "geometry"]
    cols = [c for c in cols if c in g.columns] + [c for c in g.columns if c not in cols]
    return g[cols]


# ---------- CLI entry ----------

def run_bundle(session: str,
               target_addrs: int,
               join_tol_m: float = 15.0,
               seed: int = 42,
               tag: str | None = None,
               min_bundle_sfh: int | None = None):
    """
    Create bundles for a session (DH or D2DS).
    Parameters
    ----------
    session : str       e.g., "DH" or "D2DS"
    target_addrs : int  target # of SFH addresses per bundle
    join_tol_m : float  snapping tolerance in meters for endpoint adjacency
    seed : int          RNG seed for reproducible seeding
    tag : Optional[str] if set, read eligible_* under outputs/sweep/<tag>/
    min_bundle_sfh : Optional[int] merge bundles below this SFH into nearest bundle
    """
    streets, addrs, ds, pr = load_sources()
    seg_id = pr["fields"]["streets_segment_id"]

    # Pick eligible parquet from sweep (optionally scoped to a tag)
    root, cfg, out_root = paths()
    sweep_root = out_root / "sweep"
    if tag:
        sweep_root = sweep_root / tag
        if not sweep_root.exists():
            raise SystemExit(f"--tag '{tag}' not found at {sweep_root}")

    cands = sorted((p for p in sweep_root.rglob("eligible_*.parquet")),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        where = f"outputs/sweep/{tag}/" if tag else "outputs/sweep/"
        raise SystemExit(f"No eligible_*.parquet found in {where}. Run `cli.py sweep ...` first.")

    latest = cands[-1]  # most recent by mtime
    segs = gpd.read_parquet(latest)
    print(f"[bundle] using eligible parquet: {latest}")

    if seg_id not in segs.columns:
        for c in ("segment_id", "SEGMENT_ID"):
            if c in segs.columns:
                segs = segs.rename(columns={c: seg_id})
                break
        else:
            raise SystemExit(f"Eligible parquet {latest} lacks '{seg_id}' column.")

    # Keep only required columns
    need_cols = [seg_id, "sfh_addr_count", "geometry"]
    missing = [c for c in need_cols if c not in segs.columns]
    if missing:
        raise SystemExit(f"Eligible parquet missing columns: {missing}")

    segs = segs[need_cols].copy()

    # Project to working meters CRS from params.yaml (fallback 26911)
    work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))
    segs_m = project_to(segs, work_epsg)

    # Build connected bundles with merges for tiny/singletons
    bundled = _build_connected_bundles(
        segs_m,
        seg_id_col=seg_id,
        target_addrs=target_addrs,
        join_tol_m=join_tol_m,
        seed=seed,
        min_bundle_sfh=min_bundle_sfh
    )

    # Write outputs
    out_dir = root / "outputs" / "bundles" / session.upper()
    ensure_dir(out_dir)
    bundled.to_parquet(out_dir / "bundles.parquet", index=False)

    # Optional quick map (place this INSIDE run_bundle, after we define `bundled`)
    try:
        import folium
        from sd311_fieldprep.utils import folium_map

        # Only visualize rows that actually belong to a bundle
        viz = bundled.dropna(subset=["bundle_id"]).to_crs(4326)

        # Show bundle-level fields; hide per-segment sfh in the HTML
        tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

        m = folium_map(
            viz,
            color_col="bundle_id",
            tooltip_cols=tooltip_cols
        )
        m.save(str(out_dir / "bundles_map.html"))
    except Exception as e:
        print("[bundle] map skipped:", e)

    # Summary
    n_bundles = int(bundled["bundle_id"].nunique())
    # guard in case counts are missing (shouldn’t be)
    seg_counts = bundled.groupby("bundle_id")["bundle_seg_count"].first()
    if seg_counts.isna().any():
        seg_counts = bundled.groupby("bundle_id")["sfh_addr_count"].count()
    singleton_share = float((seg_counts == 1).mean())

    print(f"[bundle] wrote {out_dir} with {n_bundles} bundles "
          f"(singleton share ≈ {singleton_share:.3f}).")

