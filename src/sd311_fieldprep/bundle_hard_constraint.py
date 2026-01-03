# fieldprep/src/sd311_fieldprep/bundle_hard_constraint.py
"""
🔒 HARD CONSTRAINT BUNDLING VERSION 🔒

Two completely independent bundling algorithms:
1. GREEDY (5 steps, no constraints) - Standard bundle.py workflow
2. MULTI-BFS (9 steps, hard constraints + regroup) - New balanced workflow

Usage:
------
from sd311_fieldprep.bundle_hard_constraint import _build_connected_bundles

# Greedy (no constraints, 5 steps)
bundled = _build_connected_bundles(segs_m, seg_id, target_addrs, method="greedy")

# Multi-BFS (hard constraints + regroup, 10 steps)
bundled = _build_connected_bundles(segs_m, seg_id, target_addrs, method="multi_bfs", hard_max_multiplier=1.1)
"""

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


# ---------- Multi-source BFS Balanced Bundling ----------

def _select_spatially_distributed_seeds(g: gpd.GeoDataFrame,
                                       indices: list,
                                       n_seeds: int,
                                       rng) -> list:
    """
    Select spatially distributed seeds using K-means++ style algorithm.
    """
    if n_seeds <= 0 or not indices:
        return []

    if n_seeds >= len(indices):
        return indices[:n_seeds]

    seeds = []
    remaining = set(indices)

    # First seed: random selection
    first = rng.choice(list(remaining))
    seeds.append(first)
    remaining.remove(first)

    # Performance optimization: only use exact furthest-point for first 100 seeds
    exact_selection_limit = min(100, n_seeds - 1)

    for i in range(n_seeds - 1):
        if not remaining:
            break

        if i < exact_selection_limit:
            # Exact furthest-point sampling
            max_min_dist = -1
            best_candidate = None

            for candidate in remaining:
                cand_geom = g.loc[candidate].geometry
                min_dist = min(cand_geom.distance(g.loc[s].geometry) for s in seeds)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                seeds.append(best_candidate)
                remaining.remove(best_candidate)
        else:
            # Random sampling for remaining seeds
            next_seed = rng.choice(list(remaining))
            seeds.append(next_seed)
            remaining.remove(next_seed)

    return seeds


def _backtrack_rebalance(g: gpd.GeoDataFrame,
                        nbrs: dict,
                        assignment: dict,
                        sizes: dict,
                        target: int) -> dict:
    """Backtrack to rebalance: transfer edge segments from oversized to undersized bundles."""
    max_iter = 10
    tolerance = 0.15  # ±15%

    for iteration in range(max_iter):
        changed = False

        target_min = int(target * (1 - tolerance))
        target_max = int(target * (1 + tolerance))

        oversized = [bid for bid, size in sizes.items() if size > target_max]
        undersized = [bid for bid, size in sizes.items() if size < target_min]

        if not oversized or not undersized:
            break

        for over_bid in oversized:
            bundle_segs = [seg for seg, bid in assignment.items() if bid == over_bid]

            edge_candidates = []
            for seg in bundle_segs:
                same_bundle_neighbors = sum(1 for n in nbrs[seg] if assignment.get(n) == over_bid)
                if same_bundle_neighbors <= 2:
                    edge_candidates.append((seg, int(g.loc[seg, "sfh_addr_count"])))

            edge_candidates.sort(key=lambda x: x[1])

            for seg, seg_addrs in edge_candidates:
                if sizes[over_bid] <= target_max:
                    break

                best_under = None
                best_dist = float('inf')

                for under_bid in undersized:
                    if sizes[under_bid] + seg_addrs <= target_max:
                        has_neighbor = any(assignment.get(n) == under_bid for n in nbrs[seg])
                        if has_neighbor:
                            under_segs = [s for s, b in assignment.items() if b == under_bid]
                            under_cent = g.loc[under_segs].geometry.unary_union.centroid
                            dist = g.loc[seg].geometry.distance(under_cent)
                            if dist < best_dist:
                                best_dist = dist
                                best_under = under_bid

                if best_under is not None:
                    assignment[seg] = best_under
                    sizes[over_bid] -= seg_addrs
                    sizes[best_under] += seg_addrs
                    changed = True

        if not changed:
            break

    return assignment


def _multi_source_balanced_bfs(g: gpd.GeoDataFrame,
                                nbrs: dict,
                                comp_indices: list,
                                target_addrs: int,
                                seed: int) -> dict:
    """Multi-source BFS with balanced growth algorithm."""
    rng = np.random.default_rng(seed)

    total_addrs = g.loc[comp_indices, "sfh_addr_count"].sum()
    n_bundles = max(1, int(np.ceil(total_addrs / target_addrs)))
    print(f"    Component: {len(comp_indices)} segments, {int(total_addrs)} addrs → {n_bundles} bundles")

    print(f"    Selecting {n_bundles} spatially distributed seeds...", end=" ", flush=True)
    seeds = _select_spatially_distributed_seeds(g, comp_indices, n_bundles, rng)
    print("✓")

    bundle_assignment = {}
    bundle_sizes = {i: 0 for i in range(n_bundles)}
    bundle_frontiers = {i: deque() for i in range(n_bundles)}

    for bundle_id, seed_idx in enumerate(seeds):
        bundle_assignment[seed_idx] = bundle_id
        bundle_sizes[bundle_id] = int(g.loc[seed_idx, "sfh_addr_count"])
        for neighbor in nbrs[seed_idx]:
            if neighbor in comp_indices and neighbor not in bundle_assignment:
                bundle_frontiers[bundle_id].append(neighbor)

    remaining = set(comp_indices) - set(bundle_assignment.keys())
    print(f"    BFS expansion: {len(remaining)} segments to assign...", end=" ", flush=True)

    initial_remaining = len(remaining)
    last_progress = 0

    while remaining:
        grown = False

        for bundle_id in range(n_bundles):
            if not bundle_frontiers[bundle_id]:
                continue

            next_seg = None
            while bundle_frontiers[bundle_id]:
                candidate = bundle_frontiers[bundle_id].popleft()
                if candidate in remaining:
                    next_seg = candidate
                    break

            if next_seg is None:
                continue

            competing_bundles = []
            for other_id in range(n_bundles):
                if other_id != bundle_id and next_seg in bundle_frontiers[other_id]:
                    competing_bundles.append(other_id)

            if competing_bundles:
                all_candidates = [bundle_id] + competing_bundles
                winner = min(all_candidates, key=lambda bid: bundle_sizes[bid])
            else:
                winner = bundle_id

            bundle_assignment[next_seg] = winner
            bundle_sizes[winner] += int(g.loc[next_seg, "sfh_addr_count"])
            remaining.remove(next_seg)
            grown = True

            for neighbor in nbrs[next_seg]:
                if neighbor in remaining:
                    bundle_frontiers[winner].append(neighbor)

        progress = int(100 * (initial_remaining - len(remaining)) / initial_remaining)
        if progress >= last_progress + 10:
            print(f"{progress}%...", end=" ", flush=True)
            last_progress = progress

        if not grown:
            for seg in list(remaining):
                min_dist = float('inf')
                closest_bundle = 0
                for assigned_seg, bid in bundle_assignment.items():
                    dist = g.loc[seg].geometry.distance(g.loc[assigned_seg].geometry)
                    if dist < min_dist:
                        min_dist = dist
                        closest_bundle = bid
                bundle_assignment[seg] = closest_bundle
                bundle_sizes[closest_bundle] += int(g.loc[seg, "sfh_addr_count"])
                remaining.remove(seg)

    print("✓")

    print(f"    Rebalancing bundles...", end=" ", flush=True)
    bundle_assignment = _backtrack_rebalance(g, nbrs, bundle_assignment, bundle_sizes, target_addrs)
    print("✓")

    return bundle_assignment


# ========== GREEDY VERSION: NO CONSTRAINTS ==========

def _merge_tiny_bundles_no_constraint(g, min_bundle_sfh=None):
    """❌ NO CONSTRAINT VERSION - Merge based on distance only."""
    if "component_id" not in g.columns:
        g["component_id"] = 0

    changed = True
    max_iter = 10
    it = 0

    while changed and it < max_iter:
        it += 1
        changed = False

        totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
        totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
        g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
        g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

        geom_cent = g.geometry.centroid

        tiny_mask = (g["bundle_seg_count"] <= 1)
        if min_bundle_sfh is not None:
            tiny_mask = tiny_mask | (g["bundle_addr_total"] < int(min_bundle_sfh))

        tiny_bundles = g.loc[tiny_mask, ["bundle_id", "component_id"]].drop_duplicates()
        if tiny_bundles.empty:
            break

        sub_all = g[["bundle_id"]].copy()
        sub_all["_cent"] = geom_cent
        bundle_cent = sub_all.groupby("bundle_id")["_cent"].apply(
            lambda s: unary_union(list(s.dropna())).centroid if len(s.dropna()) else None
        )

        for bid, comp in tiny_bundles.itertuples(index=False, name=None):
            same_comp_bids = g.loc[(g["component_id"] == comp) & (g["bundle_id"] != bid), "bundle_id"].dropna().unique().tolist()
            cand = same_comp_bids if same_comp_bids else [b for b in bundle_cent.index if b != bid]
            cand = [b for b in cand if (b in bundle_cent.index and bundle_cent[b] is not None)]
            if not cand:
                continue

            tiny_centroids = geom_cent[g["bundle_id"] == bid]
            if len(tiny_centroids.dropna()) == 0:
                continue
            src_cent = unary_union(list(tiny_centroids.dropna())).centroid

            # ❌ NO CONSTRAINT: Pick nearest candidate (no size check)
            nearest_bid = min(cand, key=lambda b: src_cent.distance(bundle_cent[b]))
            g.loc[g["bundle_id"] == bid, "bundle_id"] = nearest_bid
            changed = True

    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")
    return g


def _sweep_attach_residuals(g: gpd.GeoDataFrame,
                            soft_max_bundle_sfh: int | None = None,
                            snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """Attach unassigned segments (soft constraint 1.1x)."""
    if "bundle_id" not in g.columns:
        return g

    unassigned = g["bundle_id"].isna()
    if not unassigned.any():
        return g

    cent = g.geometry.centroid
    assigned = g.loc[~g["bundle_id"].isna()].copy()
    if assigned.empty:
        return g

    assigned["_cent"] = cent[~g["bundle_id"].isna()]
    bundle_cent = assigned.groupby("bundle_id")["_cent"].apply(lambda s: unary_union(list(s)).centroid)
    totals = assigned.groupby("bundle_id")["sfh_addr_count"].sum(min_count=1).to_dict()

    endpoint_idx = _build_bundle_endpoint_index(assigned, snap_tol=snap_tol)

    def can_accept(bid, add):
        if soft_max_bundle_sfh is None:
            return True
        return (totals.get(bid, 0) + add) <= soft_max_bundle_sfh * 1.10  # soft 10% overflow

    for idx in g.index[unassigned]:
        sfh_i = int(g.at[idx, "sfh_addr_count"]) if "sfh_addr_count" in g.columns else 0
        if sfh_i <= 0:
            continue

        comp = g.at[idx, "component_id"] if "component_id" in g.columns else None
        cand = list(bundle_cent.index)
        if comp is not None and "component_id" in g.columns:
            same = g.loc[(~g["bundle_id"].isna()) & (g["component_id"] == comp), "bundle_id"].unique().tolist()
            if same:
                cand = [b for b in cand if b in same]

        a, b = _segment_endpoints(g.at[idx, "geometry"], snap_tol)
        if not a or not b:
            continue
        a = _snap_xy(a, snap_tol); b = _snap_xy(b, snap_tol)

        cand_touching = [bID for bID in cand if (a in endpoint_idx[bID] or b in endpoint_idx[bID])]
        if not cand_touching:
            continue

        c0 = cent[idx]
        ranked = sorted(cand_touching, key=lambda bID: c0.distance(bundle_cent[bID]))
        chosen = None
        for bID in ranked:
            if can_accept(bID, sfh_i):
                chosen = bID
                break
        if chosen is None:
            chosen = ranked[0]  # last resort

        g.at[idx, "bundle_id"] = chosen
        totals[chosen] = totals.get(chosen, 0) + sfh_i
        endpoint_idx[chosen].add(a); endpoint_idx[chosen].add(b)

    totals2 = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals2 = totals2.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals2, left_on="bundle_id", right_index=True, how="left")
    return g


def _enforce_endpoint_contiguity(g: gpd.GeoDataFrame, snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """Enforce route-style contiguity by splitting non-contiguous bundles."""
    def snap_xy(pt):
        return (round(pt[0]/snap_tol)*snap_tol, round(pt[1]/snap_tol)*snap_tol)

    next_bid = int(g["bundle_id"].max()) + 1 if g["bundle_id"].notna().any() else 1
    changed = False

    for bid, sub in g.groupby("bundle_id"):
        if pd.isna(bid) or sub.empty:
            continue

        G = nx.Graph()
        edge_rows = []
        for idx, geom in zip(sub.index, sub.geometry):
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

        if G.number_of_edges() == 0 or nx.number_connected_components(G) <= 1:
            continue

        comps = []
        for comp_nodes in nx.connected_components(G):
            seg_idxs = [edata["idx"] for u, v, edata in G.edges(data=True)
                        if (u in comp_nodes and v in comp_nodes) and "idx" in edata]
            if seg_idxs:
                comps.append(seg_idxs)

        if len(comps) <= 1:
            continue

        comps.sort(key=len, reverse=True)
        largest = set(comps[0])
        for comp_seg_idxs in comps[1:]:
            g.loc[comp_seg_idxs, "bundle_id"] = next_bid
            next_bid += 1
            changed = True

    if changed:
        totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
        totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
        g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
        g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    return g


# ========== MULTI-BFS VERSION: HARD CONSTRAINTS ==========

def _merge_tiny_bundles_hard_constraint(g, min_bundle_sfh=None, target_addrs=None, hard_max_multiplier=1.1):
    """🔒 HARD CONSTRAINT VERSION - Reject merges exceeding limit."""
    if "component_id" not in g.columns:
        g["component_id"] = 0

    hard_max = None
    if target_addrs is not None and hard_max_multiplier is not None:
        hard_max = int(target_addrs * hard_max_multiplier)
        print(f"    🔒 Hard constraint: max = {hard_max} addresses ({hard_max_multiplier}x)")

    changed = True
    max_iter = 10
    it = 0
    rejected_count = 0

    while changed and it < max_iter:
        it += 1
        changed = False

        totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
        totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
        g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
        g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

        geom_cent = g.geometry.centroid

        tiny_mask = (g["bundle_seg_count"] <= 1)
        if min_bundle_sfh is not None:
            tiny_mask = tiny_mask | (g["bundle_addr_total"] < int(min_bundle_sfh))

        tiny_bundles = g.loc[tiny_mask, ["bundle_id", "component_id", "bundle_addr_total"]].drop_duplicates("bundle_id")
        if tiny_bundles.empty:
            break

        sub_all = g[["bundle_id"]].copy()
        sub_all["_cent"] = geom_cent
        bundle_cent = sub_all.groupby("bundle_id")["_cent"].apply(
            lambda s: unary_union(list(s.dropna())).centroid if len(s.dropna()) else None
        )

        bundle_sizes = g.groupby("bundle_id")["sfh_addr_count"].sum().to_dict()

        for bid, comp, tiny_size in tiny_bundles.itertuples(index=False, name=None):
            same_comp_bids = g.loc[(g["component_id"] == comp) & (g["bundle_id"] != bid), "bundle_id"].dropna().unique().tolist()
            cand = same_comp_bids if same_comp_bids else [b for b in bundle_cent.index if b != bid]
            cand = [b for b in cand if (b in bundle_cent.index and bundle_cent[b] is not None)]
            if not cand:
                continue

            tiny_centroids = geom_cent[g["bundle_id"] == bid]
            if len(tiny_centroids.dropna()) == 0:
                continue
            src_cent = unary_union(list(tiny_centroids.dropna())).centroid

            # 🔒 HARD CONSTRAINT: Filter by size limit
            tiny_addrs = int(tiny_size)
            if hard_max is not None:
                valid_cand = []
                for b in cand:
                    if bundle_sizes.get(b, 0) + tiny_addrs <= hard_max:
                        valid_cand.append(b)

                if not valid_cand:
                    rejected_count += 1
                    continue
                cand = valid_cand

            nearest_bid = min(cand, key=lambda b: src_cent.distance(bundle_cent[b]))
            g.loc[g["bundle_id"] == bid, "bundle_id"] = nearest_bid
            bundle_sizes[nearest_bid] = bundle_sizes.get(nearest_bid, 0) + tiny_addrs
            changed = True

    if hard_max is not None and rejected_count > 0:
        print(f"    ⚠️  Kept {rejected_count} tiny bundles (would exceed {hard_max})")

    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")
    return g


def _merge_tiny_bundles_connected(g: gpd.GeoDataFrame,
                                   min_bundle_sfh: int | None = None,
                                   target_addrs: int | None = None,
                                   hard_max_multiplier: float | None = 1.1,
                                   snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """🔗 Connected merge - only merge to bundles sharing endpoints."""
    if min_bundle_sfh is None:
        return g

    hard_max = None
    if target_addrs is not None and hard_max_multiplier is not None:
        hard_max = int(target_addrs * hard_max_multiplier)

    endpoint_idx = _build_bundle_endpoint_index(g, snap_tol=snap_tol)
    bundle_sizes = g.groupby("bundle_id")["sfh_addr_count"].sum().to_dict()
    tiny_bundles = {bid: size for bid, size in bundle_sizes.items() if size < min_bundle_sfh}

    if not tiny_bundles:
        return g

    print(f"    🔗 Found {len(tiny_bundles)} tiny bundles, merging to connected neighbors...")

    merged_count = 0
    rejected_count = 0

    for tiny_bid, tiny_size in tiny_bundles.items():
        if tiny_bid not in endpoint_idx:
            continue

        tiny_endpoints = endpoint_idx.get(tiny_bid, set())
        if not tiny_endpoints:
            continue

        connected_candidates = []
        for bid in bundle_sizes.keys():
            if bid == tiny_bid or bid in tiny_bundles:
                continue

            other_endpoints = endpoint_idx.get(bid, set())
            if tiny_endpoints & other_endpoints:
                connected_candidates.append(bid)

        if not connected_candidates:
            rejected_count += 1
            continue

        if hard_max is not None:
            valid_candidates = [
                bid for bid in connected_candidates
                if bundle_sizes.get(bid, 0) + tiny_size <= hard_max
            ]

            if not valid_candidates:
                rejected_count += 1
                continue

            connected_candidates = valid_candidates

        chosen_bid = min(connected_candidates, key=lambda bid: bundle_sizes.get(bid, 0))

        g.loc[g["bundle_id"] == tiny_bid, "bundle_id"] = chosen_bid
        bundle_sizes[chosen_bid] = bundle_sizes.get(chosen_bid, 0) + tiny_size
        endpoint_idx[chosen_bid] |= tiny_endpoints
        del endpoint_idx[tiny_bid]
        merged_count += 1

    print(f"    ✅ Merged {merged_count}, kept {rejected_count} tiny bundles")

    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    return g


def _regroup_invalid_bundles(g: gpd.GeoDataFrame,
                             nbrs: dict,
                             target_addrs: int,
                             min_size: int,
                             max_size: int,
                             seed: int,
                             max_iterations: int = 5) -> gpd.GeoDataFrame:
    """
    🔄 Iteratively regroup invalid bundles (too small or too large).

    Loop:
      1. Find bundles outside [min_size, max_size]
      2. Extract their segments
      3. Regroup using greedy BFS
      4. Stop if no improvement or can't form valid bundles
    """
    iteration = 0
    total_regrouped = 0

    while iteration < max_iterations:
        iteration += 1

        # Calculate current bundle sizes
        bundle_sizes = g.groupby('bundle_id')['sfh_addr_count'].sum()

        # Find invalid bundles
        invalid_bundles = bundle_sizes[
            (bundle_sizes < min_size) | (bundle_sizes > max_size)
        ]

        if len(invalid_bundles) == 0:
            print(f"    ✅ All bundles within [{min_size}, {max_size}]")
            break

        # Check if we can continue
        total_addrs_to_regroup = g[g['bundle_id'].isin(invalid_bundles.index)]['sfh_addr_count'].sum()
        if total_addrs_to_regroup < min_size:
            print(f"    ⚠️  Cannot regroup ({int(total_addrs_to_regroup)} < {min_size}), stopping")
            break

        print(f"    🔄 Regroup iteration {iteration}: {len(invalid_bundles)} invalid bundles")

        # Extract segments from invalid bundles
        segments_to_regroup = g[g['bundle_id'].isin(invalid_bundles.index)].index.tolist()

        # Greedy BFS regrouping
        rng = np.random.default_rng(seed + iteration)
        remaining = set(segments_to_regroup)
        order = list(segments_to_regroup)
        rng.shuffle(order)

        assignment = {}
        next_bid = int(g['bundle_id'].max()) + 1
        regrouped_count = 0

        while remaining:
            # Select seed
            seed_idx = next((i for i in order if i in remaining), None)
            if seed_idx is None:
                break

            # BFS grow to target
            q = deque([seed_idx])
            cur = []
            total = 0
            remaining.remove(seed_idx)

            while q and total < target_addrs:
                u = q.popleft()
                cur.append(u)
                total += int(g.loc[u, "sfh_addr_count"])

                # Prefer high-address neighbors
                for v in sorted(nbrs.get(u, set()) & remaining,
                              key=lambda j: g.loc[j, "sfh_addr_count"],
                              reverse=True):
                    remaining.remove(v)
                    q.append(v)

            # Only assign if within valid range
            if total >= min_size and total <= max_size:
                for i in cur:
                    assignment[i] = next_bid
                next_bid += 1
                regrouped_count += 1

        if regrouped_count == 0:
            print(f"    ⚠️  No valid bundles formed, stopping")
            break

        # Update bundle_ids
        for seg_idx, bid in assignment.items():
            g.at[seg_idx, 'bundle_id'] = bid

        # Unassigned segments → NaN
        for seg_idx in remaining:
            g.at[seg_idx, 'bundle_id'] = np.nan

        total_regrouped += len(assignment)
        print(f"    ✅ Formed {regrouped_count} valid bundles from {len(segments_to_regroup)} segments")

        # Check improvement
        new_bundle_sizes = g.dropna(subset=['bundle_id']).groupby('bundle_id')['sfh_addr_count'].sum()
        new_invalid = new_bundle_sizes[(new_bundle_sizes < min_size) | (new_bundle_sizes > max_size)]

        if len(new_invalid) >= len(invalid_bundles):
            print(f"    ⚠️  No improvement, stopping")
            break

    # Update totals
    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    if total_regrouped > 0:
        print(f"    📊 Regroup summary: {iteration} iterations, {total_regrouped} segments regrouped")

    return g


def _split_oversized_bundles(g: gpd.GeoDataFrame,
                             nbrs: dict,
                             target_addrs: int,
                             split_threshold: float = 1.5,
                             seed: int = 42,
                             min_bundle_sfh: int | None = None) -> gpd.GeoDataFrame:
    """🔪 Split bundles > threshold with remainder checking."""
    bundle_totals = g.groupby("bundle_id")["sfh_addr_count"].sum()
    threshold = int(target_addrs * split_threshold)
    oversized = bundle_totals[bundle_totals > threshold]

    if oversized.empty:
        return g

    print(f"    ✂️  Found {len(oversized)} oversized bundles (> {threshold}), splitting...")

    next_bid = int(g["bundle_id"].max()) + 1
    rng = np.random.default_rng(seed)

    for bid, total_size in oversized.items():
        bundle_mask = g["bundle_id"] == bid
        bundle_indices = g.loc[bundle_mask].index.tolist()

        if len(bundle_indices) < 2:
            continue

        bundle_nbrs = {}
        for idx in bundle_indices:
            bundle_nbrs[idx] = set(n for n in nbrs.get(idx, []) if n in bundle_indices)

        remaining = set(bundle_indices)
        current_size = total_size

        while current_size > threshold and len(remaining) > 1:
            edge_segs = []
            for idx in remaining:
                same_bundle_neighbors = len([n for n in bundle_nbrs.get(idx, []) if n in remaining])
                edge_segs.append((idx, same_bundle_neighbors))

            edge_segs.sort(key=lambda x: x[1])

            chunk = set()
            chunk_size = 0
            visited = set()

            n_candidates = max(1, len(edge_segs) // 5)
            start_idx = edge_segs[rng.integers(0, n_candidates)][0]

            queue = deque([start_idx])
            visited.add(start_idx)

            while queue and chunk_size < target_addrs:
                # 🔒 Remainder check: Ensure remainder >= min_bundle_sfh
                if min_bundle_sfh is not None and chunk_size >= min_bundle_sfh:
                    # Check if remainder would also be >= min_bundle_sfh
                    remainder_size = current_size - chunk_size
                    if remainder_size >= min_bundle_sfh:
                        # If chunk is sufficient (>= 90% of target) and remainder is valid, stop
                        if chunk_size >= target_addrs * 0.9:
                            break

                seg_idx = queue.popleft()
                seg_addrs = int(g.at[seg_idx, "sfh_addr_count"])

                chunk.add(seg_idx)
                chunk_size += seg_addrs

                for neighbor in bundle_nbrs.get(seg_idx, []):
                    if neighbor in remaining and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if not chunk or len(chunk) >= len(remaining):
                break

            for idx in chunk:
                g.at[idx, "bundle_id"] = next_bid

            next_bid += 1
            remaining -= chunk
            current_size -= chunk_size

            if current_size <= threshold:
                break

    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    new_totals = g.groupby("bundle_id")["sfh_addr_count"].sum()
    still_oversized = new_totals[new_totals > threshold]
    if not still_oversized.empty:
        print(f"    ⚠️  {len(still_oversized)} bundles still oversized")
    else:
        print(f"    ✅ All bundles now <= {threshold}")

    return g


# ========== EULERIAN VALIDATION ==========

def _test_bundle_eulerizable(bundle_df: gpd.GeoDataFrame, snap_tol: float = 0.5) -> tuple[bool, int, int]:
    """
    Test if a bundle can be successfully eulerized for route planning.

    Returns:
        (is_eulerizable, odd_nodes_before, odd_nodes_after)
    """
    G = nx.MultiGraph()

    for idx, row in bundle_df.iterrows():
        a, b = _segment_endpoints(row.geometry, snap_tol)
        if a is None or b is None:
            continue
        a = _snap_xy(a, snap_tol)
        b = _snap_xy(b, snap_tol)
        w = float(row.geometry.length)
        G.add_edge(a, b, key=idx, weight=w, geom=row.geometry)

    if G.number_of_edges() == 0:
        return False, 0, 0

    # Check connectivity
    if not nx.is_connected(G):
        return False, 0, 0

    # Count odd-degree nodes before eulerization
    odd_nodes_before = [n for n, d in G.degree() if d % 2 == 1]
    num_odd_before = len(odd_nodes_before)

    # If already Eulerian or semi-Eulerian, it's good
    if num_odd_before in (0, 2):
        return True, num_odd_before, num_odd_before

    # Try eulerization
    try:
        # Build min-weight matching
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        K = nx.Graph()
        for i, u in enumerate(odd_nodes_before):
            K.add_node(u)
            for v in odd_nodes_before[i+1:]:
                K.add_edge(u, v, weight=dist[u][v])

        # Find matching
        try:
            match = nx.min_weight_matching(K, True, "weight")
        except TypeError:
            try:
                match = nx.min_weight_matching(K, maxcardinality=True, weight="weight")
            except Exception:
                # Greedy fallback
                match = set()
                remaining = set(odd_nodes_before)
                edges = sorted(K.edges(data=True), key=lambda e: e[2].get("weight", 0.0))
                for u, v, data in edges:
                    if u in remaining and v in remaining:
                        match.add((u, v))
                        remaining.remove(u)
                        remaining.remove(v)
                    if not remaining:
                        break

        # Add duplicate edges
        for u, v in match:
            path = nx.shortest_path(G, u, v, weight="weight")
            for s, t in zip(path[:-1], path[1:]):
                k, data = next(iter(G.get_edge_data(s, t).items()))
                G.add_edge(s, t, key=f"{k}_dup", weight=data['weight'], geom=data['geom'])

        # Check odd nodes after eulerization
        odd_nodes_after = [n for n, d in G.degree() if d % 2 == 1]
        num_odd_after = len(odd_nodes_after)

        # Success if 0 or 2 odd nodes
        return num_odd_after in (0, 2), num_odd_before, num_odd_after

    except Exception:
        return False, num_odd_before, num_odd_before


def _fix_non_eulerian_bundle(g: gpd.GeoDataFrame, bundle_id: int, nbrs: dict, snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """
    Try to fix a non-eulerian bundle by removing problematic segments.

    Strategy:
    1. Identify segments that create high-degree nodes (degree >= 3)
    2. Remove edge segments (segments with few neighbors in the bundle)
    3. Test if removal fixes the Eulerian property
    4. If not fixable, dissolve the entire bundle
    """
    bundle_mask = g['bundle_id'] == bundle_id
    bundle_indices = g.loc[bundle_mask].index.tolist()

    # Build graph to identify high-degree nodes
    G = nx.MultiGraph()
    seg_to_nodes = {}

    for idx in bundle_indices:
        row = g.loc[idx]
        a, b = _segment_endpoints(row.geometry, snap_tol)
        if a is None or b is None:
            continue
        a = _snap_xy(a, snap_tol)
        b = _snap_xy(b, snap_tol)
        G.add_edge(a, b, key=idx)
        seg_to_nodes[idx] = (a, b)

    # Find high-degree nodes (degree >= 3)
    high_degree_nodes = {n for n, d in G.degree() if d >= 3}

    if not high_degree_nodes:
        return g  # No obvious problem nodes

    # Find segments connected to high-degree nodes that are "edge" segments
    # (segments with only 1-2 neighbors in the bundle)
    candidates_to_remove = []
    for idx in bundle_indices:
        if idx not in seg_to_nodes:
            continue

        a, b = seg_to_nodes[idx]
        if a in high_degree_nodes or b in high_degree_nodes:
            # Check how many neighbors this segment has in the bundle
            bundle_neighbors = [n for n in nbrs.get(idx, []) if n in bundle_indices]
            if len(bundle_neighbors) <= 2:
                candidates_to_remove.append((idx, len(bundle_neighbors)))

    # Sort by number of neighbors (remove segments with fewer neighbors first)
    candidates_to_remove.sort(key=lambda x: x[1])

    # Try removing segments iteratively
    removed = []
    for seg_idx, _ in candidates_to_remove:
        # Tentatively remove this segment
        test_bundle = g.loc[bundle_mask & (g.index != seg_idx)]

        if len(test_bundle) == 0:
            break

        # Test if this fixes the bundle
        is_ok, _, _ = _test_bundle_eulerizable(test_bundle, snap_tol)

        if is_ok:
            # Success! Mark this segment as unassigned
            g.at[seg_idx, 'bundle_id'] = np.nan
            removed.append(seg_idx)
            return g
        else:
            # Keep trying with more removals
            removed.append(seg_idx)
            bundle_mask = bundle_mask & (g.index != seg_idx)

            # Don't remove too many segments
            if len(removed) >= max(3, len(bundle_indices) // 3):
                break

    # If we removed some segments, test the remaining bundle
    if removed:
        remaining_bundle = g.loc[g['bundle_id'] == bundle_id]
        if len(remaining_bundle) > 0:
            is_ok, _, _ = _test_bundle_eulerizable(remaining_bundle, snap_tol)
            if is_ok:
                return g

    # Could not fix - dissolve entire bundle
    g.loc[g['bundle_id'] == bundle_id, 'bundle_id'] = np.nan

    return g


def _regroup_non_eulerian_segments(g: gpd.GeoDataFrame, nbrs: dict,
                                    failed_bundles: list, target_addrs: int,
                                    min_size: int, max_size: int, seed: int,
                                    snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """
    Regroup segments from non-eulerian bundles.

    Strategy:
    1. Dissolve failed bundles (release their segments)
    2. Try to merge segments into neighboring eulerian bundles (if space permits)
    3. Regroup remaining segments using greedy BFS with eulerian validation
    """
    print(f"    🔄 Regrouping {len(failed_bundles)} non-eulerian bundles...")

    # Dissolve failed bundles
    segments_to_regroup = g[g['bundle_id'].isin(failed_bundles)].index.tolist()
    total_addrs = g.loc[segments_to_regroup, 'sfh_addr_count'].sum()

    print(f"       Released {len(segments_to_regroup)} segments ({int(total_addrs)} addresses)")

    g.loc[g['bundle_id'].isin(failed_bundles), 'bundle_id'] = np.nan

    # Step 1: Try to merge into neighboring eulerian bundles
    print(f"       Attempting to merge into neighboring bundles...")

    endpoint_idx = _build_bundle_endpoint_index(g[g['bundle_id'].notna()], snap_tol=snap_tol)
    bundle_sizes = g[g['bundle_id'].notna()].groupby('bundle_id')['sfh_addr_count'].sum().to_dict()

    merged_count = 0
    for seg_idx in list(segments_to_regroup):
        if g.at[seg_idx, 'bundle_id'] is not np.nan and not pd.isna(g.at[seg_idx, 'bundle_id']):
            continue  # Already assigned

        seg_addrs = int(g.at[seg_idx, 'sfh_addr_count'])

        # Find endpoints
        a, b = _segment_endpoints(g.at[seg_idx, 'geometry'], snap_tol)
        if not a or not b:
            continue
        a = _snap_xy(a, snap_tol)
        b = _snap_xy(b, snap_tol)

        # Find touching bundles
        touching_bundles = []
        for bid, endpoints in endpoint_idx.items():
            if a in endpoints or b in endpoints:
                # Check if adding won't exceed max size
                if bundle_sizes.get(bid, 0) + seg_addrs <= max_size:
                    touching_bundles.append(bid)

        if not touching_bundles:
            continue

        # Try each candidate and test if it remains eulerian
        for candidate_bid in sorted(touching_bundles, key=lambda bid: bundle_sizes.get(bid, 0)):
            # Tentatively add to this bundle
            test_bundle = g[(g['bundle_id'] == candidate_bid) | (g.index == seg_idx)].copy()
            test_bundle.at[seg_idx, 'bundle_id'] = candidate_bid

            # Test if still eulerian
            is_ok, _, _ = _test_bundle_eulerizable(test_bundle, snap_tol)

            if is_ok:
                # Accept this merge
                g.at[seg_idx, 'bundle_id'] = candidate_bid
                bundle_sizes[candidate_bid] = bundle_sizes.get(candidate_bid, 0) + seg_addrs
                endpoint_idx[candidate_bid].add(a)
                endpoint_idx[candidate_bid].add(b)
                segments_to_regroup.remove(seg_idx)
                merged_count += 1
                break

    print(f"       ✅ Merged {merged_count} segments into existing bundles")

    # Step 2: Regroup remaining segments with eulerian validation
    if segments_to_regroup:
        remaining_addrs = g.loc[segments_to_regroup, 'sfh_addr_count'].sum()
        print(f"       🔧 Regrouping {len(segments_to_regroup)} segments ({int(remaining_addrs)} addresses)...")

        rng = np.random.default_rng(seed + 999)
        remaining = set(segments_to_regroup)
        order = list(segments_to_regroup)
        rng.shuffle(order)

        next_bid = int(g['bundle_id'].max()) + 1 if g['bundle_id'].notna().any() else 1
        new_bundles_created = 0

        while remaining:
            # Select seed
            seed_idx = next((i for i in order if i in remaining), None)
            if seed_idx is None:
                break

            # BFS grow with eulerian validation
            q = deque([seed_idx])
            cur = []
            total = 0
            remaining.remove(seed_idx)

            while q and total < target_addrs:
                u = q.popleft()
                cur.append(u)
                total += int(g.loc[u, "sfh_addr_count"])

                # Get neighbors sorted by address count
                candidates = sorted(nbrs.get(u, set()) & remaining,
                                  key=lambda j: g.loc[j, "sfh_addr_count"],
                                  reverse=True)

                for v in candidates:
                    # Test if adding this segment keeps bundle eulerian
                    test_indices = cur + [v]
                    test_bundle = g.loc[test_indices].copy()
                    is_ok, _, _ = _test_bundle_eulerizable(test_bundle, snap_tol)

                    if is_ok or len(test_indices) == 1:  # Always accept first segment
                        remaining.remove(v)
                        q.append(v)
                    # else: skip this segment, it would break eulerian property

            # Only create bundle if within valid range AND eulerian
            if min_size <= total <= max_size and len(cur) > 0:
                final_bundle = g.loc[cur].copy()
                is_ok, _, _ = _test_bundle_eulerizable(final_bundle, snap_tol)

                if is_ok:
                    for i in cur:
                        g.at[i, 'bundle_id'] = next_bid
                    next_bid += 1
                    new_bundles_created += 1
                else:
                    # Bundle is not eulerian, leave segments unassigned
                    pass

        still_unassigned = g['bundle_id'].isna().sum()
        print(f"       ✅ Created {new_bundles_created} new eulerian bundles")
        print(f"       ⚠️  {still_unassigned} segments remain unassigned")

    return g


def _validate_and_regroup_eulerian(g: gpd.GeoDataFrame, nbrs: dict,
                                    target_addrs: int, min_size: int, max_size: int,
                                    seed: int, snap_tol: float = 0.5) -> gpd.GeoDataFrame:
    """
    Validate all bundles for Eulerian property and regroup problematic ones.

    This ensures ALL bundles can be walked (have Eulerian paths).
    """
    print(f"    🔍 Testing {g['bundle_id'].nunique()} bundles for Eulerian property...")

    failed_bundles = []
    bundle_stats = []

    for bundle_id, bundle_df in g.groupby('bundle_id'):
        if pd.isna(bundle_id):
            continue

        is_ok, odd_before, odd_after = _test_bundle_eulerizable(bundle_df, snap_tol)

        bundle_stats.append({
            'bundle_id': bundle_id,
            'n_segments': len(bundle_df),
            'total_addrs': bundle_df['sfh_addr_count'].sum(),
            'odd_before': odd_before,
            'odd_after': odd_after,
            'is_eulerizable': is_ok
        })

        if not is_ok:
            failed_bundles.append(bundle_id)

    if failed_bundles:
        print(f"    ⚠️  Found {len(failed_bundles)} non-eulerizable bundles")

        for bid in failed_bundles:
            stats = next(s for s in bundle_stats if s['bundle_id'] == bid)
            print(f"       Bundle {bid}: {stats['n_segments']} segs, {int(stats['total_addrs'])} addrs, "
                  f"{stats['odd_before']} → {stats['odd_after']} odd nodes")

        # Regroup these bundles
        g = _regroup_non_eulerian_segments(g, nbrs, failed_bundles, target_addrs,
                                           min_size, max_size, seed, snap_tol)

        # Verify all remaining bundles are eulerian
        print(f"    🔍 Verifying all bundles are now eulerian...")
        final_check_failed = []
        for bundle_id, bundle_df in g.groupby('bundle_id'):
            if pd.isna(bundle_id):
                continue
            is_ok, _, _ = _test_bundle_eulerizable(bundle_df, snap_tol)
            if not is_ok:
                final_check_failed.append(bundle_id)

        if final_check_failed:
            print(f"    ⚠️  Warning: {len(final_check_failed)} bundles still non-eulerian, dissolving...")
            g.loc[g['bundle_id'].isin(final_check_failed), 'bundle_id'] = np.nan
        else:
            print(f"    ✅ All bundles are now eulerizable!")
    else:
        print(f"    ✅ All bundles are eulerizable!")

    # Update totals
    totals = g.groupby("bundle_id")["sfh_addr_count"].agg(["sum", "count"])
    totals = totals.rename(columns={"sum": "bundle_addr_total", "count": "bundle_seg_count"})
    g = g.drop(columns=[c for c in ["bundle_addr_total", "bundle_seg_count"] if c in g.columns], errors="ignore")
    g = g.merge(totals, left_on="bundle_id", right_index=True, how="left")

    return g


# ========== TWO INDEPENDENT WORKFLOWS ==========

def _build_connected_bundles_greedy(segs_m: gpd.GeoDataFrame, seg_id_col: str,
                                    target_addrs: int, join_tol_m: float,
                                    seed: int, min_bundle_sfh: int | None):
    """
    ❌ GREEDY (5 steps, NO CONSTRAINTS)

    标准 bundle.py 流程：
    Step 1: Build graph + components
    Step 2: Grow bundles (greedy BFS)
    Step 3: Merge tiny (distance, no constraint)
    Step 4: Sweep residuals (soft 1.1x)
    Step 5: Enforce contiguity
    （结束 - 无第二次 merge）
    """
    print("\n" + "="*70)
    print("❌ GREEDY (5 steps, NO CONSTRAINTS)")
    print("="*70)

    print(f"\n>>> [Step 1/5] Build graph + components...")
    nbrs, keep_idx, _ = _build_segment_neighbor_graph(segs_m, join_tol_m)
    g = segs_m.iloc[keep_idx].copy()
    comp_of = _components_from_neighbors(nbrs)
    g["component_id"] = g.index.map(comp_of.get)

    print(f">>> [Step 2/5] Grow bundles (greedy BFS)...")
    rng = np.random.default_rng(int(seed))
    bundle_id_global = 1
    out_ids = {}

    for comp in sorted(set(comp_of.values())):
        comp_indices = [i for i, c in comp_of.items() if c == comp]
        local_map, next_local = _grow_bundles_in_component(g, nbrs, comp_indices, target_addrs, rng)
        for i, local_bid in local_map.items():
            out_ids[i] = bundle_id_global + (local_bid - 1)
        bundle_id_global += (next_local - 1)

    g["bundle_id"] = g.index.map(out_ids.get)

    print(f">>> [Step 3/5] Merge tiny (distance, ❌ no constraint)...")
    g = _merge_tiny_bundles_no_constraint(g, min_bundle_sfh=min_bundle_sfh)

    print(f">>> [Step 4/5] Sweep residuals (soft 1.1x)...")
    g = _sweep_attach_residuals(g, soft_max_bundle_sfh=target_addrs)

    print(f">>> [Step 5/5] Enforce contiguity...")
    g = _enforce_endpoint_contiguity(g, snap_tol=0.5)

    # 直接返回！标准 bundle.py 在 enforce 之后就结束了
    cols = [seg_id_col, "sfh_addr_count", "bundle_id", "bundle_addr_total",
            "bundle_seg_count", "component_id", "geometry"]
    cols = [c for c in cols if c in g.columns] + [c for c in g.columns if c not in cols]
    return g[cols]


def _build_connected_bundles_multibfs(segs_m: gpd.GeoDataFrame, seg_id_col: str,
                                      target_addrs: int, join_tol_m: float,
                                      seed: int, min_bundle_sfh: int | None,
                                      hard_max_multiplier: float):
    """
    🔒 MULTI-BFS (10 steps, BALANCED + SPLIT + REGROUP + EULERIAN-AWARE)

    Step 1: Build graph + components
    Step 2: Grow bundles (multi-BFS balanced)
    Step 3: Merge tiny (connected, 🔒 hard 1.1x, prefer smallest)
    Step 4: Sweep residuals (soft 1.1x)
    Step 5: Split oversized (> 1.0x = 80)
    Step 6: Merge tiny (connected, 🔒 hard 1.1x)
    Step 7: Enforce contiguity
    Step 8: Final cleanup (connected, 🔒 hard 1.1x)
    Step 9: Regroup invalid bundles (循环重组 [0.9x, 1.1x])
    Step 10: Validate & regroup for Eulerian property (智能重组不可行走的 bundles)

    Key Feature: Step 10 ensures ALL bundles can be walked:
    - Detects bundles with non-Eulerian topology
    - Dissolves problematic bundles
    - Merges segments into neighboring eulerian bundles (if space permits)
    - Regroups remaining segments with real-time eulerian validation
    - Result: Only walkable bundles, no segments wasted
    """
    print("\n" + "="*70)
    print(f"🔒 MULTI-BFS (BALANCED + SPLIT + REGROUP + EULERIAN {hard_max_multiplier}x)")
    print("="*70)

    print(f"\n>>> [Step 1/10] Build graph + components...")
    nbrs, keep_idx, _ = _build_segment_neighbor_graph(segs_m, join_tol_m)
    g = segs_m.iloc[keep_idx].copy()
    comp_of = _components_from_neighbors(nbrs)
    g["component_id"] = g.index.map(comp_of.get)

    print(f">>> [Step 2/10] Grow bundles (multi-BFS balanced)...")
    bundle_id_global = 1
    out_ids = {}

    for comp in sorted(set(comp_of.values())):
        comp_indices = [i for i, c in comp_of.items() if c == comp]
        local_map = _multi_source_balanced_bfs(g, nbrs, comp_indices, target_addrs, seed)
        next_local = len(set(local_map.values())) + 1
        for i, local_bid in local_map.items():
            out_ids[i] = bundle_id_global + (local_bid - 1)
        bundle_id_global += (next_local - 1)

    g["bundle_id"] = g.index.map(out_ids.get)

    print(f">>> [Step 3/10] Merge tiny (connected, 🔒 hard {hard_max_multiplier}x, prefer smallest)...")
    g = _merge_tiny_bundles_connected(g, min_bundle_sfh=min_bundle_sfh,
                                       target_addrs=target_addrs,
                                       hard_max_multiplier=hard_max_multiplier,  # 🔒 启用硬约束
                                       snap_tol=0.5)

    print(f">>> [Step 4/10] Sweep residuals (soft 1.1x)...")
    g = _sweep_attach_residuals(g, soft_max_bundle_sfh=target_addrs)

    print(f">>> [Step 5/10] Split oversized (> 1.0x = {target_addrs})...")
    g = _split_oversized_bundles(g, nbrs=nbrs, target_addrs=target_addrs,
                                 split_threshold=1.0, seed=seed, min_bundle_sfh=min_bundle_sfh)

    print(f">>> [Step 6/10] Merge tiny (connected, 🔒 hard {hard_max_multiplier}x)...")
    g = _merge_tiny_bundles_connected(g, min_bundle_sfh=min_bundle_sfh,
                                       target_addrs=target_addrs,
                                       hard_max_multiplier=hard_max_multiplier,
                                       snap_tol=0.5)

    print(f">>> [Step 7/10] Enforce contiguity...")
    g = _enforce_endpoint_contiguity(g, snap_tol=0.5)

    print(f">>> [Step 8/10] Final cleanup (connected, 🔒 hard {hard_max_multiplier}x)...")
    g = _merge_tiny_bundles_connected(g, min_bundle_sfh=min_bundle_sfh,
                                       target_addrs=target_addrs,
                                       hard_max_multiplier=hard_max_multiplier,
                                       snap_tol=0.5)

    print(f">>> [Step 9/10] Regroup invalid bundles ([0.9x, 1.1x])...")
    min_size = int(target_addrs * 0.9)  # 72
    max_size = int(target_addrs * hard_max_multiplier)  # 88
    g = _regroup_invalid_bundles(g, nbrs=nbrs, target_addrs=target_addrs,
                                 min_size=min_size, max_size=max_size,
                                 seed=seed, max_iterations=5)

    print(f">>> [Step 10/10] Validate & regroup for Eulerian property...")
    g = _validate_and_regroup_eulerian(g, nbrs=nbrs, target_addrs=target_addrs,
                                        min_size=min_size, max_size=max_size,
                                        seed=seed, snap_tol=0.5)

    cols = [seg_id_col, "sfh_addr_count", "bundle_id", "bundle_addr_total",
            "bundle_seg_count", "component_id", "geometry"]
    cols = [c for c in cols if c in g.columns] + [c for c in g.columns if c not in cols]
    return g[cols]


# ========== MAIN ENTRY POINT ==========

def _build_connected_bundles(segs_m: gpd.GeoDataFrame, seg_id_col: str,
                             target_addrs: int, join_tol_m: float = 15.0,
                             seed: int = 42, min_bundle_sfh: int | None = None,
                             method: str = "greedy", hard_max_multiplier: float | None = 1.1):
    """
    Build bundles with two completely independent workflows.

    Parameters
    ----------
    method : str
        "greedy" = 5 steps, no constraints (standard bundle.py)
        "multi_bfs" = 10 steps, hard constraints + regroup (new version)
    hard_max_multiplier : float
        Only used for multi_bfs (default 1.1)
    """
    if method == "greedy":
        return _build_connected_bundles_greedy(
            segs_m, seg_id_col, target_addrs, join_tol_m, seed, min_bundle_sfh
        )
    elif method == "multi_bfs":
        return _build_connected_bundles_multibfs(
            segs_m, seg_id_col, target_addrs, join_tol_m, seed, min_bundle_sfh,
            hard_max_multiplier
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'greedy' or 'multi_bfs'")


# ---------- CLI entry ----------

def run_bundle(session: str,
               target_addrs: int,
               join_tol_m: float = 15.0,
               seed: int = 42,
               tag: str | None = None,
               min_bundle_sfh: int | None = None,
               method: str = "greedy",
               hard_max_multiplier: float | None = 1.1):
    """
    Create bundles for a session (DH or D2DS).

    Parameters
    ----------
    method : str
        "greedy" = no constraints (5 steps, standard bundle.py)
        "multi_bfs" = hard constraints + regroup (10 steps, new version)
    hard_max_multiplier : float
        Only for multi_bfs (default 1.1)
    """
    streets, addrs, ds, pr = load_sources()
    seg_id = pr["fields"]["streets_segment_id"]

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

    latest = cands[-1]
    segs = gpd.read_parquet(latest)
    print(f"[bundle] using eligible parquet: {latest}")

    if seg_id not in segs.columns:
        for c in ("segment_id", "SEGMENT_ID"):
            if c in segs.columns:
                segs = segs.rename(columns={c: seg_id})
                break
        else:
            raise SystemExit(f"Eligible parquet {latest} lacks '{seg_id}' column.")

    need_cols = [seg_id, "sfh_addr_count", "geometry"]
    missing = [c for c in need_cols if c not in segs.columns]
    if missing:
        raise SystemExit(f"Eligible parquet missing columns: {missing}")

    segs = segs[need_cols].copy()

    work_epsg = int(pr.get("crs", {}).get("working_meters", 26911))
    segs_m = project_to(segs, work_epsg)

    bundled = _build_connected_bundles(
        segs_m,
        seg_id_col=seg_id,
        target_addrs=target_addrs,
        join_tol_m=join_tol_m,
        seed=seed,
        min_bundle_sfh=min_bundle_sfh,
        method=method,
        hard_max_multiplier=hard_max_multiplier
    )

    out_dir = root / "outputs" / "bundles" / session.upper()
    ensure_dir(out_dir)
    bundled.to_parquet(out_dir / "bundles.parquet", index=False)

    try:
        import folium
        from sd311_fieldprep.utils import folium_map

        viz = bundled.dropna(subset=["bundle_id"]).to_crs(4326)
        tooltip_cols = [seg_id, "bundle_id", "bundle_seg_count", "bundle_addr_total"]

        m = folium_map(viz, color_col="bundle_id", tooltip_cols=tooltip_cols)
        m.save(str(out_dir / "bundles_map.html"))
    except Exception as e:
        print("[bundle] map skipped:", e)

    n_bundles = int(bundled["bundle_id"].nunique())
    seg_counts = bundled.groupby("bundle_id")["bundle_seg_count"].first()
    if seg_counts.isna().any():
        seg_counts = bundled.groupby("bundle_id")["sfh_addr_count"].count()
    singleton_share = float((seg_counts == 1).mean())

    print(f"[bundle] wrote {out_dir} with {n_bundles} bundles (singleton share ≈ {singleton_share:.3f}).")
