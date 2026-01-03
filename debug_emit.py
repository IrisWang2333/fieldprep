#!/usr/bin/env python
"""Debug why emit fails for specific bundles"""
import sys
import pandas as pd
import geopandas as gpd
import networkx as nx
from pathlib import Path

# Add src to path
SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.route import _line_endpoints_xy, _snap_xy, _pair_odds_min_weight

def analyze_bundle(bundle_df):
    """Analyze why a bundle fails Eulerian test"""
    g = bundle_df.to_crs(26911) if bundle_df.crs else bundle_df
    SNAP_TOL = 0.5

    # Determine segment_id column
    seg_id_col = 'segment_id' if 'segment_id' in g.columns else 'iamfloc'

    # Build graph
    G = nx.MultiGraph()
    for idx, row in g.iterrows():
        a, b = _line_endpoints_xy(row.geometry)
        if a is None or b is None:
            continue
        a = _snap_xy(a, SNAP_TOL)
        b = _snap_xy(b, SNAP_TOL)
        w = float(row.geometry.length)
        seg_id = row[seg_id_col]
        G.add_edge(a, b, key=seg_id,
                   segment_id=seg_id, weight=w, geom=row.geometry)

    print(f"\n{'='*70}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Connected components: {nx.number_connected_components(G)}")

    # Check odd-degree nodes BEFORE eulerization
    odd_nodes_before = [n for n, d in G.degree() if d % 2 == 1]
    print(f"\nOdd-degree nodes BEFORE eulerization: {len(odd_nodes_before)}")

    if len(odd_nodes_before) not in (0, 2):
        print(f"  → Needs eulerization")

        # Try eulerization
        try:
            pairs = _pair_odds_min_weight(G)
            print(f"  → Min-weight matching found {len(pairs)} pairs")

            edges_added = 0
            for u, v in pairs:
                try:
                    path = nx.shortest_path(G, u, v, weight="weight")
                    print(f"    Pair {u} ↔ {v}: path length = {len(path)-1} edges")
                    for s, t in zip(path[:-1], path[1:]):
                        k, data = next(iter(G.get_edge_data(s, t).items()))
                        G.add_edge(s, t, key=f"{data['segment_id']}_dup",
                                   segment_id=data['segment_id'],
                                   weight=data['weight'], geom=data['geom'])
                        edges_added += 1
                except nx.NetworkXNoPath:
                    print(f"    ⚠️  No path between {u} and {v}!")

            print(f"  → Added {edges_added} duplicate edges")

        except Exception as e:
            print(f"  ⚠️  Eulerization failed: {e}")
            return False

    # Check odd-degree nodes AFTER eulerization
    odd_nodes_after = [n for n, d in G.degree() if d % 2 == 1]
    print(f"\nOdd-degree nodes AFTER eulerization: {len(odd_nodes_after)}")

    if len(odd_nodes_after) == 0:
        print("  ✅ Graph is now Eulerian (all nodes have even degree)")
        return True
    elif len(odd_nodes_after) == 2:
        print("  ✅ Graph is semi-Eulerian (can use Eulerian path)")
        return True
    else:
        print(f"  ❌ Graph still has {len(odd_nodes_after)} odd-degree nodes!")
        print(f"     Degrees: {sorted([d for n, d in G.degree() if d % 2 == 1])}")
        return False


if __name__ == "__main__":
    # Load plan
    plan = pd.read_csv("outputs/plans/bundles_plan_2025-12-26.csv")
    print(f"Plan has {len(plan)} assignments")

    # Load bundle file
    bundles = gpd.read_parquet("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
    print(f"Bundle file has {len(bundles)} segments, {bundles['bundle_id'].nunique()} bundles")

    # Check each bundle in plan
    for idx, row in plan.iterrows():
        bundle_id = row['bundle_id']
        interviewer = row['interviewer']

        print(f"\n{'#'*70}")
        print(f"Bundle {bundle_id} (Interviewer {interviewer})")
        print(f"{'#'*70}")

        bundle_segs = bundles[bundles['bundle_id'] == bundle_id].copy()
        print(f"Segments: {len(bundle_segs)}, Addresses: {bundle_segs['sfh_addr_count'].sum()}")

        result = analyze_bundle(bundle_segs)

        if not result:
            print(f"\n❌ Bundle {bundle_id} FAILED - cannot create Eulerian path")
        else:
            print(f"\n✅ Bundle {bundle_id} OK")
