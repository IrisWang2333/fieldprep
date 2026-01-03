#!/usr/bin/env python
"""Detailed analysis of Bundle 3480"""
import sys
import pandas as pd
import geopandas as gpd
import networkx as nx
from pathlib import Path
from collections import Counter

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sd311_fieldprep.route import _line_endpoints_xy, _snap_xy

# Load bundle
bundles = gpd.read_parquet("outputs/bundles/DH/bundles_multibfs_regroup_filtered.parquet")
bundle_segs = bundles[bundles['bundle_id'] == 3480].copy()
g = bundle_segs.to_crs(26911)

SNAP_TOL = 0.5

# Build graph
G = nx.MultiGraph()
for idx, row in g.iterrows():
    a, b = _line_endpoints_xy(row.geometry)
    if a is None or b is None:
        continue
    a = _snap_xy(a, SNAP_TOL)
    b = _snap_xy(b, SNAP_TOL)
    w = float(row.geometry.length)
    seg_id = row['iamfloc']
    G.add_edge(a, b, key=seg_id, segment_id=seg_id, weight=w, geom=row.geometry)

print(f"Bundle 3480 Graph:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Connected: {nx.is_connected(G)}")

# Show degree of each node
print(f"\nNode degrees BEFORE eulerization:")
degrees_before = dict(G.degree())
for node, deg in sorted(degrees_before.items(), key=lambda x: x[1], reverse=True):
    marker = "ODD" if deg % 2 == 1 else "even"
    print(f"  {node}: {deg} ({marker})")

odd_nodes = [n for n, d in G.degree() if d % 2 == 1]
print(f"\nOdd nodes: {len(odd_nodes)}")

# Manual eulerization with tracking
if len(odd_nodes) not in (0, 2):
    print(f"\n{'='*70}")
    print("EULERIZATION PROCESS:")
    print(f"{'='*70}")

    # Build min-weight matching
    from sd311_fieldprep.route import _pair_odds_min_weight
    pairs = _pair_odds_min_weight(G)

    print(f"\nFound {len(pairs)} pairs:")
    for i, (u, v) in enumerate(pairs, 1):
        path = nx.shortest_path(G, u, v, weight="weight")
        path_dist = sum(G[path[j]][path[j+1]][list(G[path[j]][path[j+1]].keys())[0]]['weight']
                       for j in range(len(path)-1))
        print(f"\n{i}. {u} → {v}")
        print(f"   Path: {' → '.join(str(p) for p in path)}")
        print(f"   Path length: {len(path)-1} edges, distance: {path_dist:.1f}m")

        # Check which nodes on path are odd
        path_odd = [p for p in path if p in odd_nodes]
        if len(path_odd) > 2:
            print(f"   ⚠️  Path goes through {len(path_odd)-2} OTHER odd nodes: {path_odd}")

        # Add duplicate edges
        for s, t in zip(path[:-1], path[1:]):
            k, data = next(iter(G.get_edge_data(s, t).items()))
            G.add_edge(s, t, key=f"{data['segment_id']}_dup",
                       segment_id=data['segment_id'],
                       weight=data['weight'], geom=data['geom'])
            print(f"     + Duplicate edge: {s} ↔ {t}")

# Check degrees after
print(f"\n{'='*70}")
print("Node degrees AFTER eulerization:")
degrees_after = dict(G.degree())
for node in sorted(degrees_before.keys()):
    deg_before = degrees_before[node]
    deg_after = degrees_after[node]
    change = deg_after - deg_before
    marker = "ODD" if deg_after % 2 == 1 else "even"
    print(f"  {node}: {deg_before} → {deg_after} (+{change}) ({marker})")

odd_after = [n for n, d in G.degree() if d % 2 == 1]
print(f"\nOdd nodes after: {len(odd_after)}")

if len(odd_after) > 2:
    print(f"\n❌ FAILED: Still have {len(odd_after)} odd nodes")
    print(f"This means the matching/path selection created a bad configuration.")
else:
    print(f"\n✅ SUCCESS: Graph is now Eulerian")
