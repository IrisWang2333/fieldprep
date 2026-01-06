import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


def _line_endpoints_xy(ls):
    """Extract endpoints from LineString or MultiLineString geometry."""
    try:
        # Try direct access for simple LineString
        coords = list(ls.coords)
    except Exception:
        # Handle MultiLineString or other complex geometries
        m = linemerge(ls)

        # Check if still MultiLineString FIRST (before accessing coords)
        if isinstance(m, MultiLineString):
            if len(m.geoms) > 0:
                first_line = m.geoms[0]
                last_line = m.geoms[-1]
                first_coords = list(first_line.coords)
                last_coords = list(last_line.coords)
                if len(first_coords) >= 1 and len(last_coords) >= 1:
                    # Return first point of first line and last point of last line
                    return (first_coords[0][0], first_coords[0][1]), (last_coords[-1][0], last_coords[-1][1])
            coords = []
        # If linemerge succeeded and returned a simple LineString
        elif isinstance(m, LineString):
            try:
                coords = list(m.coords)
            except Exception:
                coords = []
        else:
            coords = []

    if len(coords) < 2:
        return None, None
    return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])

def _snap_xy(pt, tol=0.5):
    """Snap a (x,y) tuple to a grid of size `tol` meters to fuse near-identical endpoints."""
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

def _pair_odds_min_weight(G):
    """
    Return pairs for odd-degree nodes using min-weight matching.
    Works across different NetworkX versions; falls back to greedy if needed.
    """
    odd = [n for n, d in G.degree() if d % 2 == 1]
    if len(odd) <= 1:
        return []
    # Build complete aux graph on odd nodes with shortest-path distances as weights
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    K = nx.Graph()
    for i, u in enumerate(odd):
        K.add_node(u)
        for v in odd[i+1:]:
            K.add_edge(u, v, weight=dist[u][v])
    # Try different call signatures
    try:
        match = nx.min_weight_matching(K, True, "weight")
    except TypeError:
        try:
            match = nx.min_weight_matching(K, maxcardinality=True, weight="weight")
        except Exception:
            # Greedy fallback: pick shortest edges until all paired
            match = set()
            remaining = set(odd)
            edges = sorted(K.edges(data=True), key=lambda e: e[2].get("weight", 0.0))
            for u, v, data in edges:
                if u in remaining and v in remaining:
                    match.add((u, v))
                    remaining.remove(u); remaining.remove(v)
                if not remaining:
                    break
    return list(match)

def build_walk_order(b_segs: gpd.GeoDataFrame):
    """
    Given a CONNECTED set of segments (one bundle) in meters CRS,
    return an ordered list of (segment_id, oriented LineString) forming a continuous walk.
    Uses CPP-lite: Eulerize odd nodes if needed, then compute Eulerian path/circuit.
    """
    g = b_segs.copy()
    if g.crs is None or g.crs.to_epsg() != 26911:
        g = g.to_crs(26911)

    SNAP_TOL = 0.5  # meters

    G = nx.MultiGraph()
    for row in g.itertuples():
        a, b = _line_endpoints_xy(row.geometry)
        if a is None or b is None:
            continue
        # snap endpoints to fuse tiny coordinate discrepancies
        a = _snap_xy(a, SNAP_TOL)
        b = _snap_xy(b, SNAP_TOL)
        w = float(row.geometry.length)
        G.add_edge(a, b, key=row.segment_id,
                   segment_id=row.segment_id, weight=w, geom=row.geometry)

    # --- CONNECTIVITY CHECK (run once, after building the graph) ---
    if G.number_of_edges() == 0:
        raise SystemExit("Bundle graph has no edges after filtering; cannot route.")
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        sizes = [len(c) for c in nx.connected_components(G)]
        raise SystemExit(
            f"Bundle graph is not connected: {n_comp} components (node-counts={sizes}). "
            "Please rebundle or choose a different bundle; route builder needs a single connected bundle."
        )


    # Eulerize if needed
    odd_nodes = [n for n, d in G.degree() if d % 2 == 1]
    if len(odd_nodes) not in (0, 2):
        for u, v in _pair_odds_min_weight(G):
            path = nx.shortest_path(G, u, v, weight="weight")
            for s, t in zip(path[:-1], path[1:]):
                k, data = next(iter(G.get_edge_data(s, t).items()))
                G.add_edge(s, t, key=f"{data['segment_id']}_dup",
                           segment_id=data['segment_id'],
                           weight=data['weight'], geom=data['geom'])

    # Get Eulerian trail/circuit; support 2-tuple or 3-tuple edges
    try:
        trail = list(nx.eulerian_path(G, keys=True)) if len([n for n,d in G.degree() if d%2==1])==2 else list(nx.eulerian_circuit(G, keys=True))
    except TypeError:
        trail = list(nx.eulerian_path(G)) if len([n for n,d in G.degree() if d%2==1])==2 else list(nx.eulerian_circuit(G))

    ordered = []
    cur = None
    for edge in trail:
        if len(edge) == 3:
            u, v, k = edge
            data = G.get_edge_data(u, v, k)
        else:  # 2-tuple
            u, v = edge
            k, data = next(iter(G.get_edge_data(u, v).items()))
        seg_id, geom = data["segment_id"], data["geom"]
        a, b = _line_endpoints_xy(geom)
        # orient so the walk is continuous
        if cur is None or (abs(cur[0]-a[0]) + abs(cur[1]-a[1])) <= (abs(cur[0]-b[0]) + abs(cur[1]-b[1])):
            oriented = geom
            cur = b
        else:
            oriented = LineString(list(geom.coords)[::-1])
            cur = a
        ordered.append((seg_id, oriented))
    return ordered
