"""
Parse SUMO .net.xml to extract real TLS topology for GAT adjacency.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import unquote
import math
import os
import torch


def extract_net_file_from_sumocfg(sumocfg_path):
    """Extract the .net.xml file path from a .sumocfg file."""
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    # Try both with and without namespace
    net_elem = root.find('.//net-file')
    if net_elem is None:
        for elem in root.iter():
            if elem.tag.endswith('net-file'):
                net_elem = elem
                break
    if net_elem is None:
        raise FileNotFoundError(f"No <net-file> found in {sumocfg_path}")
    net_file = unquote(net_elem.get('value', ''))
    if not os.path.isabs(net_file):
        net_file = os.path.join(os.path.dirname(sumocfg_path), net_file)
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Network file not found: {net_file}")
    return net_file


def parse_sumo_network(net_file_path):
    """Parse .net.xml and return TLS topology info."""
    print(f"[TOPOLOGY] Parsing {os.path.basename(net_file_path)} ...")
    tree = ET.parse(net_file_path)
    root = tree.getroot()

    # Edge -> junction mapping
    edge_from = {}
    edge_to = {}
    for edge in root.findall('edge'):
        eid = edge.get('id', '')
        if eid.startswith(':'):
            continue
        edge_from[eid] = edge.get('from', '')
        edge_to[eid] = edge.get('to', '')

    # Junction positions
    junction_pos = {}
    for junc in root.findall('junction'):
        jid = junc.get('id', '')
        junction_pos[jid] = (
            float(junc.get('x', '0')),
            float(junc.get('y', '0')),
        )

    # Map TLS program IDs to junctions via <connection tl="...">
    tls_to_junctions = defaultdict(set)
    tls_edges = defaultdict(lambda: {'from': set(), 'to': set()})
    for conn in root.findall('connection'):
        tl = conn.get('tl', '')
        from_edge = conn.get('from', '')
        to_edge = conn.get('to', '')
        if not tl:
            continue
        if from_edge in edge_to:
            tls_to_junctions[tl].add(edge_to[from_edge])
        if to_edge in edge_from:
            tls_to_junctions[tl].add(edge_from[to_edge])
        tls_edges[tl]['from'].add(from_edge)
        tls_edges[tl]['to'].add(to_edge)

    # Junction connectivity graph (non-internal edges)
    junc_graph = defaultdict(set)
    for eid in edge_from:
        ef = edge_from[eid]
        et = edge_to.get(eid, '')
        if ef and et:
            junc_graph[ef].add(et)
            junc_graph[et].add(ef)

    print(f"[TOPOLOGY] {len(edge_from)} edges, {len(junction_pos)} junctions, "
          f"{len(tls_to_junctions)} TLS programs")

    return {
        'edge_from': edge_from,
        'edge_to': edge_to,
        'junction_pos': junction_pos,
        'tls_to_junctions': dict(tls_to_junctions),
        'tls_edges': {k: dict(v) for k, v in tls_edges.items()},
        'junc_graph': dict(junc_graph),
    }


def build_tls_adjacency(tls_ids, net_info,
                         distance_threshold=800,
                         max_intermediate_hops=6,
                         min_neighbors=2):
    """
    Build TLS adjacency from the real SUMO network.

    Returns:
        neighbor_dict: {tls_id: [neighbor_ids]}
        distance_dict: {(tls_a, tls_b): distance_in_meters}
        tls_center:    {tls_id: (x, y)}
    """
    tls_to_junctions = net_info['tls_to_junctions']
    tls_edges = net_info['tls_edges']
    junction_pos = net_info['junction_pos']
    junc_graph = net_info['junc_graph']
    edge_from = net_info['edge_from']
    edge_to = net_info['edge_to']

    # Center position for each TLS
    tls_center = {}
    for tl_id in tls_ids:
        juncs = tls_to_junctions.get(tl_id, set())
        positions = [junction_pos[j] for j in juncs if j in junction_pos]
        if positions:
            tls_center[tl_id] = (
                sum(p[0] for p in positions) / len(positions),
                sum(p[1] for p in positions) / len(positions),
            )
        else:
            tls_center[tl_id] = (0.0, 0.0)

    def _dist(a, b):
        dx = tls_center[a][0] - tls_center[b][0]
        dy = tls_center[a][1] - tls_center[b][1]
        return math.sqrt(dx * dx + dy * dy)

    neighbor_dict = defaultdict(set)

    # ── Method A: Direct edge connection ──
    for tl_a in tls_ids:
        edges_a = tls_edges.get(tl_a, {'from': set(), 'to': set()})
        for tl_b in tls_ids:
            if tl_a == tl_b:
                continue
            juncs_b = tls_to_junctions.get(tl_b, set())
            for to_edge in edges_a.get('to', set()):
                if to_edge in edge_to and edge_to[to_edge] in juncs_b:
                    neighbor_dict[tl_a].add(tl_b)
                    neighbor_dict[tl_b].add(tl_a)
                    break
            for from_edge in edges_a.get('from', set()):
                if from_edge in edge_from and edge_from[from_edge] in juncs_b:
                    neighbor_dict[tl_a].add(tl_b)
                    neighbor_dict[tl_b].add(tl_a)
                    break

    # ── Method B: BFS through non-TLS junctions ──
    all_tls_juncs = set()
    for juncs in tls_to_junctions.values():
        all_tls_juncs.update(juncs)

    for tl_a in tls_ids:
        start_juncs = tls_to_junctions.get(tl_a, set())
        visited = set(start_juncs)
        queue = [(j, 0) for j in start_juncs]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_intermediate_hops:
                continue
            for nbr_junc in junc_graph.get(current, set()):
                if nbr_junc in visited:
                    continue
                visited.add(nbr_junc)
                for tl_b in tls_ids:
                    if tl_b == tl_a:
                        continue
                    if nbr_junc in tls_to_junctions.get(tl_b, set()):
                        neighbor_dict[tl_a].add(tl_b)
                        neighbor_dict[tl_b].add(tl_a)
                is_tls = any(
                    nbr_junc in tls_to_junctions.get(t, set())
                    for t in tls_ids if t != tl_a
                )
                if not is_tls:
                    queue.append((nbr_junc, depth + 1))

    # ── Method C: Geographic proximity ──
    # Connect TLS nodes that are within distance_threshold of each other
    for i, tl_a in enumerate(tls_ids):
        for j, tl_b in enumerate(tls_ids):
            if i >= j:
                continue
            if tl_b in neighbor_dict[tl_a]:
                continue  # already connected
            d = _dist(tl_a, tl_b)
            if d <= distance_threshold:
                neighbor_dict[tl_a].add(tl_b)
                neighbor_dict[tl_b].add(tl_a)

    # ── Method D: Ensure minimum neighbors for leaf nodes ──
    # Leaf nodes (<=1 neighbor) get connected to their nearest unconnected TLS
    for tl_a in tls_ids:
        if len(neighbor_dict[tl_a]) >= min_neighbors:
            continue
        # Sort all other TLS by distance
        candidates = []
        for tl_b in tls_ids:
            if tl_b == tl_a or tl_b in neighbor_dict[tl_a]:
                continue
            candidates.append((tl_b, _dist(tl_a, tl_b)))
        candidates.sort(key=lambda x: x[1])
        # Add nearest neighbors until we reach min_neighbors
        for tl_b, d in candidates:
            if len(neighbor_dict[tl_a]) >= min_neighbors:
                break
            neighbor_dict[tl_a].add(tl_b)
            neighbor_dict[tl_b].add(tl_a)

    # ── Method E: Connect disconnected components ──
    def get_components():
        visited = set()
        components = []
        for tl in tls_ids:
            if tl not in visited:
                comp = set()
                queue = [tl]
                while queue:
                    curr = queue.pop(0)
                    if curr not in comp:
                        comp.add(curr)
                        visited.add(curr)
                        for nbr in neighbor_dict.get(curr, set()):
                            queue.append(nbr)
                components.append(comp)
        return components

    components = get_components()
    # Connect closest components until only 1 remains or distance is too huge
    while len(components) > 1:
        min_dist = float('inf')
        best_pair = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for n1 in components[i]:
                    for n2 in components[j]:
                        d = _dist(n1, n2)
                        if d < min_dist:
                            min_dist = d
                            best_pair = (n1, n2, i, j)
        
        if best_pair and min_dist < 2000: # reasonable max distance
            n1, n2, i, j = best_pair
            neighbor_dict[n1].add(n2)
            neighbor_dict[n2].add(n1)
            components[i].update(components[j])
            components.pop(j)
        else:
            break

    # Build distance dict
    distance_dict = {}
    for tl_a in tls_ids:
        for tl_b in neighbor_dict[tl_a]:
            distance_dict[(tl_a, tl_b)] = _dist(tl_a, tl_b)

    result = {t: sorted(list(neighbor_dict.get(t, set()))) for t in tls_ids}
    return result, distance_dict, tls_center


def build_adjacency_matrix(tls_ids, neighbor_dict, distance_dict,
                            distance_scale=500.0):
    """
    Build a distance-weighted adjacency matrix.

    adj[i,j] = 1/(1 + dist/scale) for neighbors, 1.0 on diagonal, 0 otherwise.
    """
    N = len(tls_ids)
    adj = torch.eye(N)
    tls_idx = {t: i for i, t in enumerate(tls_ids)}

    for tl_a in tls_ids:
        i = tls_idx[tl_a]
        for tl_b in neighbor_dict.get(tl_a, []):
            if tl_b not in tls_idx:
                continue
            j = tls_idx[tl_b]
            dist = distance_dict.get(
                (tl_a, tl_b),
                distance_dict.get((tl_b, tl_a), distance_scale),
            )
            weight = 1.0 / (1.0 + dist / distance_scale)
            adj[i, j] = max(weight, 0.1)
            adj[j, i] = max(weight, 0.1)

    return adj


def print_topology_summary(tls_ids, neighbor_dict, distance_dict, tls_center):
    """Print a human-readable topology summary."""
    print(f"\n{'=' * 70}")
    print(f"  SUMO NETWORK TOPOLOGY — {len(tls_ids)} Traffic Light Systems")
    print(f"{'=' * 70}")

    total_conns = sum(len(v) for v in neighbor_dict.values()) // 2
    isolated = [t for t in tls_ids if not neighbor_dict.get(t)]

    for i, tls_id in enumerate(tls_ids):
        pos = tls_center.get(tls_id, (0, 0))
        nbrs = neighbor_dict.get(tls_id, [])
        print(f"  [{i:2d}] {tls_id[:55]}")
        print(f"       pos=({pos[0]:.0f}, {pos[1]:.0f}), neighbors={len(nbrs)}")
        for n in nbrs:
            d = distance_dict.get(
                (tls_id, n), distance_dict.get((n, tls_id), 0)
            )
            j = tls_ids.index(n) if n in tls_ids else -1
            print(f"         -> [{j:2d}] {n[:50]} ({d:.0f}m)")

    print(f"\n  Total connections: {total_conns}")
    print(f"  Isolated nodes  : {len(isolated)}")
    for t in isolated:
        print(f"    ⚠ {t}")
    print(f"{'=' * 70}\n")
