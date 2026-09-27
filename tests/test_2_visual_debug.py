"""Debug version of test_2 - saves image and shows detailed connection info."""
import sys
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sumo_topology import parse_sumo_network, build_tls_adjacency

net_file = "Version 3 George.net.xml"

def run_test():
    print("2) Visual Debug Test")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    
    # Print detailed connection info
    print(f"\n{'='*60}")
    print(f"  Total TLS nodes: {len(tls_ids)}")
    total_edges = sum(len(v) for v in neighbor_dict.values()) // 2
    print(f"  Total unique connections: {total_edges}")
    print(f"{'='*60}")
    
    for i, tls in enumerate(tls_ids):
        nbrs = neighbor_dict.get(tls, [])
        status = "!! LEAF" if len(nbrs) <= 1 else "OK"
        short_name = tls[:40]
        print(f"  [{i}] {short_name:42s} | {len(nbrs)} neighbors {status}")
        for n in nbrs:
            j = tls_ids.index(n)
            d = distance_dict.get((tls, n), distance_dict.get((n, tls), 0))
            print(f"       -> [{j}] {n[:40]} ({d:.0f}m)")
    
    # Build graph
    G = nx.Graph()
    short_labels = {}
    for i, tls in enumerate(tls_ids):
        junc = list(net_info['tls_to_junctions'].get(tls, set()))
        label = junc[0] if junc else f"TLS_{i}"
        short_labels[tls] = f"[{i}] {label}"
        pos = tls_center.get(tls, (0, 0))
        G.add_node(tls, pos=pos)

    for u in tls_ids:
        for v in neighbor_dict.get(u, []):
            d = distance_dict.get((u, v), distance_dict.get((v, u), 0))
            G.add_edge(u, v, weight=d)
    
    # Color nodes by degree
    node_colors = []
    for tls in tls_ids:
        deg = len(neighbor_dict.get(tls, []))
        if deg == 0:
            node_colors.append('#ff4444')  # red = isolated
        elif deg == 1:
            node_colors.append('#ffaa00')  # orange = leaf
        else:
            node_colors.append('#44cc44')  # green = well connected
    
    plt.figure(figsize=(14, 10))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw edges with distance labels
    nx.draw_networkx_edges(G, pos, edge_color='#666666', width=2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, 
                           edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels=short_labels, font_size=7, 
                            font_weight='bold')
    
    # Edge distance labels
    edge_labels = {(u, v): f"{d['weight']:.0f}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title(f"SUMO TLS Topology — {len(tls_ids)} nodes, {total_edges} connections\n"
              f"Green=well connected, Orange=leaf (1 neighbor), Red=isolated",
              fontsize=12)
    plt.tight_layout()
    plt.savefig("visual_test_2_debug.png", dpi=150)
    print(f"\n  Saved: visual_test_2_debug.png")

if __name__ == "__main__":
    run_test()
