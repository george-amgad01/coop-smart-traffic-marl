import sys
import networkx as nx
import matplotlib.pyplot as plt
from sumo_topology import parse_sumo_network, build_tls_adjacency

net_file = "Version 3 George.net.xml"

def run_test():
    print("2) Visual Test")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    
    G = nx.Graph()
    for tls in tls_ids:
        pos = tls_center.get(tls, (0, 0))
        G.add_node(tls, pos=pos)

    for u in tls_ids:
        for v in neighbor_dict.get(u, []):
            G.add_edge(u, v)
                
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    if pos:
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold')
    else:
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold')
    
    plt.title("SUMO Traffic Light Topology Visual Test")
    plt.show()

if __name__ == "__main__":
    run_test()