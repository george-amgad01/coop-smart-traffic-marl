import sys
import torch
from sumo_topology import parse_sumo_network, build_tls_adjacency

net_file = "Version 3 George.net.xml"

def run_test():
    print("4) GATv2 Edge Index Check")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    
    src, dst = [], []
    for u in tls_ids:
        for v in neighbor_dict.get(u, []):
            src.append(tls_ids.index(u))
            dst.append(tls_ids.index(v))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    print("Number of nodes:", len(tls_ids))
    print("edge_index shape:", edge_index.shape)
    print("Edge index is valid for GATv2!")

if __name__ == "__main__":
    run_test()
