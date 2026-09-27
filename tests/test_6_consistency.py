import sys
import numpy as np
from sumo_topology import parse_sumo_network, build_tls_adjacency, build_adjacency_matrix

net_file = "Version 3 George.net.xml"

def run_test():
    print("6) Graph Consistency Score")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    adj_tensor = build_adjacency_matrix(tls_ids, neighbor_dict, distance_dict)
    adjacency = adj_tensor.numpy()
    
    # Consistency check: is the graph symmetric?
    is_symmetric = np.allclose(adjacency, adjacency.T)
    print(f"Is Adjacency Matrix Symmetric? {is_symmetric}")
    
    # Are there self-loops?
    has_self_loops = np.any(np.diag(adjacency) > 0)
    print(f"Has self loops? {has_self_loops}")
    
    # Are there isolated nodes? (Check degree ignoring self-loops)
    np.fill_diagonal(adjacency, 0)
    isolated = [tls_ids[i] for i in range(len(tls_ids)) if np.sum(adjacency[i]) == 0]
    print(f"Isolated Nodes: {isolated}")
    
    score = 100
    if not is_symmetric: score -= 20
    if not has_self_loops: score -= 10
    score -= len(isolated) * 5
    
    print(f"Graph Consistency Score: {max(0, score)}/100")

if __name__ == "__main__":
    run_test()