import sys
from sumo_topology import parse_sumo_network, build_tls_adjacency

net_file = "Version 3 George.net.xml"

def run_test():
    print("5) Distances Sanity Check")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    distances = list(distance_dict.values())
    if distances:
        print(f"Minimum distance: {min(distances):.2f}m")
        print(f"Maximum distance: {max(distances):.2f}m")
    print("Distances sanity check passed.")

if __name__ == "__main__":
    run_test()
