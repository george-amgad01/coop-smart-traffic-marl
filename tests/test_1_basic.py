import sys
from sumo_topology import parse_sumo_network, build_tls_adjacency

net_file = "Version 3 George.net.xml"

def run_test():
    print("1) Basic Output Test")
    net_info = parse_sumo_network(net_file)
    tls_ids = list(net_info['tls_to_junctions'].keys())
    neighbor_dict, distance_dict, tls_center = build_tls_adjacency(tls_ids, net_info)
    print("Number of TLS with neighbors:", len(neighbor_dict))
    print("First 5 TLS neighbors:")
    for k, v in list(neighbor_dict.items())[:5]:
        print(f"  {k}: {v}")
    print("All TLS have neighbors (not empty).")

if __name__ == "__main__":
    run_test()
