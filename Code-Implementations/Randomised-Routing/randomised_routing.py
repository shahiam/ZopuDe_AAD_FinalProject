import random

def get_bit_fixing_path(src, dst, dims):
    """
    Calculates the deterministic bit-fixing path between two nodes in a Hypercube.

    This algorithm traverses the hypercube by fixing the address bits from the
    least significant bit (dimension 0) to the most significant bit.

    Args:
        src (int): The source node identifier (integer representation).
        dst (int): The destination node identifier.
        dims (int): The dimension of the hypercube.

    Returns:
        list of tuple: A list of edges representing the path. Each edge is a 
        sorted tuple (u, v) to ensure undirected edge representation.
    """
    path_edges = []
    current = src
    diff = src ^ dst
    
    for i in range(dims):
        # Check if the i-th bit is different
        if (diff >> i) & 1:
            next_node = current ^ (1 << i)
            
            # We sort the edge tuple (u, v) to ensure that traffic from A->B 
            # and B->A counts as using the same wire. This creates the collision.
            edge = tuple(sorted((current, next_node)))
            path_edges.append(edge)
            current = next_node
            
    return path_edges

def randomised_routing(num_nodes, d):
    """
    Simulates Randomized Routing algorithm for a random permutation pattern.

    This implements the "Two-Phase" routing strategy to avoid worst-case congestion:
    1. Phase 1: Route from Source -> Random Intermediate Node.
    2. Phase 2: Route from Intermediate Node -> Destination.

    Args:
        num_nodes (int): Total number of nodes in the network (usually 2^d).
        d (int): The dimension of the hypercube.

    Returns:
        list of list: A list of full paths for every node in the network. 
        Each item is a list of edges.
    """
    paths_rand = []
    
    # Generate a random permutation of destinations
    # (i.e., Node 0 goes to dests[0], Node 1 goes to dests[1], etc.)
    dests = list(range(num_nodes))
    random.shuffle(dests)

    for src in range(num_nodes):
        dst = dests[src]
        
        # Pick a random intermediate node (Valiant's Phase 1 target)
        intermediate = random.randint(0, num_nodes - 1)
        
        print(f'{src} --> {intermediate} --> {dst}')
        
        # Phase 1: Src -> Intermediate
        path_p1 = get_bit_fixing_path(src, intermediate, d)
        
        # Phase 2: Intermediate -> Dst
        path_p2 = get_bit_fixing_path(intermediate, dst, d)
        
        # Combine phases
        paths_rand.append(path_p1 + path_p2)

    return paths_rand

def main():
    """
    Main execution block to set dimensions and run the routing simulation.
    """
    d = 5
    randomised_routing(2**d, d)

if __name__ == "__main__":
    main()