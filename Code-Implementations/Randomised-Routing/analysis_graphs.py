from randomised_routing import get_bit_fixing_path
import random
import collections
import matplotlib.pyplot as plt
import numpy as np

def randomised_routing_wc(num_nodes, d):
    """
    Generates routing paths using Randomized Routing strategy for a specific Bit-Reversal traffic pattern.

    This function implements a two-phase routing scheme to mitigate worst-case congestion. 
    Instead of sending packets directly to the destination (which causes bottlenecks in 
    Bit-Reversal traffic), it sends them to a random intermediate node first.

    1. Phase 1: Route from Source to a randomly selected Intermediate node.
    2. Phase 2: Route from Intermediate node to the Destination (Bit-Reversal of Source).

    Args:
        num_nodes (int): The total number of nodes in the hypercube (2^d).
        d (int): The dimension of the hypercube.

    Returns:
        list: A list of paths, where each path is a list of edge tuples 
              representing the full journey (Phase 1 + Phase 2).
    """
    paths_rand = []
    for src in range(num_nodes):
        # Calculate Destination: The bit-wise reversal of the source address
        # This is the "Worst Case" (wc) pattern for deterministic routing.
        dst = int('{:0{w}b}'.format(src, w=d)[::-1], 2) 

        # Pick a random intermediate node to scatter traffic
        intermediate = random.randint(0, num_nodes - 1)
        
        # Calculate segments
        path_p1 = get_bit_fixing_path(src, intermediate, d)
        path_p2 = get_bit_fixing_path(intermediate, dst, d)
        
        paths_rand.append(path_p1 + path_p2)
        
    return paths_rand

def run_simulation(paths):
    """
    Simulates the movement of packets through the network subject to physical bandwidth constraints.

    This is a discrete-event simulator. The key constraint modeled is that 
    a directed edge (wire) can transmit only one packet per time step. If multiple 
    packets attempt to use the same wire simultaneously, they form a queue.

    Args:
        paths (list): A list of paths for all packets. Each path is a list of edge tuples.

    Returns:
        tuple: A tuple containing:
            - arrival_times (dict): A mapping of {packet_id: time_step_finished}.
            - max_q_depth (int): The maximum number of packets that piled up in 
              any single queue during the entire simulation.
    """
    # Track where each packet is (Index in its path list)
    packet_progress = {pid: 0 for pid in range(len(paths))}
    arrival_times = {}
    
    # The Network State: Queues for every wire
    # Key: Edge Tuple, Value: Queue of Packet IDs
    edge_queues = collections.defaultdict(collections.deque)
    active_packets = set(range(len(paths)))
    
    # Statistic: Track the worst traffic jam seen
    max_q_depth = 0
    
    # 1. Initialization: Load packets into their starting wires
    for pid in list(active_packets):
        if len(paths[pid]) > 0:
            first_edge = paths[pid][0]
            edge_queues[first_edge].append(pid)
        else:
            # Packet starts at destination (Edge case)
            arrival_times[pid] = 0
            active_packets.remove(pid)

    time_step = 0
    
    # 2. The Clock Loop
    while active_packets:
        time_step += 1
        moves_to_make = [] 
        
        # Snapshot current queues to check statistics
        current_max = 0
        if edge_queues:
            current_max = max(len(q) for q in edge_queues.values())
        max_q_depth = max(max_q_depth, current_max)
        
        # 3. Schedule Moves
        # Iterate over a list of keys to allow modifying the dict during iteration
        active_edges = list(edge_queues.keys())
        
        for edge in active_edges:
            queue = edge_queues[edge]
            if queue:
                # BANDWIDTH LIMIT: Only ONE packet pops per edge per step
                pid = queue.popleft() 
                moves_to_make.append(pid)
                
                # Clean up empty queues to keep memory clean
                if not queue:
                    del edge_queues[edge]
        
        #Apply Moves
        for pid in moves_to_make:
            packet_progress[pid] += 1
            
            # Check Arrival
            if packet_progress[pid] == len(paths[pid]):
                arrival_times[pid] = time_step
                active_packets.remove(pid)
            else:
                # Enqueue for next hop
                current_pos = packet_progress[pid]
                next_edge = paths[pid][current_pos]
                edge_queues[next_edge].append(pid)
                
    return arrival_times, max_q_depth

dims_scaling = [4, 6, 8, 10, 12] 

det_latency = []
rand_latency = []
det_max_q = []
rand_max_q = []

for d in dims_scaling:
    num_nodes = 2 ** d
    
    paths_det = []
    paths_rand = []
    
    for src in range(num_nodes):
        # Deterministic: Bit Reversal (Worst Case)
        # e.g., 0001 -> 1000
        dst = int('{:0{w}b}'.format(src, w=d)[::-1], 2)
        paths_det.append(get_bit_fixing_path(src, dst, d))
        
    paths_rand = randomised_routing_wc(num_nodes, d)

    times_d, q_d = run_simulation(paths_det)
    times_r, q_r = run_simulation(paths_rand)
    
    det_latency.append(max(times_d.values()) if times_d else 0)
    rand_latency.append(max(times_r.values()) if times_r else 0)
    
    det_max_q.append(q_d)
    rand_max_q.append(q_r)
    
    print(f"Dimension {d} complete.")

# Data for Histograms (using the last run, Dim 12)
hist_det = list(times_d.values())
hist_rand = list(times_r.values())

#Plotting
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Network Routing Analysis: Deterministic vs Randomized (Dim {dims_scaling[-1]})', fontsize=16)

# Graph 1: Latency Scaling
# Shows how time grows as network size grows
axs[0, 0].plot(dims_scaling, det_latency, marker='o', color='r', linewidth=2, label='Deterministic')
axs[0, 0].plot(dims_scaling, rand_latency, marker='o', color='g', linewidth=2, label='Randomized')
axs[0, 0].set_title('Total Time Taken (Latency)')
axs[0, 0].set_ylabel('Total Time Steps')
axs[0, 0].set_xlabel('Hypercube Dimension')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Graph 2: Queue Depth
axs[0, 1].plot(dims_scaling, det_max_q, marker='s', linestyle='--', color='r', label='Deterministic')
axs[0, 1].plot(dims_scaling, rand_max_q, marker='s', linestyle='--', color='g', label='Randomized')
axs[0, 1].set_title('Max Queue Size')
axs[0, 1].set_ylabel('Max Packets Waiting at One Wire')
axs[0, 1].set_xlabel('Hypercube Dimension')
axs[0, 1].legend()

# Graph 3: Arrival Distribution
axs[1, 0].hist(hist_det, alpha=0.6, color='r', label='Deterministic', density=True)
axs[1, 0].hist(hist_rand, alpha=0.6, color='g', label='Randomized', density=True)
axs[1, 0].set_title('Arrival Time Distribution')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_xlabel('Time Step')
axs[1, 0].legend()

# Graph 4: Completion Rate (CDF)
# Shows the percentage of packets delivered over time
sorted_det = np.sort(hist_det)
sorted_rand = np.sort(hist_rand)
y_det = np.arange(len(sorted_det)) / float(len(sorted_det))
y_rand = np.arange(len(sorted_rand)) / float(len(sorted_rand))

axs[1, 1].plot(sorted_det, y_det, color='r', linewidth=2, label='Deterministic')
axs[1, 1].plot(sorted_rand, y_rand, color='g', linewidth=2, label='Randomized')
axs[1, 1].set_title('Completion (Cumulative)')
axs[1, 1].set_ylabel('Fraction of Packets Delivered')
axs[1, 1].set_xlabel('Time Step')
axs[1, 1].legend(loc='lower right')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()