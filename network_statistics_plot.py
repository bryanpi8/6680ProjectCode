import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms.connectivity import node_disjoint_paths
from networkx.algorithms.flow import maximum_flow

# Step 1: Load the CSV file into a pandas DataFrame
data = pd.read_csv('agent_positions.csv')

# Step 2: Helper function to calculate the Euclidean distance between two agents
def euclidean_distance(agent1, agent2):
    return np.sqrt((agent1['x'] - agent2['x']) ** 2 + (agent1['y'] - agent2['y']) ** 2)

# Step 3: Function to calculate network metrics for each time step
def calculate_network_metrics(G):
    metrics = {}

    # 1. Network Density
    metrics['density'] = nx.density(G)

    # 2. Average Path Length (unweighted since all weights are treated as 1)
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G, weight=None)
    else:
        metrics['avg_path_length'] = np.nan  # For disconnected graphs

    # 3. Network Diameter (only for connected graphs)
    if nx.is_connected(G):
        metrics['diameter'] = nx.diameter(G)
    else:
        metrics['diameter'] = np.nan  # Undefined for disconnected graphs

    # 4. Network Robustness (Node Connectivity)
    metrics['robustness'] = nx.node_connectivity(G)

    return metrics

# Step 4: Helper function to calculate TX-RX specific metrics
def calculate_tx_rx_metrics(G, tx_nodes, rx_nodes):
    tx_rx_metrics = {
        'shortest_path': [],
        'effective_resistance': [],
        'disjoint_paths': [],
        # 'network_flow': []
    }

    for tx in tx_nodes:
        for rx in rx_nodes:
            # 1. Shortest Path
            try:
                shortest_path = nx.shortest_path_length(G, source=tx, target=rx, weight=None)
            except nx.NetworkXNoPath:
                shortest_path = np.nan
            tx_rx_metrics['shortest_path'].append(shortest_path)

            # 2. Effective Resistance (Approximation: Treat disconnected nodes as infinite resistance)
            try:
                resistance = nx.resistance_distance(G, tx, rx, weight=None)
            except nx.NetworkXError:
                resistance = np.nan
            tx_rx_metrics['effective_resistance'].append(resistance)

            # 3. Number of Node Disjoint Paths
            try:
                disjoint_paths = len(list(node_disjoint_paths(G, tx, rx)))
            except (nx.NetworkXNoPath, nx.NetworkXError):
                disjoint_paths = 0  # No disjoint paths if there's no connection
            tx_rx_metrics['disjoint_paths'].append(disjoint_paths)

            # # 4. Network Flow
            # try:
            #     flow_value, _ = maximum_flow(G, tx, rx)
            # except (nx.NetworkXNoPath, nx.NetworkXError):
            #     flow_value = 0  # No flow if there's no connection
            # tx_rx_metrics['network_flow'].append(flow_value)

    return tx_rx_metrics

# Step 5: Iterate over each unique time and create a graph
time_steps = data['time'].unique()

# Store the overall and TX-RX specific metrics for plotting later
all_metrics = []
tx_rx_metrics_over_time = {
    'time': [],
    'shortest_path': [],
    'effective_resistance': [],
    'disjoint_paths': [],
    # 'network_flow': []
}

for time in time_steps:
    # Step 6: Filter the data for the current time step
    time_data = data[data['time'] == time]

    # Step 7: Create a new graph
    G = nx.Graph()

    # Add nodes (agents) to the graph
    for _, agent in time_data.iterrows():
        G.add_node(agent['agent_id'], pos=(agent['x'], agent['y']), is_tx=agent['is_tx'], is_rx=agent['is_rx'])

    # Step 8: Create edges between agents within the detection range
    detection_range = time_data['dist_detection_range'].iloc[0]  # Assuming all agents have the same detection range

    for agent1, agent2 in combinations(time_data.iterrows(), 2):
        agent1_data = agent1[1]  # Access the agent's row data
        agent2_data = agent2[1]

        distance = euclidean_distance(agent1_data, agent2_data)
        if distance <= detection_range:
            # Add an edge with the weight as 1 (since all weights are treated as 1)
            G.add_edge(agent1_data['agent_id'], agent2_data['agent_id'], weight=1)

    # Step 9: Calculate network-wide metrics for this time step
    metrics = calculate_network_metrics(G)
    metrics['time'] = time  # Add the time step to the metrics
    all_metrics.append(metrics)

    # Step 10: Identify TX and RX nodes
    tx_nodes = [n for n, attr in G.nodes(data=True) if attr['is_tx']]
    rx_nodes = [n for n, attr in G.nodes(data=True) if attr['is_rx']]

    # Step 11: Calculate TX-RX specific metrics for each time step
    if tx_nodes and rx_nodes:
        tx_rx_metrics = calculate_tx_rx_metrics(G, tx_nodes, rx_nodes)
        tx_rx_metrics_over_time['time'].extend([time] * len(tx_rx_metrics['shortest_path']))
        tx_rx_metrics_over_time['shortest_path'].extend(tx_rx_metrics['shortest_path'])
        tx_rx_metrics_over_time['effective_resistance'].extend(tx_rx_metrics['effective_resistance'])
        tx_rx_metrics_over_time['disjoint_paths'].extend(tx_rx_metrics['disjoint_paths'])

# Step 12: Convert the list of metrics to DataFrames for easier plotting
metrics_df = pd.DataFrame(all_metrics)
tx_rx_metrics_df = pd.DataFrame(tx_rx_metrics_over_time)

# Step 13: Plot the network-wide metrics against time
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Network-Wide Metrics Over Time')

# Plot Network Density
axs[0, 0].plot(metrics_df['time'], metrics_df['density'], marker='o', color='b')
axs[0, 0].set_title('Network Density')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Density')

# Plot Average Path Length
axs[0, 1].plot(metrics_df['time'], metrics_df['avg_path_length'], marker='o', color='g')
axs[0, 1].set_title('Average Path Length')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Avg Path Length')

# Plot Network Diameter
axs[1, 0].plot(metrics_df['time'], metrics_df['diameter'], marker='o', color='r')
axs[1, 0].set_title('Network Diameter')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Diameter')

# Plot Network Robustness (Node Connectivity)
axs[1, 1].plot(metrics_df['time'], metrics_df['robustness'], marker='o', color='m')
axs[1, 1].set_title('Network Robustness (Node Connectivity)')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Node Connectivity')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Step 14: Plot the TX-RX specific metrics against time
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TX-RX Metrics Over Time')

# Plot Shortest Path Between TX and RX
axs[0, 0].plot(tx_rx_metrics_df['time'], tx_rx_metrics_df['shortest_path'], marker='o', color='b')
axs[0, 0].set_title('Shortest Path Between TX and RX')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Shortest Path Length')

# Plot Effective Resistance Between TX and RX
axs[0, 1].plot(tx_rx_metrics_df['time'], tx_rx_metrics_df['effective_resistance'], marker='o', color='g')
axs[0, 1].set_title('Effective Resistance Between TX and RX')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Effective Resistance')

# Plot Number of Disjoint Paths Between TX and RX
axs[1, 0].plot(tx_rx_metrics_df['time'], tx_rx_metrics_df['disjoint_paths'], marker='o', color='r')
axs[1, 0].set_title('Number of Disjoint Paths Between TX and RX')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Number of Disjoint Paths')

# # Plot Network Flow Between TX and RX
# axs[1, 1].plot(tx_rx_metrics_df['time'], tx_rx_metrics_df['network_flow'], marker='o', color='m')
# axs[1, 1].set_title('Network Flow Between TX and RX')
# axs[1, 1].set_xlabel('Time')
# axs[1, 1].set_ylabel('Network Flow')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

'''
TX-RX Metrics
1. Shortest Path

    Definition: The shortest path is the minimum number of edges (or the minimum total weight in a weighted graph) between two nodes.
    Interpretation: This measures how "close" two nodes are within the network. A shorter path suggests a more direct relationship or connection between the two nodes.
    Application: In a communication network, it could represent the minimum number of hops between two computers or devices.

3. Effective Resistance (Electrical Distance)

    Definition: The effective resistance between two nodes is derived from treating the network as an electrical circuit where edges represent resistors. The metric reflects how easily current (or information) can flow between two nodes.
    Interpretation: Lower effective resistance indicates a stronger, more direct connection between the nodes. This metric considers all possible paths between two nodes, not just the shortest one.
    Application: This is useful in evaluating redundancy, since multiple paths between two nodes reduce the resistance.

8. Number of Disjoint Paths (Edge/Node Disjoint)

    Definition: The number of disjoint paths between two nodes counts the number of paths between them that don’t share any edges or nodes.
    Interpretation: This indicates how many independent ways two nodes can communicate. More disjoint paths mean higher fault tolerance or redundancy in the connection between the two nodes.
    Application: In a computer network, multiple disjoint paths ensure that communication between two devices can continue even if some connections fail.

Network Metrics
1. Network Density

    Definition: The ratio of the number of actual edges to the number of possible edges in the network. In weighted networks, density can be adapted to consider the total weight of all edges relative to the maximum possible weight.
    Interpretation: In a computer network, density measures how interconnected the network is. High density means most computers are directly connected, while low density indicates a sparse network.
    Impact: High-density networks are typically more robust and fault-tolerant, as there are more alternative paths available for communication. However, very dense networks may suffer from increased overhead or congestion.

2. Average Weighted Path Length

    Definition: The average of the shortest weighted paths between all pairs of nodes in the network. The weights represent costs, such as distance, bandwidth, or latency.
    Interpretation: This reflects the average efficiency of communication across the network, considering the costs associated with each link.
    Impact: Shorter average path lengths indicate a more efficient network, where communication can occur quickly and at lower cost. Longer paths suggest potential latency issues or inefficiencies.

Average Weighted Path Length=1N(N−1)2∑i≠jdij
Average Weighted Path Length=2N(N−1)​1​i=j∑​dij​

where dijdij​ is the weighted shortest path between nodes ii and jj.

3. Network Diameter (Weighted)

    Definition: The longest shortest weighted path between any two nodes in the network.
    Interpretation: In a computer network, the diameter reflects the worst-case scenario for communication delay or cost. A smaller diameter indicates that even the most distant nodes can communicate efficiently, while a larger diameter may signal potential bottlenecks.
    Impact: Networks with smaller diameters are better connected and have lower latency for communication between distant parts.

7. Network Robustness (Weighted Connectivity)

    Definition: A measure of how resilient the network is to node or edge failures, considering the weights of edges. This can include metrics like weighted node connectivity (the minimum number of nodes that need to be removed to disconnect the network) and weighted edge connectivity (the minimum number of edges that need to be removed).
    Interpretation: In a communication network, higher robustness means the network can withstand failures without major disruptions. Networks with higher connectivity tend to have redundant paths that help maintain communication even if certain links or nodes fail.
    Impact: Robust networks are fault-tolerant and ensure reliable communication, even in the face of failures or attacks.
'''