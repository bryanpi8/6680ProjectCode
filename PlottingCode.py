import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Step 1: Load the CSV file into a pandas DataFrame
data = pd.read_csv('agent_positions.csv')

# Step 2: Helper function to calculate the Euclidean distance between two agents
def euclidean_distance(agent1, agent2):
    return np.sqrt((agent1['x'] - agent2['x']) ** 2 + (agent1['y'] - agent2['y']) ** 2)

# Step 3: Iterate over each unique time and create a graph
time_steps = data['time'].unique()

for time in time_steps:
    # Step 4: Filter the data for the current time step
    time_data = data[data['time'] == time]

    # Step 5: Create a new graph
    G = nx.Graph()

    # Add nodes (agents) to the graph
    for _, agent in time_data.iterrows():
        G.add_node(agent['agent_id'], pos=(agent['x'], agent['y']), is_tx=agent['is_tx'], is_rx=agent['is_rx'])

    # Step 6: Create edges between agents within the detection range
    detection_range = time_data['dist_detection_range'].iloc[0]  # Assuming all agents have the same detection range

    # Compare every pair of agents (combinations) and create an edge if they are within the detection range
    for agent1, agent2 in combinations(time_data.iterrows(), 2):
        agent1_data = agent1[1]  # Access the agent's row data
        agent2_data = agent2[1]

        distance = euclidean_distance(agent1_data, agent2_data)
        if distance <= detection_range:
            # Add an edge with the weight as the distance
            G.add_edge(agent1_data['agent_id'], agent2_data['agent_id'], weight=round(distance, 2))

    # Step 7: Define node colors based on attributes
    node_colors = []
    for _, agent in time_data.iterrows():
        if agent['is_rx']:
            node_colors.append('yellow')  # Receiver (RX)
        elif agent['is_tx']:
            node_colors.append('red')     # Transmitter (TX)
        else:
            node_colors.append('lightblue')  # Regular agent

    # Step 8: Draw the graph for visualization
    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes}
    edge_weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(f"Agent Graph at Time {time}", figsize=(8, 8))

    # Draw the graph with custom node colors
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

    plt.title(f"Agent Graph at Time {time}")

    # Show the plot
    plt.show()

    break  # Optional: Remove this to process all time steps

    # You can also save the graph if needed
    # nx.write_gml(G, f"graph_time_{time}.gml")
