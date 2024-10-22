import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import argparse

# Step 1: Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Plot agent graph at a specified time step.")
parser.add_argument("time_index", type=int, help="The time step index to plot (1-based).")
args = parser.parse_args()

# Step 2: Load the CSV file into a pandas DataFrame
data = pd.read_csv('agent_positions.csv')

# Step 3: Helper function to calculate the Euclidean distance between two agents
def euclidean_distance(agent1, agent2):
    return np.sqrt((agent1['x'] - agent2['x']) ** 2 + (agent1['y'] - agent2['y']) ** 2)

# Step 4: Get unique time steps
time_steps = data['time'].unique()

# Step 5: Validate the index and get the selected time
if 1 <= args.time_index <= len(time_steps):
    selected_time = time_steps[args.time_index - 1]  # Convert to 0-based index
else:
    raise ValueError("Time index is out of bounds. Please choose a valid index.")

# Step 6: Filter the data for the selected time step
time_data = data[data['time'] == selected_time]

# Step 7: Create a new graph
G = nx.Graph()

# Add nodes (agents) to the graph
for _, agent in time_data.iterrows():
    G.add_node(agent['agent_id'], pos=(agent['x'], agent['y']), is_tx=agent['is_tx'], is_rx=agent['is_rx'])

# Step 8: Create edges between agents within the detection range
detection_range = time_data['dist_detection_range'].iloc[0]  # Assuming all agents have the same detection range

# Compare every pair of agents (combinations) and create an edge if they are within the detection range
for agent1, agent2 in combinations(time_data.iterrows(), 2):
    agent1_data = agent1[1]  # Access the agent's row data
    agent2_data = agent2[1]

    distance = euclidean_distance(agent1_data, agent2_data)
    if distance <= detection_range:
        # Add an edge with the weight as the distance
        G.add_edge(agent1_data['agent_id'], agent2_data['agent_id'], weight=round(distance, 2))

# Step 9: Define node colors based on attributes
node_colors = []
for _, agent in time_data.iterrows():
    if agent['is_rx']:
        node_colors.append('yellow')  # Receiver (RX)
    elif agent['is_tx']:
        node_colors.append('red')     # Transmitter (TX)
    else:
        node_colors.append('lightblue')  # Regular agent

# Step 10: Draw the graph for visualization
pos = nx.get_node_attributes(G, 'pos')
labels = {node: node for node in G.nodes}
edge_weights = nx.get_edge_attributes(G, 'weight')

plt.figure(f"Agent Graph at Time {selected_time}", figsize=(8, 8))

# Draw the graph with custom node colors
nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, node_size=500)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

plt.title(f"Agent Graph at Time {selected_time}")

# Show the plot
plt.show()
