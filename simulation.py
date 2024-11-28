import argparse
import math
import random
import signal
import sys
import time
from collections import deque

import pandas as pd
import pygame

from leader_follower_final import (
    COMM_RANGE,
    DIST_DETECTION_RANGE,
    MAX_SPEED,
    AGENT_RADIUS,
    Agent,
)

### PYGAME PARAMETERS ###

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Search and Communication Simulation")
font = pygame.font.SysFont(None, 12)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GREY = (128, 128, 128)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
PINK = (255, 128, 170)

### SIMULATION PARAMETERS ###

FPS = 100

AGENT_COUNT = 16

RX_SPAWN_X = WIDTH / 2
RX_SPAWN_Y = HEIGHT / 2
TX_SPAWN_X = 175
TX_SPAWN_Y = 175

MAX_AGENT_INIT_RADIUS = AGENT_RADIUS * 10  # Maximum radius for placement around rx
MIN_AGENT_INIT_DISTANCE = AGENT_RADIUS * 4  # Miniimum radius for placement around rx

# Create a pandas DataFrame to store agents' positions and statuses
data = pd.DataFrame(columns=["frame", "agent_id", "x", "y", "is_tx", "is_rx"])

### CMD Argument Parsing ###
parser = argparse.ArgumentParser(
    description="Agent Search and Communication Simulation"
)

parser.add_argument(
    "--network_mass_ratio",
    type=float,
    default=0.5,
    help="Ratio of agents in network_mass ruleset (0.0 to 1.0)",
)

# Parse the arguments
args = parser.parse_args()
NETWORK_MASS_RATIO = args.network_mass_ratio

### SIMULATION METHODS ###

data_list = []
critical_chain = []
all_agents = []

# Centralised critical chain search


def find_critical_chain(tx_agent, rx_agent):
    global critical_chain
    if len(critical_chain) == 0:
        for agent in all_agents:
            agent.part_of_critical_chain = False
            agent.ruleset = agent.initial_ruleset
            agent.ruleset = agent.initial_ruleset
            if agent.is_rx:
                agent.ruleset = "rx"
                agent.part_of_critical_chain = True
            elif agent.is_tx:
                agent.ruleset = "tx"
                agent.part_of_critical_chain = True

    queue = deque([rx_agent])
    visited = {rx_agent: 0}
    predecessors = {rx_agent: None}

    while queue:
        current_agent = queue.popleft()

        if current_agent is tx_agent:
            break

        for other_agent in all_agents:
            if (
                other_agent not in visited
                and current_agent.distance_to(other_agent) < COMM_RANGE
            ):
                visited[other_agent] = visited[current_agent] + 1
                predecessors[other_agent] = current_agent
                queue.append(other_agent)

    if len(critical_chain) == 0:
        current_agent = tx_agent
    else:
        current_agent = critical_chain[-1]
        critical_chain = critical_chain[:-1]
    while current_agent is not None:
        critical_chain.append(current_agent)
        current_agent = predecessors.get(current_agent)

    hop_counter = 0
    for critical_agent in critical_chain:
        critical_agent.part_of_critical_chain = True
        if not (critical_agent.is_tx or critical_agent.is_rx):
            critical_agent.ruleset = "critical_chain"
        critical_agent.hop_count = hop_counter
        hop_counter += 1


def is_critical_path_valid(dist_detection_range):
    """
    Check if the critical path is still valid based on the communication radius.
    """
    for i in range(len(critical_chain) - 1):
        if critical_chain[i].distance_to(critical_chain[i + 1]) > dist_detection_range:
            return False
    return True


def calculate_random_position(rx_agent, max_radius):
    # Random angle between 0 and 2 * pi
    angle = random.uniform(0, 2 * math.pi)

    # Random radius (between 0 and max_radius)
    radius = random.uniform(0, max_radius)

    # Convert polar coordinates to Cartesian (x, y)
    random_x = rx_agent.x + radius * math.cos(angle)
    random_y = rx_agent.y + radius * math.sin(angle)

    return random_x, random_y


# Function to check if the new agent has sufficient distance from all existing agents
def is_valid_position(rx_agent, new_agent, agents, min_distance):
    if new_agent.distance_to(rx_agent) < min_distance:
        return False
    for agent in agents:
        if new_agent.distance_to(agent) < min_distance:
            return False
    return True


def handle_sigint(signal, frame):
    global data_list, pygame
    # Save the data to CSV
    print("Interrupt received, saving data to CSV and quitting...")
    data = pd.DataFrame(data_list)
    data.to_csv("agent_positions.csv", index=False)
    pygame.quit()
    sys.exit(0)


### SIMULATION LOGIC ###

# Register the signal handler
signal.signal(signal.SIGINT, handle_sigint)

# Initialize agents, TX, and RX.
tx = Agent(TX_SPAWN_X, TX_SPAWN_Y)
tx.is_tx = True

rx = Agent(RX_SPAWN_X, RX_SPAWN_Y)
rx.is_rx = True

# Have one agent that will ensure a critical chain exists at the beginning of the simulation
critical_agent_1 = Agent(RX_SPAWN_X - 10, RX_SPAWN_X - 10)

all_agents = [rx, tx, critical_agent_1]

# Loop to place agents
while len(all_agents) < AGENT_COUNT:
    # calculate random position
    agent_x, agent_y = calculate_random_position(rx, MAX_AGENT_INIT_RADIUS)

    # Create a new agent
    new_agent = Agent(agent_x, agent_y)

    # Check if the new agent is at least MIN_DISTANCE away from all others
    if is_valid_position(rx, new_agent, all_agents, MIN_AGENT_INIT_DISTANCE):
        all_agents.append(new_agent)

# Main simulation loop
running = True
dt = 0
clock = pygame.time.Clock()
start_time = time.time()
program_start_time = time.time()

find_critical_chain(tx, rx)

# Filter out RX and TX agents
non_rx_tx_agents = [
    agent for agent in all_agents if not agent.is_rx and not agent.is_tx
]

# Calculate the distance to TX for each agent
non_rx_tx_agents.sort(key=lambda agent: agent.distance_to(tx))  # Sort by distance to TX

# Split the sorted agents into two halves
half_count = int(len(non_rx_tx_agents) * NETWORK_MASS_RATIO)

# Assign target_critical_chain based on proximity to TX
for i, agent in enumerate(non_rx_tx_agents):
    if i < half_count:
        agent.ruleset = "network_mass"
        agent.initial_ruleset = "network_mass"

    else:
        agent.ruleset = "chain_extension"
        agent.initial_ruleset = "chain_extension"

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if the critical path still exists
    if not is_critical_path_valid(COMM_RANGE):
        critical_chain = critical_chain[:-1]
        find_critical_chain(tx, rx)

    screen.fill(WHITE)

    # Update and draw agents
    for agent in all_agents:
        if agent.is_tx:
            # Set velocity towards (0, 0) with a desired speed (for example, speed = 2.0)
            tx.move_at_velocity_towards(0, 0, 0.15 * MAX_SPEED, dt)
        else:
            agent.move(dt, critical_chain, all_agents, rx)  # Pass tx

        # Draw agent
        if agent.part_of_critical_chain and not agent.is_rx and not agent.is_tx:
            color = GREEN  # Critical chain agents are green
        elif agent.is_rx:
            color = CYAN
        elif agent.is_tx:
            color = BLACK
        elif agent.ruleset == "network_mass":
            color = BLUE
        else:
            color = PINK
        pygame.draw.circle(screen, color, (int(agent.x), int(agent.y)), AGENT_RADIUS)
        pygame.draw.circle(
            screen, GREY, (int(agent.x), int(agent.y)), COMM_RANGE, width=1
        )

        # Render hop count
        hop_count_text = font.render(str(agent.hop_count), True, WHITE)
        text_rect = hop_count_text.get_rect(
            center=(int(agent.x), int(agent.y) - AGENT_RADIUS - 10)
        )
        screen.blit(hop_count_text, text_rect)

    # Draw TX and RX
    pygame.draw.circle(screen, BLUE, (int(rx.x), int(rx.y)), COMM_RANGE, width=1)

    # Update display
    pygame.display.flip()

    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # Save every second
        for i, agent in enumerate(all_agents):
            data_list.append(
                {
                    "time": round(time.time() - program_start_time, 2),
                    "agent_id": i,
                    "x": agent.x,
                    "y": agent.y,
                    "is_tx": agent.is_tx,
                    "is_rx": agent.is_rx,
                    "dist_detection_range": DIST_DETECTION_RANGE,
                }
            )
        start_time = time.time()  # Reset start time

    dt = clock.tick(FPS)
