
import pygame
import random
import math
from collections import deque
import numpy as np
import pandas as pd  # Import pandas for saving data to CSV
import signal
import sys
import time  # Import time to track seconds

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Search and Communication Simulation")

font = pygame.font.SysFont(None, 12)  # You can change the size as needed

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Simulation parameters
FPS = 20
AGENT_COUNT = 15
AGENT_RADIUS = 5
COMM_RANGE = AGENT_RADIUS * 5
DIST_DETECTION_RANGE = COMM_RANGE
MAX_SPEED = 1
BLEND_FACTOR = 0.3

RXSPAWNX = WIDTH/2
RXSPAWNY = HEIGHT/2

MAX_AGENT_INIT_RADIUS = AGENT_RADIUS * 15  # Define a maximum radius for placement around rx
MIN_AGENT_INIT_DISTANCE = AGENT_RADIUS * 4

CHAIN_MULITIPLIER = 1.3
REPULSION_CONSTANT = 0.02   # Constant for repulsive force
ATTRACTION_CONSTANT = 0.0002  # Constant for attractive force
CHAIN_DELAY = 40 # Constant for slowing down chain updates, one update per N moves


movement = 0
# Create a pandas DataFrame to store agents' positions and statuses
data = pd.DataFrame(columns=['frame', 'agent_id', 'x', 'y', 'is_tx', 'is_rx'])

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = np.array([0.0, 0.0])  # Initial velocity
        self.acceleration = np.array([0.0, 0.0])  # Initial acceleration
        self.is_rx = False  # is rx (is the start point)
        self.is_tx = False
        self.part_of_chain = False  # Whether this agent is part of a chain
        self.part_of_critical_chain = False  # New attribute for critical chain
        self.hop_count = -1  # Initialize hop count to infinity

    def position(self):
        return np.array([self.x, self.y])

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def check_part_of_chain(self, all_agents):
        global tx
        if self.distance_to(tx) < DIST_DETECTION_RANGE:
            if (not self.part_of_chain):
                self.hop_count = 1  # TX has hop count 0
            self.part_of_chain = True
            return

        for other_agent in all_agents:
            if other_agent.part_of_chain and self.distance_to(other_agent) < DIST_DETECTION_RANGE:
                if (not self.part_of_chain):
                    self.hop_count = max(self.hop_count, other_agent.hop_count + 1)  # Increment hop count from the repeater
                self.part_of_chain = True
                return

    # def calculate_net_force(self, all_agents):
    #     max_force_distance = DIST_DETECTION_RANGE * 3  # Maximum distance for force to apply

    #     net_force = np.array([0.0, 0.0])  # Initialize the net force as zero

    #     for other_agent in all_agents:
    #         if other_agent is not self:  # Don't calculate force from the agent itself
    #             # Calculate the displacement vector between agents
    #             displacement = np.array([other_agent.x - self.x, other_agent.y - self.y])
    #             distance = np.linalg.norm(displacement)

    #             if distance > 0 and distance < max_force_distance:  # Avoid division by zero, cap at max distance
    #                 # Normalize displacement to get the direction of force
    #                 direction = displacement / distance

    #                 # Linear relationship for attraction (force increases with distance)
    #                 attractive_force_magnitude = ATTRACTION_CONSTANT * distance
    #                 attractive_force = direction * attractive_force_magnitude

    #                 # Linear relationship for repulsion (force decreases with distance, stronger when close)
    #                 # The repulsive force is strongest when the agents are closest (distance approaches 0)
    #                 repulsive_force_magnitude = REPULSION_CONSTANT * (1 - (distance / max_force_distance))
    #                 repulsive_force = -direction * repulsive_force_magnitude

    #                 # Combine the forces
    #                 net_force += attractive_force + repulsive_force

    #     return net_force

    def calculate_net_force(self, all_agents):
        max_force_distance = DIST_DETECTION_RANGE * 2  # Maximum distance for force to apply
        net_force = np.array([0.0, 0.0])  # Initialize the net force as zero

        for other_agent in all_agents:
            if other_agent is not self:  # Don't calculate force from the agent itself
                # Calculate the displacement vector between agents
                displacement = np.array([other_agent.x - self.x, other_agent.y - self.y])
                distance = np.linalg.norm(displacement)

                if distance > 0 and distance < max_force_distance:  # Avoid division by zero, cap at max distance
                    direction = displacement / distance

                    # Force Alteration Between Critical Chain Agents
                    if self.part_of_critical_chain and other_agent.part_of_critical_chain:
                        # Linear relationship for repulsion (force decreases with distance, stronger when close)
                        # The repulsive force is strongest when the agents are closest (distance approaches 0)
                        repulsive_force_magnitude = REPULSION_CONSTANT * (1 - (distance / max_force_distance)) * CHAIN_MULITIPLIER
                        # repulsive_force_magnitude = REPULSION_CONSTANT * distance ### Particle Ejection
                        repulsive_force = -direction * repulsive_force_magnitude
                        
                        attractive_force_magnitude = ATTRACTION_CONSTANT * distance * CHAIN_MULITIPLIER
                        attractive_force = direction * attractive_force_magnitude

                        # Combine the forces
                        net_force += repulsive_force
                        net_force += attractive_force
                    else:

                        # if other_agent.is_rx or other_agent.is_tx:
                        if other_agent.part_of_critical_chain:
                            # Linear relationship for attraction (force increases with distance)
                            attractive_force_magnitude = ATTRACTION_CONSTANT * distance * 1.3
                            attractive_force = direction * attractive_force_magnitude
                        
                        elif other_agent.is_rx:
                            # Linear relationship for attraction (force increases with distance)
                            attractive_force_magnitude = ATTRACTION_CONSTANT * distance * 1.1
                            attractive_force = direction * attractive_force_magnitude
                        
                        else:
                            attractive_force_magnitude = ATTRACTION_CONSTANT * distance 
                            attractive_force = direction * attractive_force_magnitude

                        ## Hop Count Based Movement
                        # if other_agent.hop_count < self.hop_count and other_agent.part_of_critical_chain: 
                        #     # Linear relationship for attraction (force increases with distance)
                        #     attractive_force_magnitude = ATTRACTION_CONSTANT * distance * 1.1
                        #     attractive_force = direction * attractive_force_magnitude
                        # else:
                        #     # Linear relationship for attraction (force increases with distance)
                        #     attractive_force_magnitude = ATTRACTION_CONSTANT * distance
                        #     attractive_force = direction * attractive_force_magnitude

                        # if other_agent.hop_count < self.hop_count and other_agent.part_of_critical_chain: 
                        #     # Linear relationship for attraction (force increases with distance)
                        #     repulsive_force_magnitude = 1.05*REPULSION_CONSTANT * (1 - (distance / max_force_distance))
                        #     repulsive_force = -direction * repulsive_force_magnitude
                        # else:
                        #     # Linear relationship for attraction (force increases with distance)
                        #     repulsive_force_magnitude = REPULSION_CONSTANT * (1 - (distance / max_force_distance))
                        #     repulsive_force = -direction * repulsive_force_magnitude

                        # Linear relationship for repulsion (force decreases with distance, stronger when close)
                        # The repulsive force is strongest when the agents are closest (distance approaches 0)
                        repulsive_force_magnitude = REPULSION_CONSTANT * (1 - (distance / max_force_distance))
                        repulsive_force = -direction * repulsive_force_magnitude

                        # Combine the forces
                        net_force += attractive_force + repulsive_force

        return net_force


    def move(self, all_agents, rx, tx):
        # Calculate the net force from neighbors
        net_force = self.calculate_net_force(all_agents)

        # Update the acceleration based on net force (F = ma, with m = 1, so a = F)
        self.acceleration = net_force

        # Update the density map based on the agent's current position (unchanged from original code)
        cell_x = int(self.x // (WIDTH // density_map.shape[1]))
        cell_y = int(self.y // (HEIGHT // density_map.shape[0]))

        # Increment the density value for the current cell (unchanged)
        if 0 <= cell_x < density_map.shape[1] and 0 <= cell_y < density_map.shape[0]:
            density_map[cell_y, cell_x] += 1

        # Update velocity and limit it to a maximum speed
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED

        # Apply damping to simulate friction
        self.velocity *= 0.9

        # if self.part_of_critical_chain:
        #     self.velocity *= 0.5
            # self.velocity = np.array([0.0, 0.0])

        # Update position (p = p + v)
        self.x += dt * self.velocity[0]
        self.y += dt * self.velocity[1]

        # Apply bounds (walls)
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def move_at_velocity_towards(self, target_x, target_y, speed):
        # Calculate the direction vector towards (0, 0)
        direction = np.array([target_x - self.x, target_y - self.y])

        # Normalize the direction vector to unit length
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance

        # Set velocity with the desired speed towards the target
        self.velocity = direction * speed

        # Apply damping to simulate friction
        self.velocity *= 0.5

        # Update position (p = p + v)
        self.x += dt * self.velocity[0]
        self.y += dt * self.velocity[1]

        # Apply bounds (walls)
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

def find_critical_chain(tx_agent, rx_agent, all_agents):
    global movement
    movement += 1
    if movement % CHAIN_DELAY == 0: ### Remove if necessary
        for an_agent in all_agents:
            an_agent.part_of_critical_chain = False
            if an_agent.is_rx or an_agent.is_tx:
                an_agent.part_of_critical_chain = True

        # Initialize a queue for BFS
        try:
            visited.clear()
            predecessors.clear()
        except:
            pass
        queue = deque([tx_agent])
        visited = {tx_agent: 0}  # Track visited nodes and their hop count
        predecessors = {tx_agent: None}  # To reconstruct the path

        while queue:
            current_agent = queue.popleft()

            if current_agent is rx_agent:
                break  # Stop when we reach the RX

            for other_agent in all_agents:
                if other_agent not in visited and current_agent.distance_to(other_agent) < DIST_DETECTION_RANGE:
                    visited[other_agent] = visited[current_agent] + 1
                    predecessors[other_agent] = current_agent
                    queue.append(other_agent)

        # Reconstruct the shortest path
        critical_chain = []
        current_agent = rx_agent
        while current_agent is not None:
            critical_chain.append(current_agent)
            current_agent = predecessors.get(current_agent)

        # Mark agents in the critical chain
        for critical_agent in critical_chain:
            critical_agent.part_of_critical_chain = True


# Initialize agents, TX, and RX. Agents should not be initialized in the exact same spot
tx = Agent(175, 175)
tx.is_tx = True

rx = Agent(RXSPAWNX, RXSPAWNY)
rx.is_rx = True

def calculate_random_position(rx, max_radius):
    # Random angle between 0 and 2 * pi
    angle = random.uniform(0, 2 * math.pi)

    # Random radius (between 0 and max_radius)
    radius = random.uniform(0, max_radius)

    # Convert polar coordinates to Cartesian (x, y)
    x = rx.x + radius * math.cos(angle)
    y = rx.y + radius * math.sin(angle)

    return x, y

# Function to check if the new agent has sufficient distance from all existing agents
def is_valid_position(new_agent, agents, min_distance):
    if new_agent.distance_to(rx) < min_distance:
        return False
    for agent in agents:
        if new_agent.distance_to(agent) < min_distance:
            return False
    return True

# Main initialization logic
agents = [rx, tx]

density_map = np.zeros((40, 40))  # Initialize the density map

# Loop to place agents
while len(agents) < AGENT_COUNT:
    # calculate random position
    agent_x, agent_y = calculate_random_position(rx, MAX_AGENT_INIT_RADIUS)

    # Create a new agent
    new_agent = Agent(agent_x, agent_y)

    # Check if the new agent is at least MIN_DISTANCE away from all others
    if is_valid_position(new_agent, agents, MIN_AGENT_INIT_DISTANCE):
        agents.append(new_agent)

# Main simulation loop
running = True
clock = pygame.time.Clock()
dt = 0
start_time = time.time()
program_start_time = time.time()

frame_count = 0
data_list = []

def handle_sigint(signal, frame):
    global data_list, pygame
    # Save the data to CSV
    print("Interrupt received, saving data to CSV and quitting...")
    data = pd.DataFrame(data_list)
    data.to_csv('agent_positions.csv', index=False)
    pygame.quit()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_sigint)

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    heatmap_surface = pygame.Surface((WIDTH, HEIGHT))
    heatmap_surface.fill((255, 255, 255))  # Fill with white for the background
    heatmap_surface.set_colorkey((255, 255, 255))  # Make the background transparent

    # Draw density map as a heatmap overlay
    for y in range(density_map.shape[0]):
        for x in range(density_map.shape[1]):
            # Normalize the density for color mapping
            density_value = density_map[y, x]
            color_intensity = min(255, int(density_value * 5))  # Adjust the multiplier for visibility
            pygame.draw.rect(heatmap_surface, (color_intensity, 0, 0), (x * (WIDTH // density_map.shape[1]), y * (HEIGHT // density_map.shape[0]), WIDTH // density_map.shape[1], HEIGHT // density_map.shape[0]))

    # Blit the heatmap surface onto the main screen with alpha
    screen.blit(heatmap_surface, (0, 0))

    # Update and draw agents
    for agent in agents:
        agent.check_part_of_chain(agents)
        if not agent.is_rx and not agent.is_tx:
            agent.move(agents, rx, tx)  # Pass tx
        elif agent.is_tx:
            # Set velocity towards (0, 0) with a desired speed (for example, speed = 2.0)
            tx.move_at_velocity_towards(0, 0, 0.01*MAX_SPEED)

        # Find the critical chain after moving agents
        find_critical_chain(tx, rx, agents)

        # Draw agent
        if agent.part_of_critical_chain:
            color = GREEN  # Critical chain agents are green
        else:
            color = BLUE if not agent.is_rx else YELLOW  # Non-critical chain agents
        pygame.draw.circle(screen, color, (int(agent.x), int(agent.y)), AGENT_RADIUS)
        pygame.draw.circle(screen, WHITE, (int(agent.x), int(agent.y)), COMM_RANGE, width=1)

        # Render hop count
        hop_count_text = font.render(str(agent.hop_count), True, WHITE)
        text_rect = hop_count_text.get_rect(center=(int(agent.x), int(agent.y) - AGENT_RADIUS - 10))
        screen.blit(hop_count_text, text_rect)

    # Draw TX and RX
    pygame.draw.rect(screen, BLUE, (tx.x - 10, tx.y - 10, 20, 20))  # TX as a square
    pygame.draw.circle(screen, YELLOW, (int(rx.x), int(rx.y)), AGENT_RADIUS)
    pygame.draw.circle(screen, BLUE, (int(rx.x), int(rx.y)), COMM_RANGE, width=1)

    # Update display
    pygame.display.flip()

    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # Save every second
        for i, agent in enumerate(agents):
            data_list.append({
                'time': round(time.time()-program_start_time, 2),  # You could also store actual time using time.time()
                'agent_id': i,
                'x': agent.x,
                'y': agent.y,
                'is_tx': agent.is_tx,
                'is_rx': agent.is_rx,
                'dist_detection_range': DIST_DETECTION_RANGE
            })
        start_time = time.time()  # Reset start time

    frame_count += 1

    dt = clock.tick(FPS)

