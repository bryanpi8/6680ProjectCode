
import pygame
import random
import math
from collections import deque
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Search and Communication Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Simulation parameters
FPS = 30
AGENT_COUNT = 20
AGENT_RADIUS = 5
COMM_RANGE = AGENT_RADIUS * 4
DIST_DETECTION_RANGE = COMM_RANGE
TX_MOVE_DISTANCE = 4
SPEED = 2
BLEND_FACTOR = 0.3

RXSPAWNX = WIDTH/2
RXSPAWNY = HEIGHT/2

MAX_AGENT_INIT_RADIUS = AGENT_RADIUS * 15  # Define a maximum radius for placement around rx
MIN_AGENT_INIT_DISTANCE = AGENT_RADIUS * 4

REPULSION_CONSTANT = 100.0   # Strength of the repulsive force
ATTRACTION_CONSTANT = 100.0  # Strength of the attractive force

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_repeater = False
        self.target = None  # Target to follow (TX or another repeater)
        self.connected_to_tx = False  # Tracks if the agent is connected to the TX
        self.movement_vector = SPEED * np.array([math.cos(random.uniform(0, 2 * math.pi)), math.sin(random.uniform(0, 2 * math.pi))])
        self.part_of_chain = False

    def position(self):
        return np.array([self.x, self.y])

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def check_part_of_chain(self, all_agents):
        global tx
        if self.distance_to(tx) < DIST_DETECTION_RANGE:
            self.part_of_chain = True
            return

        for other_agent in all_agents:
            if other_agent.part_of_chain and self.distance_to(other_agent) < DIST_DETECTION_RANGE:
                self.part_of_chain = True

    def calculate_repulsive_direction(self, other):
        distance = self.distance_to(other)

        if distance < DIST_DETECTION_RANGE:
            # Repulsive force follows inverse square law
            # Force direction is away from the neighbor
            force_direction = (self.position() - other.position())
            return force_direction
        else:
            # If robots are too close, apply a strong repulsive force to avoid collision
            return np.array([0.0, 0.0])

    def calculate_attractive_force(self, other):
        distance = self.distance_to(other)

        if distance > AGENT_RADIUS:
            # Attractive force follows inverse square law
            force_magnitude = ATTRACTION_CONSTANT / (distance ** 2)
            # Force direction is towards the neighbor
            force_direction = (self.position() + other.position()) / distance
            return force_magnitude * force_direction
        else:
            return np.array([0.0, 0.0])

    def calculate_wall_repulsion_direction(self):
        net_direction = np.array([0.0, 0.0])  # Initialize net force to zero
        # Wall boundaries (left, right, top, bottom)
        walls = {
            "left": 0,
            "right": WIDTH,
            "top": 0,
            "bottom": HEIGHT
        }

        # Check distance to each wall and calculate repulsion
        distance_to_walls = {
            "left": self.x,
            "right": WIDTH - self.x,
            "top": self.y,
            "bottom": HEIGHT - self.y
        }

        for wall, distance in distance_to_walls.items():
            if distance < DIST_DETECTION_RANGE:  # Threshold for repulsion
                # Calculate repulsive force away from the wall
                if wall == "left":
                    net_direction[0] += 1  # Push to the right
                elif wall == "right":
                    net_direction[0] -= 1  # Push to the left
                elif wall == "top":
                    net_direction[1] += 1  # Push down
                elif wall == "bottom":
                    net_direction[1] -= 1  # Push up

        return net_direction

    def calculate_net_direction(self, all_agents):
        net_direction = np.array([0.0, 0.0])  # Initialize net force to zero

        # Sum the repulsive forces from all neighbors
        for other_agent in all_agents:
            if other_agent != self and self.distance_to(other_agent) < DIST_DETECTION_RANGE:  # Only consider it if within range
                repulsive_direction = self.calculate_repulsive_direction(other_agent)
                # attractive_force = self.calculate_attractive_force(other_agent)
                net_direction += repulsive_direction #+ attractive_force


        # Add wall repulsion to net force
        wall_direction = self.calculate_wall_repulsion_direction()
        net_direction += wall_direction

        return net_direction

    def move(self, all_agents):
        # Calculate the net force from neighbors
        net_direction = self.calculate_net_direction(all_agents)

        if self.part_of_chain:
            self.movement_vector = np.array([0, 0])
        # Normalize the net force to ensure uniform speed
        elif np.linalg.norm(net_direction) > 0:
            # Normalize the net force to determine the primary movement direction
            normalized_net_direction = net_direction / np.linalg.norm(net_direction)

            # Decompose the movement vector into parallel and perpendicular components
            current_movement_norm = np.dot(self.movement_vector, normalized_net_direction) * normalized_net_direction
            perpendicular_movement = self.movement_vector - current_movement_norm

            # Blend only the perpendicular component
            blended_vector = (perpendicular_movement / np.linalg.norm(net_direction)) +  normalized_net_direction

            # Normalize the result and scale by speed
            self.movement_vector = blended_vector / np.linalg.norm(blended_vector) * SPEED
        else:
            # If no net force, continue in the previous direction
            pass

        # Update the agent's position based on the movement vector
        self.x += self.movement_vector[0]
        self.y += self.movement_vector[1]

    def random_move(self):
        if self.is_repeater and self.target:
            # Follow the target (TX or another repeater)
            direction_x = self.target.x - self.x
            direction_y = self.target.y - self.y
            distance = math.sqrt(direction_x**2 + direction_y**2)

            if distance > 0:
                # Normalize the direction vector and move towards the target
                self.x += SPEED * (direction_x / distance)
                self.y += SPEED * (direction_y / distance)

            # Apply bounds
            self.x = max(0, min(WIDTH, self.x))
            self.y = max(0, min(HEIGHT, self.y))
        else:
            # Move randomly (for non-repeaters)
            self.x += random.uniform(-SPEED, SPEED)
            self.y += random.uniform(-SPEED, SPEED)
            self.x = max(0, min(WIDTH, self.x))  # Keep agent within bounds
            self.y = max(0, min(HEIGHT, self.y))  # Keep agent within bounds

# Transmitter (TX) class
class Transmitter:
    def __init__(self):
        self.x = 50
        self.y = 50

    def random_move(self):
        self.x += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
        self.y += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

def propagate_disconnection(starting_agent):
    """ Propagate disconnection status throughout the network if an agent loses connection to the TX. """
    queue = deque([starting_agent])
    while queue:
        agent = queue.popleft()
        agent.is_repeater = False
        agent.target = None
        agent.connected_to_tx = False

        # Check neighbors and propagate disconnection
        for other_agent in agents:
            if other_agent.target == agent:  # Only check agents directly connected to this one
                queue.append(other_agent)

# Initialize agents, TX, and RX. Agents should not be initialized in the exact same spot
tx = Transmitter()
rx = Agent(RXSPAWNX, RXSPAWNY)

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
agents = []


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

while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move TX
    # tx.move()

    # Update and draw agents
    for agent in agents:
        agent.check_part_of_chain(agents)
        agent.move(agents)
        # # Check if agent is near TX or any repeater
        # if agent.distance_to(tx) <= COMM_RANGE:
        #     agent.is_repeater = True
        #     agent.target = tx
        #     agent.connected_to_tx = True  # Directly connected to TX
        # else:
        #     # Check if agent is near any other repeater
        #     for other_agent in agents:
        #         if other_agent.is_repeater and agent.distance_to(other_agent) <= COMM_RANGE:
        #             agent.is_repeater = True
        #             agent.target = other_agent
        #             agent.connected_to_tx = other_agent.connected_to_tx
        #             break  # Stop checking once we find a repeater nearby

        # # If the agent is a repeater but no longer connected to TX, return to search mode and propagate disconnection
        # if agent.is_repeater and not agent.connected_to_tx:
        #     propagate_disconnection(agent)

        # Draw agent
        color = GREEN if agent.is_repeater else BLUE
        pygame.draw.circle(screen, color, (int(agent.x), int(agent.y)), AGENT_RADIUS)
        pygame.draw.circle(screen, BLACK, (int(agent.x), int(agent.y)), COMM_RANGE, width=1)

    # Draw TX and RX
    pygame.draw.rect(screen, RED, (tx.x - 10, tx.y - 10, 20, 20))  # TX as a square
    pygame.draw.circle(screen, YELLOW, (int(rx.x), int(rx.y)), AGENT_RADIUS)
    pygame.draw.circle(screen, BLUE, (int(rx.x), int(rx.y)), COMM_RANGE, width=1)

    # Update display
    pygame.display.flip()
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
