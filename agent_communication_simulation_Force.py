
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

font = pygame.font.SysFont(None, 12)  # You can change the size as needed

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Simulation parameters
FPS = 100
AGENT_COUNT = 25
AGENT_RADIUS = 5
COMM_RANGE = AGENT_RADIUS * 5
DIST_DETECTION_RANGE = COMM_RANGE
TX_MOVE_DISTANCE = 4
SPEED = 2
BLEND_FACTOR = 0.3

RXSPAWNX = WIDTH/2
RXSPAWNY = HEIGHT/2

MAX_AGENT_INIT_RADIUS = AGENT_RADIUS * 15  # Define a maximum radius for placement around rx
MIN_AGENT_INIT_DISTANCE = AGENT_RADIUS * 4

REPULSION_CONSTANT = 100.0   # Strength of the repulsive force
ATTRACTION_CONSTANT = 90.0  # Strength of the attractive force

randomMovementChance = .01

tx_discovered = False  # Initially, the TX is not discovered

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = np.array([0.0, 0.0])  # Initial velocity
        self.acceleration = np.array([0.0, 0.0])  # Initial acceleration
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * SPEED
        self.is_rx = False # is rx (is the start point)
        self.target = None  # Target to follow (TX or another repeater)
        self.connected_to_tx = False  # Tracks if the agent is connected to the TX
        self.movement_vector = SPEED * np.array([math.cos(random.uniform(0, 2 * math.pi)), math.sin(random.uniform(0, 2 * math.pi))])
        self.part_of_chain = False  # Whether this agent is part of a chain
        self.tx_discovered = False  # Whether this agent has discovered the TX
        self.hop_count = -1  # Initialize hop count to infinity
        self.angle_offset = 0  # Angle offset for circular movement

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
                self.target = other_agent
                return
            
    def calculate_attraction_to_tx(self, tx):
        if self.tx_discovered:
            distance_to_tx = self.distance_to(tx)
            
            if distance_to_tx > AGENT_RADIUS:  # Only apply force if not too close
                tx_attraction_strength = ATTRACTION_CONSTANT * 10  # Stronger attraction after discovery
                force_magnitude = tx_attraction_strength / (distance_to_tx ** 2)  # Inverse square law
                force_direction = np.array([tx.x - self.x, tx.y - self.y]) / distance_to_tx
                return force_magnitude * force_direction
        return np.array([0.0, 0.0])  # No attraction if TX not discovered



    def calculate_chain_forces(self, other_agent):
        distance_to_other = self.distance_to(other_agent)

        # Balanced attraction and repulsion between chain neighbors
        if other_agent.part_of_chain and self.part_of_chain:
            if distance_to_other > AGENT_RADIUS:
                chain_force_strength = ATTRACTION_CONSTANT * 5  # Stronger force within chain
                force_magnitude = chain_force_strength / (distance_to_other ** 2)
                force_direction = (other_agent.position() - self.position()) / distance_to_other
                return force_magnitude * force_direction
        
        # Apply repulsion even if not part of the chain to spread agents out
        if distance_to_other < COMM_RANGE and distance_to_other > 0:
            repulsion_strength = REPULSION_CONSTANT / (distance_to_other ** 2)  # Inverse square law
            force_direction = (self.position() - other_agent.position()) / distance_to_other
            return repulsion_strength * force_direction
        
        # If they are not in the chain and not close enough to repel, return no force
        return np.array([0.0, 0.0])



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
        net_force = np.array([0.0, 0.0])  # Initialize net force
        
        # Sum the repulsive forces from all neighbors
        for other_agent in all_agents:
            if other_agent != self and self.distance_to(other_agent) < DIST_DETECTION_RANGE:
                repulsive_force = self.calculate_repulsive_direction(other_agent)
                attractive_force = self.calculate_attractive_force(other_agent)
                net_force += repulsive_force + attractive_force
        
        # Add wall repulsion to net force
        wall_repulsion = self.calculate_wall_repulsion_direction()
        net_force += wall_repulsion
        
        return net_force

    def move(self, all_agents, tx):
        # Calculate the net force from neighbors
        net_force = self.calculate_net_direction(all_agents)
        
        # Apply stronger attraction to TX if part of the chain or TX discovered
        if self.tx_discovered or self.part_of_chain:
            tx_attraction_force = self.calculate_attraction_to_tx(tx)
            net_force += tx_attraction_force  # Always attract to the TX
        
        # Apply forces between chain members or repulsion between agents
        for other_agent in all_agents:
            if other_agent != self:
                force = self.calculate_chain_forces(other_agent)
                net_force += force  # Add chain or repulsion forces to the net force

        # Occasionally apply a random perturbation for non-chain agents
        if not self.part_of_chain and random.random() < randomMovementChance:
            random_perturbation = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) * SPEED * 0.5
            net_force += random_perturbation  # Add the random force to the net force
        
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
        max_speed = SPEED if not self.part_of_chain else SPEED * 0.2  # Slow down when in chain
        if np.linalg.norm(self.velocity) > max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * max_speed

        # Apply damping to simulate friction
        self.velocity *= 0.9

        # Update position (p = p + v)
        self.x += self.velocity[0]
        self.y += self.velocity[1]

        # Apply bounds (walls)
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))


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
agents = [rx]

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

# Main simulation loop
running = True
clock = pygame.time.Clock()

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
        if not agent.is_rx:
            agent.move(agents, tx)  # Pass tx

        # Check if TX is discovered by this agent
        if agent.distance_to(tx) < COMM_RANGE:
            agent.tx_discovered = True  # Mark this agent as having discovered the TX

        # Draw agent
        color = GREEN if agent.is_rx else BLUE
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
    clock.tick(FPS)

# Quit Pygame
pygame.quit()

