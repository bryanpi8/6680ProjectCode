
import pygame
import random
import math
from collections import deque

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 200, 200
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
AGENT_COUNT = 20
AGENT_RADIUS = 5
COMM_RANGE = AGENT_RADIUS * 4
TX_MOVE_DISTANCE = 4
SPEED = 2

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_repeater = False
        self.target = None  # Target to follow (TX or another repeater)
        self.connected_to_tx = False  # Tracks if the agent is connected to the TX
    
    def move(self, agents):
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

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

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



# Transmitter (TX) class
class Transmitter:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
    
    def move(self):
        self.x += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
        self.y += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

# Initialize agents, TX, and RX
agents = [Agent(WIDTH // 2, HEIGHT // 2) for _ in range(AGENT_COUNT)]
tx = Transmitter()
rx = Agent(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
agents = [Agent(rx.x, rx.y) for _ in range(AGENT_COUNT)]  # Spawn all agents at RX

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
    tx.move()
    
    # Update and draw agents
    for agent in agents:
        agent.move(agents)  # Pass in the list of agents to check for neighbors
        
        # Check if agent is near TX or any repeater
        if agent.distance_to(tx) <= COMM_RANGE:
            agent.is_repeater = True
            agent.target = tx
            agent.connected_to_tx = True  # Directly connected to TX
        else:
            # Check if agent is near any other repeater
            for other_agent in agents:
                if other_agent.is_repeater and agent.distance_to(other_agent) <= COMM_RANGE:
                    agent.is_repeater = True
                    agent.target = other_agent
                    agent.connected_to_tx = other_agent.connected_to_tx
                    break  # Stop checking once we find a repeater nearby
        
        # If the agent is a repeater but no longer connected to TX, return to search mode and propagate disconnection
        if agent.is_repeater and not agent.connected_to_tx:
            propagate_disconnection(agent)
        
        # Draw agent
        color = GREEN if agent.is_repeater else BLUE
        pygame.draw.circle(screen, color, (int(agent.x), int(agent.y)), AGENT_RADIUS)



    
    # Draw TX and RX
    pygame.draw.rect(screen, RED, (tx.x - 10, tx.y - 10, 20, 20))  # TX as a square
    pygame.draw.circle(screen, YELLOW, (int(rx.x), int(rx.y)), AGENT_RADIUS + 5)  # RX as a larger circle
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
