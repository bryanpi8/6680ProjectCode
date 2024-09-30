
import pygame
import random
import math

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
COMM_RANGE = AGENT_RADIUS * 4.5
TX_MOVE_DISTANCE = 2
SPEED = 2

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_repeater = False
        self.neighbors = []
        self.target = None  # Target to follow (another agent or TX)
    
    def move(self):
        if self.is_repeater and self.target:  # Follow the target
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance > 0:
                self.x += (dx / distance) * SPEED
                self.y += (dy / distance) * SPEED
        else:
            # Move randomly if not a repeater and not too far from neighbors
            if len(self.neighbors) > 0:
                closest_neighbor = min(self.neighbors, key=lambda n: self.distance_to(n))
                if self.distance_to(closest_neighbor) < COMM_RANGE:
                    self.x += random.uniform(-SPEED, SPEED)
                    self.y += random.uniform(-SPEED, SPEED)
                    self.x = max(0, min(WIDTH, self.x))
                    self.y = max(0, min(HEIGHT, self.y))
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

# Transmitter (TX) class
class Transmitter:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.direction = random.choice(["horizontal", "vertical"])
    
    def move(self):
        if self.direction == "horizontal":
            self.x += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
            self.x = max(0, min(WIDTH, self.x))
        else:
            self.y += random.uniform(-TX_MOVE_DISTANCE, TX_MOVE_DISTANCE)
            self.y = max(0, min(HEIGHT, self.y))

# Initialize agents, TX, and RX
agents = [Agent(WIDTH // 2, HEIGHT // 2) for _ in range(AGENT_COUNT)]
tx = Transmitter()
rx = Agent(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))

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
        agent.neighbors = [other for other in agents if agent.distance_to(other) <= COMM_RANGE]
        
        # Check if agent is near a repeater or TX and becomes a repeater
        if not agent.is_repeater:
            if agent.distance_to(tx) <= COMM_RANGE or any(neighbor.is_repeater for neighbor in agent.neighbors):
                agent.is_repeater = True
                agent.target = tx if agent.distance_to(tx) <= COMM_RANGE else min(agent.neighbors, key=lambda n: agent.distance_to(n) if n.is_repeater else float('inf'))
        
        # Move agent
        agent.move()
        
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
