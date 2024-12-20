import math
import numpy as np

### AGENT PARAMETERS ###
AGENT_RADIUS = 5
AGENT_VIRTUAL_MASS = (
    1  # virtual because it is the 'mass' acted on by repulsion and attraction
)
COMM_RANGE = AGENT_RADIUS * 5
DIST_DETECTION_RANGE = COMM_RANGE * 2
MIN_DISTANCE = COMM_RANGE / 2
MAX_SPEED = 0.01
# REPULSION_CONSTANT = 0.0020  # Constant for repulsive force
# REPULSION_CONSTANT = 0.001  # Constant for repulsive force
# ATTRACTION_CONSTANT = 0.0002  # Constant for attractive force
REPULSION_CONSTANT = 5  # Constant for repulsive force
ATTRACTION_CONSTANT = 1  # Constant for attractive force

# Critical Chain Ruleset Parameter
FOLLOW_DISTANCE = COMM_RANGE / 1.5


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.is_rx = False
        self.is_tx = False
        self.initial_ruleset = None
        self.ruleset = None  # "chain_extension", "network_mass", "critical_chain"
        self.part_of_critical_chain = False
        self.hop_count = -1
        self.neighbors = []  # List of neighbor agents
        self.received_messages = {}  # Stores messages received {source_agent: (cost, path)}

    def position(self):
        return np.array([self.x, self.y])

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    # def compute_repulsive_force(self, distance, max_force_distance, repulsion_constant, epsilon=1e-3):
    #     if distance < max_force_distance:
    #         force = repulsion_constant * (1 / (distance + epsilon) - 1 / max_force_distance)
    #         return max(force, 0)  # Clamp to ensure no negative forces
    #     else:
    #         return 0.0


    def compute_repulsive_force(self, distance, max_force_distance, repulsion_constant, epsilon=1e-3):
        radius = 3* AGENT_RADIUS
        # Effective distance is measured from the robot's radius
        effective_distance = max(distance - radius, epsilon)

        # Compute repulsive force
        if distance < max_force_distance:
            force = repulsion_constant * (1 / effective_distance - 1 / (max_force_distance - radius))
            return max(force, 0)  # Ensure no negative forces
        else:
            return 0.0


    def move(self, dt, critical_chain, all_agents, rx_agent):
        max_force_distance = DIST_DETECTION_RANGE
        net_force = np.array([0.0, 0.0])

        if self.ruleset == "network_mass":
            # Add attraction to critical chain components
            for critical_agent in critical_chain:
                # if not critical_agent.is_rx:
                displacement = critical_agent.position() - self.position()
                distance = np.linalg.norm(displacement)

                if distance < max_force_distance:
                    attraction_force = ATTRACTION_CONSTANT * displacement
                    net_force += attraction_force

            # repulsion from all agents
            for other_agent in all_agents:
                if other_agent is not self:  # Avoid self-interaction
                    # Calculate the displacement vector between agents
                    displacement = other_agent.position() - self.position()
                    distance = np.linalg.norm(displacement)

                    if distance < max_force_distance:
                        direction = displacement / distance

                        # Apply repulsion from all agents
                        # repulsive_force_magnitude = REPULSION_CONSTANT * (
                        #     1 - (distance / max_force_distance)
                        # )

                        repulsive_force_magnitude = self.compute_repulsive_force(distance, max_force_distance, REPULSION_CONSTANT)
                        repulsive_force = -direction * repulsive_force_magnitude
                        net_force += repulsive_force

            # Update the acceleration based on net force (F = ma, with m = 1, so a = F)
            self.acceleration = AGENT_VIRTUAL_MASS * net_force
            magnitude = np.linalg.norm(self.acceleration)
            if magnitude > 0:
                self.acceleration = (self.acceleration / magnitude) * 0.01

            # Update velocity and limit it to a maximum speed
            self.velocity += self.acceleration
            if np.linalg.norm(self.velocity) > MAX_SPEED:
                self.velocity = (
                    self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED
                )

        elif self.ruleset == "critical_chain":
            # find the preceding agent in the critical chain
            target_agent = critical_chain[critical_chain.index(self) - 1]
            displacement = target_agent.position() - self.position()
            distance = np.linalg.norm(displacement)

            # follow the preceding agent
            if distance > FOLLOW_DISTANCE:
                self.velocity = displacement / np.linalg.norm(displacement) * MAX_SPEED
            elif distance < FOLLOW_DISTANCE:
                self.velocity = (
                    displacement / np.linalg.norm(displacement) * (-MAX_SPEED / 2)
                )

        elif self.ruleset == "chain_extension":
            # Add attraction force towards the RX only
            displacement_to_rx = rx_agent.position() - self.position()
            distance_to_rx = np.linalg.norm(displacement_to_rx)

            if distance_to_rx < max_force_distance:
                attraction_force = ATTRACTION_CONSTANT * displacement_to_rx
                net_force += attraction_force

            ### Add attraction to first critical chain agent

            displacement_to_target = critical_chain[-2].position() - self.position()
            distance_to_target = np.linalg.norm(displacement_to_target)

            if distance_to_target < max_force_distance:
                attraction_force = ATTRACTION_CONSTANT * displacement_to_target
                net_force += attraction_force

            # repulsion from all agents
            for other_agent in all_agents:
                # if other_agent is not self and (other_agent.ruleset == "chain_extension" or other_agent.is_rx or other_agent.ruleset == "critical_chain"):
                if other_agent is not self:
                    # Calculate the displacement vector between agents
                    displacement = other_agent.position() - self.position()
                    distance = np.linalg.norm(displacement)

                    if distance < max_force_distance:
                        direction = displacement / distance

                        # Apply repulsion from all agents
                        # repulsive_force_magnitude = REPULSION_CONSTANT * (
                        #     1 - (distance / max_force_distance)
                        # )
                        repulsive_force_magnitude = self.compute_repulsive_force(distance, max_force_distance, REPULSION_CONSTANT)

                        repulsive_force = -direction * repulsive_force_magnitude
                        net_force += repulsive_force

            # Update the acceleration based on net force (F = ma, with m = 1, so a = F)
            self.acceleration = AGENT_VIRTUAL_MASS * net_force
            magnitude = np.linalg.norm(self.acceleration)
            if magnitude > 0:
                self.acceleration = (self.acceleration / magnitude) * 0.01

            # Update velocity and limit it to a maximum speed
            self.velocity += self.acceleration
            if np.linalg.norm(self.velocity) > MAX_SPEED:
                self.velocity = (
                    self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED
                )

        # Update position
        self.x += dt * self.velocity[0]
        self.y += dt * self.velocity[1]

        # # Apply bounds (walls)
        # self.x = max(0, min(WIDTH, self.x))
        # self.y = max(0, min(HEIGHT, self.y))

        # Adjust position to maintain minimum distance from all agents
        # if self.ruleset in ["chain_extension", "network_mass"]:
        #     for other_agent in all_agents:
        #         if other_agent is not self:  # Avoid self-comparison
        #             other_position = np.array([other_agent.x, other_agent.y])
        #             current_position = np.array([self.x, self.y])
        #             vector_to_other = current_position - other_position
        #             distance_to_other = np.linalg.norm(vector_to_other)

        #             if distance_to_other < MIN_DISTANCE:
        #                 # Reposition the agent to maintain the minimum distance
        #                 direction = vector_to_other / (
        #                     distance_to_other + 1e-8
        #                 )  # Avoid division by zero
        #                 clamped_position = other_position + direction * MIN_DISTANCE
        #                 self.x, self.y = clamped_position[0], clamped_position[1]

    def move_at_velocity_towards(self, target_x, target_y, speed, dt):
        direction = np.array([target_x - self.x, target_y - self.y])

        # Normalize the direction vector to unit length
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance

        # Set velocity with the desired speed towards the target
        self.velocity = direction * speed

        # Update position (p = p + v)
        self.x += dt * self.velocity[0]
        self.y += dt * self.velocity[1]

        # # Apply bounds (walls)
        # self.x = max(0, min(WIDTH, self.x))
        # self.y = max(0, min(HEIGHT, self.y))

    def discover_neighbors(self, all_agents):
        """Determine neighbors based on communication range."""
        self.neighbors = []  # List of neighbor agents
        self.received_messages = {}  # Stores messages received {source_agent: (cost, path)}
        if self.part_of_critical_chain:
            self.part_of_critical_chain = False
            self.ruleset = self.initial_ruleset
        self.neighbors = [
            agent
            for agent in all_agents
            if agent is not self and self.distance_to(agent) < COMM_RANGE
        ]

    def flood_message(self, source_agent, cost, path):
        """Flood message to neighbors, accumulating cost."""
        # Check if this message is better than what we've received before
        if (
            source_agent not in self.received_messages
            or cost < self.received_messages[source_agent][0]
        ):
            # Update the best known cost and path for this source
            self.received_messages[source_agent] = (cost, path)

            # Forward the message to neighbors
            for neighbor in self.neighbors:
                if neighbor not in path:  # Avoid loops
                    new_cost = cost + self.distance_to(neighbor)
                    new_path = path + [self]
                    neighbor.flood_message(source_agent, new_cost, new_path)

    def find_critical_path(self, tx_agent, all_agents):
        """Discover neighbors, flood messages, and determine critical path."""

        # If this is the Rx agent, determine the shortest path from Tx
        if self.is_rx:
            if tx_agent in self.received_messages:
                cost, path = self.received_messages[tx_agent]
                for critical_agent in path:
                    critical_agent.part_of_critical_chain = True
                    if not (critical_agent.is_tx or critical_agent.is_rx):
                        critical_agent.ruleset = "critical_chain"
                path.append(self)
                return path, cost
            else:
                print(
                    f"No path found from Tx({tx_agent.position()}) to Rx({self.position()})."
                )
                return [], float("inf")
