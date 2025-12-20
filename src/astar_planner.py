import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import heapq

# Define grid resolution
GRID_SIZE = 40  # 40x40 grid
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2

# Define obstacles
obstacles = [
    Polygon([[0.0, -0.15], [0.5, 0.3], [0.3, 0.8], [-0.2, 0.4]]),
    Polygon([[-1.0, 0.0], [-0.6, 0.4], [-1.2, 0.7], [-1.4, 0.3]])
]

# Define start and goal positions
start = (-1.5, 1.2)
goal = (0.5, -0.25)

# Convert world coordinates to grid
def world_to_grid(x, y):
    x_idx = int((x - X_MIN) / (X_MAX - X_MIN) * (GRID_SIZE - 1))
    y_idx = int((y - Y_MIN) / (Y_MAX - Y_MIN) * (GRID_SIZE - 1))
    return x_idx, y_idx

# Convert grid coordinates back to world
def grid_to_world(x_idx, y_idx):
    x = X_MIN + (X_MAX - X_MIN) * (x_idx / (GRID_SIZE - 1))
    y = Y_MIN + (Y_MAX - Y_MIN) * (y_idx / (GRID_SIZE - 1))
    return x, y

# A* Node
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost

    def __lt__(self, other):
        return self.f < other.f

# Heuristic function (Euclidean distance)
def heuristic(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Check if a point is inside an obstacle
def is_obstacle_free(x, y):
    point = Polygon([(x, y), (x+0.01, y), (x, y+0.01)])  # Small bounding box
    return all(not point.intersects(obstacle) for obstacle in obstacles)

# Generate 8-connected neighbors (Up, Down, Left, Right, Diagonals)
def get_neighbors(node):
    x, y = node.position
    neighbors = [
        (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # Cardinal directions
        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # Diagonals
    ]
    valid_neighbors = [
        (nx, ny) for nx, ny in neighbors
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and is_obstacle_free(*grid_to_world(nx, ny))
    ]
    return valid_neighbors

# A* Algorithm with interactive visualization
def a_star(start, goal):
    """ Implements A* search algorithm with real-time visualization. """
    start_node = Node(world_to_grid(*start))
    goal_node = Node(world_to_grid(*goal))

    open_set = []
    heapq.heappush(open_set, start_node)
    came_from = {}
    g_cost = {start_node.position: 0}
    f_cost = {start_node.position: heuristic(start_node.position, goal_node.position)}

    # Initialize interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw obstacles
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        ax.fill(x, y, alpha=0.5, label="Obstacle")

    ax.scatter(*start, c="green", s=100, label="Start")
    ax.scatter(*goal, c="red", s=100, label="Goal")
    # plt.xlim(X_MIN, X_MAX)
    # plt.ylim(Y_MIN, Y_MAX)
    plt.grid(True)

    while open_set:
        current_node = heapq.heappop(open_set)
        current_pos = current_node.position

        # If goal is reached, reconstruct path
        if current_pos == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path = [grid_to_world(*p) for p in reversed(path)]

            # Plot final path
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], c="blue", linewidth=3, label="Planned Path")
            plt.pause(0.1)
            plt.ioff()
            plt.show()
            return path

        # Expand neighbors
        for neighbor in get_neighbors(current_node):
            tentative_g = g_cost[current_pos] + heuristic(current_pos, neighbor)

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + heuristic(neighbor, goal_node.position)
                neighbor_node = Node(neighbor, parent=current_node, g=tentative_g, h=f_cost[neighbor])

                heapq.heappush(open_set, neighbor_node)
                came_from[neighbor] = current_pos

                # Convert grid coordinates to world for visualization
                world_x, world_y = grid_to_world(*neighbor)
                ax.scatter(world_x, world_y, c="gray", s=5, alpha=0.3)  # Draw explored nodes
                plt.pause(0.001)

    print("No path found!")
    plt.ioff()
    plt.show()
    return None

# Run A* search with interactive visualization
path = a_star(start, goal)
