
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import random

# Define obstacles
obstacles = [
    Polygon([[0.0, -0.15], [0.5, 0.3], [0.3, 0.8], [-0.2, 0.4]]),  # Obstacle 1
    Polygon([[-1.0, 0.0], [-0.6, 0.4], [-1.2, 0.7], [-1.4, 0.3]])  # Obstacle 2
]

# Define start and goal positions
start = (-1.5, 1.2)
goal = (0.5, -0.25)

# RRT Parameters
step_size = 0.1  # Distance to move in each step
max_iters = 1000  # Maximum number of iterations
goal_threshold = 0.05  # Distance at which we consider the goal reached
plot_interval = 50  # How often to update the interactive plot

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_collision_free(p1, p2, obstacles):
    """ Checks if the line between p1 and p2 collides with any obstacles. """
    line = LineString([p1, p2])
    return all(not line.intersects(obstacle) for obstacle in obstacles)

def generate_random_point():
    """ Generates a random point within a specified range. """
    return (random.uniform(-2, 2), random.uniform(-2, 2))

def nearest_node(nodes, random_point):
    """ Finds the nearest node in the tree to a given random point. """
    return min(nodes, key=lambda node: euclidean_distance(node.position, random_point))

def steer(from_node, to_point, step_size):
    """ Moves from 'from_node' towards 'to_point' with a fixed step size. """
    direction = np.array(to_point) - np.array(from_node.position)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return from_node.position  # Avoid division by zero
    direction = (direction / norm) * step_size
    new_position = np.array(from_node.position) + direction
    return tuple(new_position)

def extract_path(node):
    """ Extracts the path from goal to start by following parent nodes. """
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Reverse to get path from start to goal

def initialize_plot():
    """ Initializes the interactive plot. """
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot obstacles
    for i, obstacle in enumerate(obstacles):
        x, y = obstacle.exterior.xy
        ax.fill(x, y, alpha=0.5, label=f"Obstacle {i + 1}")

    # Plot start and goal
    ax.scatter(*start, c="green", s=100, label="Start")
    ax.scatter(*goal, c="red", s=100, label="Goal")

    # Set grid and axis limits
    ax.set_xticks(np.arange(-2.5, 2.6, 0.5))
    ax.set_yticks(np.arange(-2.5, 2.6, 0.5))
    # ax.set_xlim(-2.5, 2.5)
    # ax.set_ylim(-2.5, 2.5)
    ax.legend()
    plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)

    return fig, ax

def rrt_interactive(start, goal, obstacles, step_size, max_iters, goal_threshold):
    """ Implements RRT with interactive visualization. """
    nodes = [Node(start)]
    fig, ax = initialize_plot()

    for i in range(max_iters):
        random_point = generate_random_point() if random.random() > 0.1 else goal  # 10% chance to bias towards goal
        nearest = nearest_node(nodes, random_point)
        new_point = steer(nearest, random_point, step_size)

        if is_collision_free(nearest.position, new_point, obstacles):
            new_node = Node(new_point, parent=nearest)
            nodes.append(new_node)

            # Draw tree edges dynamically
            ax.plot([nearest.position[0], new_node.position[0]], 
                    [nearest.position[1], new_node.position[1]], 
                    c="gray", alpha=0.5, linewidth=1)

            # Update plot every few iterations
            if i % plot_interval == 0:
                plt.pause(1)

            # If close to the goal, extract and draw the path
            if euclidean_distance(new_point, goal) < goal_threshold:
                path = extract_path(new_node)
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], c="blue", linewidth=3, label="Planned Path")
                plt.pause(0.01)
                plt.ioff()  # Turn off interactive mode after completion
                plt.show()
                return path

    print("No path found!")
    plt.ioff()
    plt.show()
    return None

# Run RRT with interactive visualization
path = rrt_interactive(start, goal, obstacles, step_size, max_iters, goal_threshold)


