import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import random
import math

# -------------------------
# Environment (same as yours)
# -------------------------
GRID_SIZE = 50
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2

obstacles = [
    Polygon([[0.0, -0.15], [0.5, 0.3], [0.3, 0.8], [-0.2, 0.4]]),
    Polygon([[-1.0, 0.0], [-0.6, 0.4], [-1.2, 0.7], [-1.4, 0.3]])
]

start = (-1.5, 1.2)
goal = (0.5, -0.25)

def world_to_grid(x, y):
    xi = int((x - X_MIN) / (X_MAX - X_MIN) * (GRID_SIZE - 1))
    yi = int((y - Y_MIN) / (Y_MAX - Y_MIN) * (GRID_SIZE - 1))
    return xi, yi

def grid_to_world(xi, yi):
    x = X_MIN + (X_MAX - X_MIN) * xi / (GRID_SIZE - 1)
    y = Y_MIN + (Y_MAX - Y_MIN) * yi / (GRID_SIZE - 1)
    return x, y

def is_obstacle_free(x, y):
    p = Polygon([(x, y), (x+0.01, y), (x, y+0.01)])
    return all(not p.intersects(o) for o in obstacles)

ACTIONS = [
    (-1,0),(1,0),(0,-1),(0,1),
    (-1,-1),(-1,1),(1,-1),(1,1)
]

def valid_actions(state):
    x, y = state
    out = []
    for dx, dy in ACTIONS:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            wx, wy = grid_to_world(nx, ny)
            if is_obstacle_free(wx, wy):
                out.append((nx, ny))
    return out

def dist_to_goal(s):
    return np.linalg.norm(np.array(s) - goal_grid)

# -------------------------
# MCTS Node
# -------------------------
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.Q = 0.0

    def uct(self, c=1.4):
        if self.N == 0:
            return float("inf")
        return self.Q / self.N + c * math.sqrt(math.log(self.parent.N + 1) / self.N)

# -------------------------
# MCTS algorithm
# -------------------------
def mcts(start_state, iterations=4000, rollout_depth=30):
    root = MCTSNode(start_state)

    for _ in range(iterations):
        node = root

        # Selection
        while node.children:
            node = max(node.children.values(), key=lambda n: n.uct())

        # Expansion
        actions = valid_actions(node.state)
        if actions:
            for a in actions:
                if a not in node.children:
                    node.children[a] = MCTSNode(a, node)
            node = random.choice(list(node.children.values()))

        # Rollout
        s = node.state
        total_reward = 0
        for t in range(rollout_depth):
            if s == goal_grid:
                total_reward += 100
                break
            acts = valid_actions(s)
            if not acts:
                break
            s = random.choice(acts)
            total_reward -= dist_to_goal(s)

        # Backprop
        while node:
            node.N += 1
            node.Q += total_reward
            node = node.parent

    # Extract greedy path
    path = [start_state]
    node = root
    while node.children:
        node = max(node.children.values(), key=lambda n: n.N)
        path.append(node.state)
        if node.state == goal_grid:
            break

    return path

# -------------------------
# Visualization
# -------------------------
start_grid = world_to_grid(*start)
goal_grid = world_to_grid(*goal)

plt.figure(figsize=(6,6))
ax = plt.gca()

for o in obstacles:
    x, y = o.exterior.xy
    ax.fill(x, y, alpha=0.5)

ax.scatter(*start, c="green", s=100)
ax.scatter(*goal, c="red", s=100)

path = mcts(start_grid)

path_world = np.array([grid_to_world(*p) for p in path])
ax.plot(path_world[:,0], path_world[:,1], c="blue", linewidth=3)

plt.grid(True)
plt.show()
