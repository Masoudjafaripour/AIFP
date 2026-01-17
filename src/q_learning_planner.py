import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import random
import math
from tqdm import trange   # <-- added

# -------------------------
# Environment (unchanged)
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
                out.append((dx, dy))
    return out

def dist_to_goal(s):
    return np.linalg.norm(np.array(s) - goal_grid)

# -------------------------
# Q-learning
# -------------------------
alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 3000
max_steps = 200

Q = {}  # Q[(x,y)][(dx,dy)]

def get_Q(s, a):
    return Q.get(s, {}).get(a, 0.0)

def set_Q(s, a, val):
    Q.setdefault(s, {})[a] = val

start_grid = world_to_grid(*start)
goal_grid = world_to_grid(*goal)

pbar = trange(episodes, desc="Q-learning")  # <-- added
for ep in pbar:                             # <-- changed (range → pbar)
    s = start_grid
    total_reward = 0                        # <-- added (for tqdm display)

    for _ in range(max_steps):
        if s == goal_grid:
            break

        actions = valid_actions(s)
        if not actions:
            break

        if random.random() < epsilon:
            a = random.choice(actions)
        else:
            a = max(actions, key=lambda x: get_Q(s, x))

        nx, ny = s[0] + a[0], s[1] + a[1]
        s_next = (nx, ny)

        reward = 100 if s_next == goal_grid else -dist_to_goal(s_next)
        total_reward += reward              # <-- added

        best_next = max(
            [get_Q(s_next, ap) for ap in valid_actions(s_next)],
            default=0
        )
        td_target = reward + gamma * best_next
        td_error = td_target - get_Q(s, a)

        set_Q(s, a, get_Q(s, a) + alpha * td_error)
        s = s_next

    if ep % 50 == 0:                        # <-- added
        pbar.set_postfix(R=f"{total_reward:.1f}")

# -------------------------
# Extract greedy path
# -------------------------
path = [start_grid]
s = start_grid
for _ in range(max_steps):
    if s == goal_grid:
        break
    actions = valid_actions(s)
    if not actions:
        break
    a = max(actions, key=lambda x: get_Q(s, x))
    s = (s[0] + a[0], s[1] + a[1])
    path.append(s)

# -------------------------
# Visualization (unchanged)
# -------------------------
plt.figure(figsize=(6,6))
ax = plt.gca()

for o in obstacles:
    x, y = o.exterior.xy
    ax.fill(x, y, alpha=0.5)

ax.scatter(*start, c="green", s=100)
ax.scatter(*goal, c="red", s=100)

path_world = np.array([grid_to_world(*p) for p in path])
ax.plot(path_world[:,0], path_world[:,1], c="blue", linewidth=3)

plt.grid(True)
plt.show()
