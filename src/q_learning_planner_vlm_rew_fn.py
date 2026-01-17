"""
Q-learning grid planner with polygon obstacles + visualization
VLM (Qwen-VL) as NUMERICAL reward model (episode-level, cached, fast).

- No loss of original details
- VLM loaded ONCE
- VLM queried ONCE per episode
- Reward cached
- Uses rendered grid image
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import random
from tqdm import trange
from collections import defaultdict
from PIL import Image, ImageDraw

# ============================================================
# 0) VLM CONFIG (FAST + SAFE)
# ============================================================
USE_VLM_REWARD = True
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"   # or local path
VLM_REWARD_CLIP = 50.0

GOAL_TEXT = "the agent reaches the red goal safely without hitting obstacles"
BASELINE_TEXT = "the agent wanders randomly without a goal"

reward_cache = {}

if USE_VLM_REWARD:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch

    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    model = AutoModelForVision2Seq.from_pretrained(
        VLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()


# ============================================================
# 1) ENVIRONMENT (UNCHANGED)
# ============================================================
GRID_SIZE = 50
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2

obstacles = [
    Polygon([[0.0, -0.15], [0.5, 0.3], [0.3, 0.8], [-0.2, 0.4]]),
    Polygon([[-1.0, 0.0], [-0.6, 0.4], [-1.2, 0.7], [-1.4, 0.3]]),
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
    p = Polygon([(x, y), (x + 0.01, y), (x, y + 0.01)])
    return all(not p.intersects(o) for o in obstacles)


ACTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]


def valid_actions(state):
    x, y = state
    out = []
    for dx, dy in ACTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            wx, wy = grid_to_world(nx, ny)
            if is_obstacle_free(wx, wy):
                out.append((dx, dy))
    return out


# ============================================================
# 2) GRID RENDERING (FOR VLM)
# ============================================================
def render_grid_image(agent_state):
    img_size = 256
    cell = img_size // GRID_SIZE

    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # obstacles
    for o in obstacles:
        poly = [world_to_grid(x, y) for x, y in o.exterior.coords]
        poly_px = [(p[0]*cell, p[1]*cell) for p in poly]
        draw.polygon(poly_px, fill=(160, 160, 160))

    # goal (red)
    gx, gy = goal_grid
    draw.rectangle(
        [gx*cell, gy*cell, (gx+1)*cell, (gy+1)*cell],
        fill=(255, 0, 0)
    )

    # agent (blue)
    ax, ay = agent_state
    draw.rectangle(
        [ax*cell, ay*cell, (ax+1)*cell, (ay+1)*cell],
        fill=(0, 0, 255)
    )

    return img


# ============================================================
# 3) VLM EPISODE REWARD (NUMERICAL, CACHED) - FIXED
# ============================================================
@torch.no_grad()
def vlm_episode_reward(final_state):
    img = render_grid_image(final_state)
    key = hash(img.tobytes())

    if key in reward_cache:
        return reward_cache[key]

    # Qwen2.5-VL expects messages format with image content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Score how well this image shows the agent reaching the red goal safely. Output only a single real number. Higher is better."}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
    )

    decoded = processor.tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).strip()

    # Extract the last token as the numerical score
    token = decoded.split()[-1]

    try:
        reward = float(token)
    except ValueError:
        reward = 0.0

    reward = float(np.clip(reward, -VLM_REWARD_CLIP, VLM_REWARD_CLIP))
    reward_cache[key] = reward
    return reward


# ============================================================
# 4) Q-LEARNING SETUP (UNCHANGED)
# ============================================================
alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 3000
max_steps = 200

Q = {}


def get_Q(s, a):
    return Q.get(s, {}).get(a, 0.0)


def set_Q(s, a, v):
    Q.setdefault(s, {})[a] = v


start_grid = world_to_grid(*start)
goal_grid = world_to_grid(*goal)


def dist_to_goal(s):
    return np.linalg.norm(np.array(s) - goal_grid)


# ============================================================
# 5) Q-LEARNING LOOP (EPISODE-LEVEL VLM REWARD)
# ============================================================
pbar = trange(episodes, desc="Q-learning")

for ep in pbar:
    s = start_grid
    visit_count = defaultdict(int)
    transitions = []

    start_dist = dist_to_goal(s)

    for step in range(max_steps):
        if s == goal_grid:
            break

        actions = valid_actions(s)
        if not actions:
            break

        if random.random() < epsilon:
            a = random.choice(actions)
        else:
            a = max(actions, key=lambda x: get_Q(s, x))

        s_next = (s[0] + a[0], s[1] + a[1])
        visit_count[s_next] += 1

        # minimal geometric shaping (kept)
        reward = -dist_to_goal(s_next)

        transitions.append((s, a, reward, s_next))
        s = s_next

    # ---------------- EPISODE VLM REWARD ----------------
    if USE_VLM_REWARD:
        R_vlm = vlm_episode_reward(s)
    else:
        R_vlm = 0.0

    # Monte-Carlo-style update (stable)
    for (s, a, r, s_next) in transitions:
        target = r + R_vlm
        set_Q(s, a, get_Q(s, a) + alpha * (target - get_Q(s, a)))

    if ep % 50 == 0:
        pbar.set_postfix(VLM_R=f"{R_vlm:.2f}", cache=len(reward_cache))


# ============================================================
# 6) EXTRACT GREEDY PATH (UNCHANGED)
# ============================================================
path = [start_grid]
s = start_grid

for _ in range(max_steps):
    if s == goal_grid:
        break
    a = max(valid_actions(s), key=lambda x: get_Q(s, x))
    s = (s[0] + a[0], s[1] + a[1])
    path.append(s)


# ============================================================
# 7) VISUALIZATION (UNCHANGED)
# ============================================================
plt.figure(figsize=(6, 6))
ax = plt.gca()

for o in obstacles:
    x, y = o.exterior.xy
    ax.fill(x, y, alpha=0.5)

ax.scatter(*start, c="green", s=100)
ax.scatter(*goal, c="red", s=100)

path_world = np.array([grid_to_world(*p) for p in path])
ax.plot(path_world[:, 0], path_world[:, 1], c="blue", linewidth=3)

plt.grid(True)
plt.show()