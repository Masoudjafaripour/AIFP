# """
# Q-learning grid planner with polygon obstacles + visualization
# LLM (Qwen-3B) acts as a NUMERICAL reward function (proxy reward model).

# - LLM returns a single scalar reward value
# - Reward is applied at CHUNK-LEVEL for stability
# - No loss of environment, RL loop, or visualization
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon
# import random
# from tqdm import trange
# from collections import defaultdict

# # ============================================================
# # 0) LLM CONFIG
# # ============================================================
# USE_LLM_REWARD = True
# QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # or local path
# CHUNK_SIZE = 10          # how often LLM reward is queried
# LLM_REWARD_SCALE = 0.2   # how strongly LLM reward influences learning
# LLM_REWARD_CLIP = 20.0   # clamp reward to avoid instability

# LLM_PROMPT_TEMPLATE = """You are a reward function for reinforcement learning.

# Output a single real number reward.
# Higher is better.

# Guidelines:
# - Reaching the goal is very good
# - Decreasing distance to goal is good
# - Loops and revisits are bad
# - Wandering is bad
# - Do NOT explain
# - Output ONLY a number

# Trajectory summary:
# {summary}
# """


# """
# Calls Qwen and returns a scalar reward.
# """
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     QWEN_MODEL,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# def llm_numeric_reward(summary: str) -> float:


#     prompt = LLM_PROMPT_TEMPLATE.format(summary=summary)

#     messages = [
#         {"role": "system", "content": "Output only a number."},
#         {"role": "user", "content": prompt},
#     ]

#     if hasattr(tokenizer, "apply_chat_template"):
#         text = tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#     else:
#         text = "SYSTEM: Output only a number.\nUSER: " + prompt + "\nASSISTANT:"

#     inputs = tokenizer(text, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=10,
#             temperature=0.3,
#             do_sample=False,
#         )

#     raw = tokenizer.decode(output[0], skip_special_tokens=True)
#     raw = raw.strip().split()[-1]

#     try:
#         r = float(raw)
#     except ValueError:
#         r = 0.0

#     return float(np.clip(r, -LLM_REWARD_CLIP, LLM_REWARD_CLIP))


# # ============================================================
# # 1) ENVIRONMENT (UNCHANGED)
# # ============================================================
# GRID_SIZE = 50
# X_MIN, X_MAX = -2, 2
# Y_MIN, Y_MAX = -2, 2

# obstacles = [
#     Polygon([[0.0, -0.15], [0.5, 0.3], [0.3, 0.8], [-0.2, 0.4]]),
#     Polygon([[-1.0, 0.0], [-0.6, 0.4], [-1.2, 0.7], [-1.4, 0.3]]),
# ]

# start = (-1.5, 1.2)
# goal = (0.5, -0.25)


# def world_to_grid(x, y):
#     xi = int((x - X_MIN) / (X_MAX - X_MIN) * (GRID_SIZE - 1))
#     yi = int((y - Y_MIN) / (Y_MAX - Y_MIN) * (GRID_SIZE - 1))
#     return xi, yi


# def grid_to_world(xi, yi):
#     x = X_MIN + (X_MAX - X_MIN) * xi / (GRID_SIZE - 1)
#     y = Y_MIN + (Y_MAX - Y_MIN) * yi / (GRID_SIZE - 1)
#     return x, y


# def is_obstacle_free(x, y):
#     p = Polygon([(x, y), (x + 0.01, y), (x, y + 0.01)])
#     return all(not p.intersects(o) for o in obstacles)


# ACTIONS = [
#     (-1, 0), (1, 0), (0, -1), (0, 1),
#     (-1, -1), (-1, 1), (1, -1), (1, 1)
# ]


# def valid_actions(state):
#     x, y = state
#     out = []
#     for dx, dy in ACTIONS:
#         nx, ny = x + dx, y + dy
#         if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
#             wx, wy = grid_to_world(nx, ny)
#             if is_obstacle_free(wx, wy):
#                 out.append((dx, dy))
#     return out


# # ============================================================
# # 2) Q-LEARNING SETUP
# # ============================================================
# alpha = 0.1
# gamma = 0.95
# epsilon = 0.2
# episodes = 3000
# max_steps = 200

# Q = {}


# def get_Q(s, a):
#     return Q.get(s, {}).get(a, 0.0)


# def set_Q(s, a, v):
#     Q.setdefault(s, {})[a] = v


# start_grid = world_to_grid(*start)
# goal_grid = world_to_grid(*goal)


# def dist_to_goal(s):
#     return np.linalg.norm(np.array(s) - goal_grid)


# # ============================================================
# # 3) Q-LEARNING LOOP (LLM NUMERIC REWARD)
# # ============================================================
# pbar = trange(episodes, desc="Q-learning")

# for ep in pbar:
#     s = start_grid
#     total_reward = 0.0
#     visit_count = defaultdict(int)

#     chunk_start_dist = dist_to_goal(s)
#     chunk_revisits = 0
#     chunk_steps = 0

#     for step in range(max_steps):
#         if s == goal_grid:
#             break

#         actions = valid_actions(s)
#         if not actions:
#             break

#         if random.random() < epsilon:
#             a = random.choice(actions)
#         else:
#             a = max(actions, key=lambda x: get_Q(s, x))

#         s_next = (s[0] + a[0], s[1] + a[1])
#         visit_count[s_next] += 1

#         # ---- BASE SHAPING (kept minimal) ----
#         base_reward = -dist_to_goal(s_next)

#         # ---- CHUNK STATISTICS ----
#         chunk_revisits += (visit_count[s_next] > 1)
#         chunk_steps += 1

#         # ---- LLM REWARD (chunk-level) ----
#         llm_r = 0.0
#         if USE_LLM_REWARD and (chunk_steps % CHUNK_SIZE == 0):
#             summary = f"""
# Start distance: {chunk_start_dist:.2f}
# End distance: {dist_to_goal(s_next):.2f}
# Revisited states: {chunk_revisits}
# Steps: {chunk_steps}
# Reached goal: {s_next == goal_grid}
# """
#             llm_r = llm_numeric_reward(summary)
#             chunk_start_dist = dist_to_goal(s_next)
#             chunk_revisits = 0
#             chunk_steps = 0

#         reward = base_reward + LLM_REWARD_SCALE * llm_r
#         total_reward += reward

#         best_next = max(
#             [get_Q(s_next, ap) for ap in valid_actions(s_next)],
#             default=0.0,
#         )

#         td_target = reward + gamma * best_next
#         td_error = td_target - get_Q(s, a)
#         set_Q(s, a, get_Q(s, a) + alpha * td_error)

#         s = s_next

#     if ep % 50 == 0:
#         pbar.set_postfix(R=f"{total_reward:.1f}")


# # ============================================================
# # 4) EXTRACT GREEDY PATH (UNCHANGED)
# # ============================================================
# path = [start_grid]
# s = start_grid

# for _ in range(max_steps):
#     if s == goal_grid:
#         break
#     a = max(valid_actions(s), key=lambda x: get_Q(s, x))
#     s = (s[0] + a[0], s[1] + a[1])
#     path.append(s)


# # ============================================================
# # 5) VISUALIZATION (UNCHANGED)
# # ============================================================
# plt.figure(figsize=(6, 6))
# ax = plt.gca()

# for o in obstacles:
#     x, y = o.exterior.xy
#     ax.fill(x, y, alpha=0.5)

# ax.scatter(*start, c="green", s=100)
# ax.scatter(*goal, c="red", s=100)

# path_world = np.array([grid_to_world(*p) for p in path])
# ax.plot(path_world[:, 0], path_world[:, 1], c="blue", linewidth=3)

# plt.grid(True)
# plt.show()


"""
Q-learning grid planner with polygon obstacles + visualization
LLM (Qwen-3B) as NUMERICAL reward model (episode-level, cached, fast).

- No loss of original details
- LLM loaded ONCE
- LLM queried ONCE per episode
- Reward cached
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import random
from tqdm import trange
from collections import defaultdict

# ============================================================
# 0) LLM CONFIG (FAST + SAFE)
# ============================================================
USE_LLM_REWARD = True
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # or local path
LLM_REWARD_CLIP = 50.0

LLM_PROMPT_TEMPLATE = """You are a reward function for reinforcement learning.

Output a single real number.
Higher is better.

Guidelines:
- Reaching the goal is very good
- Getting closer to the goal is good
- Revisiting states is bad
- Wandering is bad
- Do NOT explain
- Output ONLY a number

Episode summary:
{summary}
"""

reward_cache = {}

if USE_LLM_REWARD:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()


def llm_episode_reward(summary: str) -> float:
    """
    Cached episode-level LLM reward.
    """
    key = hash(summary)
    if key in reward_cache:
        return reward_cache[key]

    prompt = LLM_PROMPT_TEMPLATE.format(summary=summary)

    messages = [
        {"role": "system", "content": "Output only a number."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = "SYSTEM: Output only a number.\nUSER: " + prompt + "\nASSISTANT:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    token = raw.split()[-1]

    try:
        r = float(token)
    except ValueError:
        r = 0.0

    r = float(np.clip(r, -LLM_REWARD_CLIP, LLM_REWARD_CLIP))
    reward_cache[key] = r
    return r


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
# 2) Q-LEARNING SETUP (UNCHANGED)
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
# 3) Q-LEARNING LOOP (EPISODE-LEVEL LLM REWARD)
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

    # ---------------- EPISODE LLM REWARD ----------------
    end_dist = dist_to_goal(s)
    revisits = sum(v > 1 for v in visit_count.values())
    reached_goal = (s == goal_grid)

    if USE_LLM_REWARD:
        summary = f"""
Start distance: {start_dist:.2f}
End distance: {end_dist:.2f}
Steps taken: {len(transitions)}
Revisited states: {revisits}
Reached goal: {reached_goal}
"""
        R_llm = llm_episode_reward(summary)
    else:
        R_llm = 0.0

    # Monte-Carlo-style update (stable with noisy rewards)
    for (s, a, r, s_next) in transitions:
        target = r + R_llm
        set_Q(s, a, get_Q(s, a) + alpha * (target - get_Q(s, a)))

    if ep % 50 == 0:
        pbar.set_postfix(LLM_R=f"{R_llm:.1f}", cache=len(reward_cache))


# ============================================================
# 4) EXTRACT GREEDY PATH (UNCHANGED)
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
# 5) VISUALIZATION (UNCHANGED)
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

