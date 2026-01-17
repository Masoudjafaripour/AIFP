"""
PURE VLM grid planner with polygon obstacles + visualization
Qwen-VL used as NUMERICAL planner:

(text + rendered image of scene) -> VLM -> numerical plan (sequence of dx dy)

- VLM loaded ONCE
- Planning is cached
- Uses rendered grid image
- Inference modes:
    1) naive
    2) CoT
    3) tree-search (ToT-style: model proposes candidates; we verify + optionally expand)
    4) Best-of-N (sample N plans; pick best by numeric verifier)
- Visualize solution found by all inference approaches

"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from collections import deque
import random
from tqdm import tqdm


# ============================================================
# 0) VLM CONFIG (FAST + SAFE)
# ============================================================
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"   # or local path
MAX_PLAN_STEPS = 140

USE_VLM_VERIFIER = False     # if True: extra VLM calls for verifying candidates (slower)
VLM_REWARD_CLIP = 50.0

plan_cache = {}   # (mode, state, history_sig, args...) -> (path, score, raw_decoded)
step_cache = {}   # (mode, state, history_sig, args...) -> action (dx,dy)

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
# 1) ENVIRONMENT (same as yours)
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


start_grid = world_to_grid(*start)
goal_grid = world_to_grid(*goal)


def dist_to_goal(s):
    return float(np.linalg.norm(np.array(s) - np.array(goal_grid)))


# ============================================================
# 2) GRID RENDERING (your style)
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
# 3) OPTIONAL VLM VERIFIER (numeric reward from final image)
# ============================================================
reward_cache = {}

@torch.no_grad()
def vlm_episode_reward(final_state):
    img = render_grid_image(final_state)
    key = hash(img.tobytes())
    if key in reward_cache:
        return reward_cache[key]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Score how well this image shows the agent reaching the red goal safely. Output only a single real number. Higher is better."}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    token = decoded.split()[-1]

    try:
        r = float(token)
    except ValueError:
        r = 0.0

    r = float(np.clip(r, -VLM_REWARD_CLIP, VLM_REWARD_CLIP))
    reward_cache[key] = r
    return r


# ============================================================
# 4) PROMPTS (state placeholders updated each call)
# ============================================================
PROMPT_NAIVE = """You see a grid image:
- BLUE cell = agent
- RED cell = goal
- GRAY polygons = obstacles (do not enter)

Current state (grid): {sx}, {sy}
Goal state (grid): {gx}, {gy}

Return a NUMERICAL PLAN as a sequence of actions.
Use only these moves (dx,dy): {actions}

Output ONLY pairs "dx dy" separated by spaces or newlines. No extra text.
"""

PROMPT_COT = """You see a grid image:
- BLUE cell = agent
- RED cell = goal
- GRAY polygons = obstacles (do not enter)

Current state (grid): {sx}, {sy}
Goal state (grid): {gx}, {gy}
Recent visited states: {history}

Use only these moves (dx,dy): {actions}

Think step-by-step to avoid obstacles and reach the goal.
Then output ONLY the action sequence as pairs "dx dy". No extra text.
"""

PROMPT_TOT = """You see a grid image:
- BLUE cell = agent
- RED cell = goal
- GRAY polygons = obstacles (do not enter)

Current state (grid): {sx}, {sy}
Goal state (grid): {gx}, {gy}
Recent visited states: {history}

Use only these moves (dx,dy): {actions}

Do tree-search reasoning:
1) Propose {k} different candidate action sequences (each up to {h} steps).
2) Pick the best candidate.

Output ONLY the chosen final action sequence as pairs "dx dy". No extra text.
"""

PROMPT_BESTOFN = """You see a grid image:
- BLUE cell = agent
- RED cell = goal
- GRAY polygons = obstacles (do not enter)

Current state (grid): {sx}, {sy}
Goal state (grid): {gx}, {gy}
Recent visited states: {history}

Use only these moves (dx,dy): {actions}

Output ONLY one action sequence as pairs "dx dy". No extra text.
"""


# ============================================================
# 5) VLM CALL + PARSING
# ============================================================
def _history_sig(history, k=12):
    if not history:
        return "[]"
    return str(list(history)[-k:])


def _extract_ints(text):
    nums = []
    cur = ""
    for ch in text:
        if ch in "-0123456789":
            cur += ch
        else:
            if cur and cur != "-":
                nums.append(cur)
            cur = ""
    if cur and cur != "-":
        nums.append(cur)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except ValueError:
            pass
    return out


def parse_action_sequence(decoded, max_len=MAX_PLAN_STEPS):
    """
    Accepts any text that contains integers.
    We parse consecutive pairs as (dx,dy) and keep only those in ACTIONS.
    """
    tail = decoded.strip()[-4000:]
    ints = _extract_ints(tail)
    acts = []
    for i in range(0, len(ints) - 1, 2):
        dx, dy = ints[i], ints[i + 1]
        if (dx, dy) in ACTIONS:
            acts.append((dx, dy))
        if len(acts) >= max_len:
            break
    return acts


@torch.no_grad()
def vlm_generate_plan_text(img, prompt, do_sample=False, num_return=1, max_new_tokens=384):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text] * num_return,
        images=[img] * num_return,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.9 if do_sample else None,
        top_p=0.9 if do_sample else None,
    )

    decoded_list = []
    for i in range(outputs.shape[0]):
        decoded_list.append(processor.tokenizer.decode(outputs[i], skip_special_tokens=True).strip())
    return decoded_list


# ============================================================
# 6) SIMULATOR + NUMERIC VERIFIER (fast, deterministic)
# ============================================================
def simulate_from_actions(s0, actions):
    s = s0
    path = [s]
    for (dx, dy) in actions:
        nx, ny = s[0] + dx, s[1] + dy
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            break
        wx, wy = grid_to_world(nx, ny)
        if not is_obstacle_free(wx, wy):
            break
        s = (nx, ny)
        path.append(s)
        if s == goal_grid:
            break
    return path


def score_path(path):
    if not path:
        return -1e9
    final = path[-1]
    success = 1.0 if final == goal_grid else 0.0
    length = len(path)
    d = dist_to_goal(final)

    # Optionally include VLM reward on final state (slower if used heavily)
    if USE_VLM_VERIFIER:
        rv = vlm_episode_reward(final)
    else:
        rv = 0.0

    # Big success bonus, prefer shorter, prefer closer; add optional VLM reward
    return 1000.0 * success - 1.0 * length - 10.0 * d + 1.0 * rv


# ============================================================
# 7) PLANNERS (naive / CoT / ToT / Best-of-N)
# ============================================================
from tqdm import tqdm

@torch.no_grad()
def plan_full(mode, s0, history=None, horizon=MAX_PLAN_STEPS, k_tot=6, bestof_n=10):
    if history is None:
        history = deque(maxlen=12)

    img = render_grid_image(s0)
    sx, sy = s0
    gx, gy = goal_grid

    hist = _history_sig(history)
    actions_set = str(ACTIONS)

    cache_key = (mode, s0, hist, horizon, k_tot, bestof_n, USE_VLM_VERIFIER)
    if cache_key in plan_cache:
        return plan_cache[cache_key]

    # ---------------- GENERATION PHASE ----------------
    gen_pbar = tqdm(
        total=1,
        desc=f"{mode}: generating plan",
        leave=False,
        dynamic_ncols=True
    )

    if mode == "naive":
        prompt = PROMPT_NAIVE.format(
            sx=sx, sy=sy, gx=gx, gy=gy, actions=actions_set
        )
        decoded_list = vlm_generate_plan_text(
            img, prompt,
            do_sample=False,
            num_return=1,
            max_new_tokens=256
        )

    elif mode == "cot":
        prompt = PROMPT_COT.format(
            sx=sx, sy=sy, gx=gx, gy=gy,
            history=hist, actions=actions_set
        )
        decoded_list = vlm_generate_plan_text(
            img, prompt,
            do_sample=False,
            num_return=1,
            max_new_tokens=384
        )

    elif mode == "tot":
        prompt = PROMPT_TOT.format(
            sx=sx, sy=sy, gx=gx, gy=gy,
            history=hist, actions=actions_set,
            k=k_tot, h=horizon
        )
        decoded_list = vlm_generate_plan_text(
            img, prompt,
            do_sample=False,
            num_return=1,
            max_new_tokens=512
        )

    elif mode == "bestofn":
        prompt = PROMPT_BESTOFN.format(
            sx=sx, sy=sy, gx=gx, gy=gy,
            history=hist, actions=actions_set
        )
        decoded_list = vlm_generate_plan_text(
            img, prompt,
            do_sample=True,
            num_return=bestof_n,
            max_new_tokens=256
        )

    else:
        gen_pbar.close()
        raise ValueError("mode must be one of: naive, cot, tot, bestofn")

    gen_pbar.update(1)
    gen_pbar.close()

    # ---------------- SCORING PHASE ----------------
    best_path, best_score, best_decoded = None, -1e18, None

    for decoded in tqdm(
        decoded_list,
        desc=f"{mode}: scoring candidates",
        leave=False,
        dynamic_ncols=True
    ):
        acts = parse_action_sequence(decoded, max_len=horizon)
        path = simulate_from_actions(s0, acts)
        sc = score_path(path)

        if sc > best_score:
            best_score = sc
            best_path = path
            best_decoded = decoded

    if best_path is None:
        best_path = [s0]
        best_score = score_path(best_path)

    plan_cache[cache_key] = (best_path, best_score, best_decoded)
    return best_path, best_score, best_decoded



# ============================================================
# 8) RUN ALL MODES + VISUALIZATION
# ============================================================
modes = ["naive", "cot", "tot", "bestofn"]
paths = {}
scores = {}
raw = {}

history = deque(maxlen=12)

for mode in modes:
    p, sc, dec = plan_full(mode, start_grid, history=history, horizon=MAX_PLAN_STEPS, k_tot=6, bestof_n=10)
    paths[mode] = p
    scores[mode] = sc
    raw[mode] = dec

plt.figure(figsize=(7, 7))
ax = plt.gca()

for o in obstacles:
    x, y = o.exterior.xy
    ax.fill(x, y, alpha=0.5)

ax.scatter(*start, c="green", s=120, label="start")
ax.scatter(*goal, c="red", s=120, label="goal")

# for mode in modes:
for mode in tqdm(modes, desc="VLM planning modes"):
    p = paths[mode]
    p_world = np.array([grid_to_world(*pp) for pp in p])
    ax.plot(p_world[:, 0], p_world[:, 1], linewidth=2.5, label=f"{mode} ({scores[mode]:.1f})")

plt.grid(True)
plt.legend()
plt.title("PURE VLM planning: naive vs CoT vs ToT vs Best-of-N")
plt.show()
