# 🚀 Adaptive Iterative Feedback Prompting (AIFP) for Obstacle-Aware Path Planning via LLMs

## 📜 Overview

This repository contains the code and dataset for the paper:

**"Adaptive Iterative Feedback Prompting for Obstacle-Aware Path Planning via LLMs."**

### 📌 Abstract

Planning is a critical component for intelligent agents, especially in **Human-Robot Interaction (HRI)**. Large Language Models (**LLMs**) demonstrate potential in planning but struggle with **spatial reasoning**. This work introduces **Adaptive Iterative Feedback Prompting (AIFP)**, a novel framework that improves **LLM-based path planning** by incorporating **real-time environmental feedback**. AIFP prompts an LLM iteratively to generate partial trajectories, evaluates them for **collision detection**, and refines them when necessary using a **Receding Horizon Planning (RHP) approach**.

## 🔑 Key Features

✅ **LLM-based path planning** with adaptive feedback  
✅ **Collision-aware trajectory generation**  
✅ **Iterative re-planning mechanism** using **Receding Horizon Planning (RHP)**  
✅ **Handles static and dynamic obstacles**  
✅ **Improves success rate by 33.3% compared to naive prompting**  
✅ **Fully implemented with OpenAI's GPT-4 API**  

## 📂 Repository Structure

```
├── src/                   # Source code for AIFP framework
│   ├── llmplanner.ipynb         # Core implementation of AIFP
├── results/               # Outputs of path planning trials
├── README.md              # This README file
└── requirements.txt       # Required Python dependencies
```

## 🛠 Installation & Setup

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/AIFP-PathPlanning.git
cd AIFP-PathPlanning
```


## 📊 Experimental Results

| Environment         | AIFP Success Rate (%) | Naïve Prompting (%) |
|--------------------|---------------------|--------------------|
| Single Obstacle   | **55.6%**            | 22.3%             |
| Double Obstacles  | **36.7%**            | 14.0%             |
| Random Obstacles  | **31.5%**            | 12.5%             |
| Moving Obstacle   | **48.5%**            | N/A               |
| Moving Goal       | **51.5%**            | N/A               |

✔️ **AIFP significantly outperforms naïve prompting**, especially in static environments! 🚀

## 📌 Citation

If you use this work, please cite:

```
@article{AIFP2025,
  title={Adaptive Iterative Feedback Prompting for Obstacle-Aware Path Planning via LLMs},
  author={Masoud Jafaripour, Shadan Golestan, Shotaro Miwa, Yoshihiro Mitsuka, Osmar R. Zaiane},
  year={2025},
  journal={Preprint}
}
```

## 🏗️ Future Work

- 🔹 Extend AIFP to **3D navigation tasks**  
- 🔹 Integrate **Vision-Language Models (VLMs)** for richer environmental perception  
- 🔹 Explore **graph-based path representations** for improved trajectory optimization  

---

🚀 **Star** ⭐ this repo if you find it useful!  
📧 Feel free to submit issues, PRs, or suggestions.

