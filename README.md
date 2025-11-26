# Mission-Based Reinforcement Learning: Learning Path Optimizer

A custom reinforcement learning system modeling student improvement, topic mastery, and efficient learning behavior.

## Overview

This project implements a **custom reinforcement learning environment** where an agent represents a *student* navigating a map of learning topics. The student chooses **study**, **review**, or **rest** actions to increase mastery while managing **fatigue** and **time limits**.

The environment rewards **improvement**, not raw performance — aligning with real educational settings where **consistent learning progress** is more important than one-time outcomes.

Four RL algorithms were implemented:

- **DQN (Deep Q-Network)**
- **PPO (Proximal Policy Optimization)**
- **A2C (Advantage Actor-Critic)**
- **REINFORCE (Policy Gradient)**

Each algorithm was tuned via a **10-configuration hyperparameter sweep**, evaluated, and visualized in a custom **pygame viewer**.

## Custom Environment: `LearningPathEnv`

### Agent Behavior

The agent represents a **student** with mastery levels across multiple topics. It must balance:

- **Studying**
- **Reviewing**
- **Resting**

…to maximize total learning progress under fatigue and time pressure.

## Action Space (Discrete: 5)

| **Action** | **Meaning** |
|-----------|-------------|
| `move-0`  | Move to neighbor 1 (study) |
| `move-1`  | Move to neighbor 2 (study) |
| `move-2`  | Long-jump neighbor (study) |
| `review`  | Review current topic |
| `rest`    | Reduce fatigue |

## Observation Space (14 dims)

The agent observes:

- **One-hot current topic position (6)**
- **Mastery vector (6)**
- **Normalized time left (1)**
- **Fatigue level (1)**

---

## Reward Function (Improvement-Based)

```python
reward = 10 * delta - 0.2 * time_cost - 0.5 * fatigue
```

### Where:

- **delta** = improvement in mastery for the current topic  
- **time_cost** = how expensive the action is  
- **fatigue** = penalizes studying too much  

### Completion Bonus

`+50` if all topics reach mastery ≥ 0.8.

This creates a reward system based entirely on **measurable improvement**, not just activity.

## Training the Models

#### Install dependencies:

```bash
pip install -r requirements.txt
```

#### Train individual models:

```bash
python -m training.train_dqn
python -m training.train_ppo
python -m training.train_a2c
python -m training.train_reinforce
```

## Visualization

### Run the best DQN model (cfg 0)

```bash
python main.py
```

## Best Models Summary

| **Algorithm** | **Best Config**    | **Mean Reward** |
|---------------|--------------------|------------------|
| **DQN**       | `dqn_cfg_0`        | 12.11            |
| **PPO**       | `ppo_cfg_0`        | 11.54            |
| **A2C**       | `a2c_cfg_8`        | 10.60            |
| **REINFORCE** | `reinforce_cfg_0`  | 8.99             |
