# environment/learning_path_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LearningPathEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_topics=6, time_budget=30, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.n = n_topics
        # topic attributes
        self.difficulty = rng.integers(1, 4, size=self.n)          # 1,2,3
        self.base_gain  = np.array([0.10, 0.13, 0.16])[self.difficulty-1]
        self.cost       = np.array([1, 2, 3])[self.difficulty-1]

        # adjacency (ring or small-world for simplicity)
        self.adj = {i: [(i-1)%self.n, (i+1)%self.n] for i in range(self.n)}
        # add two non-local edges for variety
        self.adj[0].append(self.n//2); self.adj[self.n//2].append(0)

        self.time_budget = time_budget
        self.alpha = 0.8
        self.beta, self.lc, self.lf = 10.0, 0.2, 0.5
        self.spaced_bonus_step = 2

        # state spaces
        obs_dim = 2*self.n + 2
        self.observation_space = spaces.Box(low=0., high=1., shape=(obs_dim,), dtype=np.float32)
        # actions: choose neighbor index OR review OR rest
        self.max_neighbors = 3  # left, right, long-jump
        self.ACTION_REVIEW = self.max_neighbors
        self.ACTION_REST   = self.max_neighbors + 1
        self.action_space  = spaces.Discrete(self.max_neighbors + 2)

        self.reset(seed=seed)

    def _obs(self):
        onehot = np.zeros(self.n, dtype=np.float32); onehot[self.pos]=1
        return np.concatenate([onehot, self.mastery.astype(np.float32),
                               np.array([self.time_left/self.time_budget, self.fatigue], np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mastery = np.zeros(self.n, dtype=np.float32)
        self.last_step_on_topic = -np.ones(self.n, dtype=np.int32)
        self.pos = 0
        self.fatigue = 0.0
        self.step_idx = 0
        self.time_left = self.time_budget
        return self._obs(), {}

    def step(self, action):
        reward = 0.0
        done = False

        # map discrete action to actual next topic or control
        neighbors = self.adj[self.pos][:self.max_neighbors] + [None]*(self.max_neighbors-len(self.adj[self.pos]))
        if action < self.max_neighbors and neighbors[action] is not None:
            self.pos = neighbors[action]
            i = self.pos
            # spaced bonus
            spaced = (self.last_step_on_topic[i] >= 0) and (self.step_idx - self.last_step_on_topic[i] >= self.spaced_bonus_step)
            s_bonus = 1.15 if spaced else 1.0
            delta = self.alpha*(1.0 - self.mastery[i]) * self.base_gain[i] * s_bonus
            self.mastery[i] = np.clip(self.mastery[i] + delta, 0, 1)
            self.last_step_on_topic[i] = self.step_idx
            self.fatigue = np.clip(self.fatigue + (0.05 if self.difficulty[i]==3 else 0.02), 0, 1)
            cost = self.cost[i]
            reward = self.beta*delta - self.lc*cost - self.lf*self.fatigue

        elif action == self.ACTION_REVIEW:
            i = self.pos
            delta = 0.6*self.alpha*(1.0 - self.mastery[i]) * self.base_gain[i] * 1.2
            self.mastery[i] = np.clip(self.mastery[i] + delta, 0, 1)
            self.fatigue = np.clip(self.fatigue + 0.01, 0, 1)
            cost = 1
            reward = self.beta*delta - self.lc*cost - self.lf*self.fatigue

        elif action == self.ACTION_REST:
            self.fatigue = max(0.0, self.fatigue - 0.1)
            cost = 1
            reward = - self.lc*cost - 0.1  # small negative to discourage idling
        else:
            reward = -1.0  # invalid/no-op

        self.time_left -= cost if 'cost' in locals() else 1
        self.step_idx += 1

        if (self.mastery >= 0.8).all():
            reward += 50.0
            done = True
        if self.time_left <= 0 or self.step_idx >= 200:
            done = True

        return self._obs(), float(reward), done, False, {}

    def render(self):
        # simple print; you’ll replace with matplotlib/pygame
        print(f"pos={self.pos} mastery={np.round(self.mastery,2)} time={self.time_left} fatigue={self.fatigue:.2f}")
