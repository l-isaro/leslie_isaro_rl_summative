# training/train_a2c.py
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.learning_path_env import LearningPathEnv
from tqdm import tqdm

TOTAL_STEPS = 200_000

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total, desc="A2C training", unit="ts")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.n = self.model.num_timesteps
            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.n = self.total
            self.pbar.close()

if __name__ == "__main__":
    def make_env():
        return Monitor(LearningPathEnv(n_topics=6, time_budget=30, seed=7))
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=8,
        gamma=0.99,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="results/tb_a2c",
    )

    cb = TqdmCallback(TOTAL_STEPS)
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False, tb_log_name="A2C", callback=cb)
    model.save("results/a2c_model")
