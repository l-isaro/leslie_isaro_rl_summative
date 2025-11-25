# training/train_reinforce.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from environment.learning_path_env import LearningPathEnv
from tqdm import tqdm

# Simple educational REINFORCE with TensorBoard + tqdm
TOTAL_EPISODES = 200000
GAMMA = 0.99

def main():
    env = LearningPathEnv(n_topics=6, time_budget=30, seed=0)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    policy = nn.Sequential(
        nn.Linear(obs_dim, 128), nn.ReLU(),
        nn.Linear(128, n_actions), nn.Softmax(dim=-1)
    )
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir="results/tb_reinforce")

    def rollout(max_steps=200):
        s, _ = env.reset()
        traj = []
        for _ in range(max_steps):
            with torch.no_grad():
                pi = policy(torch.tensor(s, dtype=torch.float32))
            a = torch.distributions.Categorical(pi).sample().item()
            s2, r, done, _, _ = env.step(a)
            traj.append((s, a, r))
            s = s2
            if done:
                break
        return traj

    pbar = tqdm(range(1, TOTAL_EPISODES + 1), desc="REINFORCE training", unit="ep")
    for ep in pbar:
        traj = rollout()
        # compute returns
        G, returns = 0.0, []
        for _, _, r in reversed(traj):
            G = r + GAMMA * G
            returns.append(G)
        returns = list(reversed(returns))
        Rt = torch.tensor(returns, dtype=torch.float32)
        Rt = (Rt - Rt.mean()) / (Rt.std() + 1e-8)

        logps = []
        for (s, a, _), Gt in zip(traj, Rt):
            pi = policy(torch.tensor(s, dtype=torch.float32))
            logps.append(torch.log(pi[a]) * Gt)
        loss = -torch.stack(logps).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging
        ep_return = sum(r for *_, r in traj)
        writer.add_scalar("episode_return", ep_return, ep)
        writer.add_scalar("loss", loss.item(), ep)
        pbar.set_postfix(ret=f"{ep_return:.1f}", loss=f"{loss.item():.2f}")

    writer.close()
    torch.save(policy.state_dict(), "results/reinforce_policy.pt")

if __name__ == "__main__":
    main()
