import os
import torch
from tqdm import trange
from env.pacman_env import PacmanEnv
from agents.dqn_agent import DQNAgent

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class Trainer:
    def __init__(self, env: PacmanEnv, agent: DQNAgent, device: torch.device):
        self.env = env
        self.agent = agent
        self.device = device

    def train(self, episodes=1000, render=False, render_every=50):
        best_reward = -1e9
        for ep in trange(episodes, desc="Episodes"):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            while not done:
                epsilon = self.agent.get_epsilon(ep, episodes)
                action = self.agent.act(obs, epsilon)
                next_obs, reward, done, info = self.env.step(action)
                self.agent.remember(obs, action, reward, next_obs, done)
                self.agent.learn()
                obs = next_obs
                ep_reward += reward
                step += 1
                if render and (ep % render_every == 0):
                    self.env.render(delay=10)  # small delay in ms
            if (ep_reward > best_reward):
                best_reward = ep_reward
                self.agent.save(os.path.join(CHECKPOINT_DIR, f"best_dqn.pth"))
            if ep % 50 == 0:
                self.agent.save(os.path.join(CHECKPOINT_DIR, f"dqn_ep{ep}.pth"))
            if ep % 10 == 0:
                print(f"Episode {ep} reward {ep_reward:.1f} epsilon {epsilon:.3f}")
        print("Training completed.")
