import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer

class ConvQNet(nn.Module):
    def __init__(self, in_channels, grid_size, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc_input = 64 * grid_size * grid_size
        self.fc1 = nn.Linear(self.fc_input, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        # x: B x C x H x W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, obs_shape, action_size, device=None, lr=1e-3, gamma=0.99,
                 buffer_size=20000, batch_size=64, sync_every=1000, double_dqn=False):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device if device is not None else torch.device("cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.double_dqn = double_dqn

        in_ch, h, w = obs_shape
        self.online = ConvQNet(in_ch, h, action_size).to(self.device)
        self.target = ConvQNet(in_ch, h, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.sync_every = sync_every
        self.learn_steps = 0
        self.frame_idx = 0

        # epsilon schedule
        self.eps_start = 1.0
        self.eps_final = 0.05
        self.eps_decay = 10000

    def get_epsilon(self, episode, total_episodes):
        # simple schedule: decay with episodes
        fraction = min(1.0, episode / (total_episodes * 0.6))
        return self.eps_start + (self.eps_final - self.eps_start) * fraction

    def act(self, obs, epsilon=0.0):
        # obs: numpy (C,H,W)
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x)
        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)
        self.frame_idx += 1

    def learn(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(states).gather(1, actions)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_states).argmax(1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
            else:
                next_q = self.target(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save(self.online.state_dict(), path)
        print("Saved model to", path)

    def load(self, path):
        if os.path.exists(path):
            self.online.load_state_dict(torch.load(path, map_location=self.device))
            self.target.load_state_dict(self.online.state_dict())
            print("Loaded model from", path)
        else:
            print("Checkpoint not found:", path)
