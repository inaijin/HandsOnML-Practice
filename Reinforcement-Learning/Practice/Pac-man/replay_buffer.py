import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done))
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.stack(states)
        next_states = np.stack(next_states)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
