# agent.py
# Dueling DQN implementation (PyTorch).
#
# Architecture follows:
#   Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning"
# Applied to ETF selection as recommended by:
#   Yasin & Gill (2024) "RL Framework for Quantitative Trading" arXiv:2411.07585
#
# Key design choices:
#   - MLP policy (paper showed MLP > LSTM for daily ETF data)
#   - Separate Value and Advantage streams (Dueling — better for multi-action spaces)
#   - Experience replay buffer (100k transitions)
#   - Hard target network update every TARGET_UPDATE_FREQ steps
#   - epsilon-greedy exploration: 1.0 → 0.05 over first 50% of training

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config


# ── Dueling DQN Network ───────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling architecture:
        Input → shared trunk → [Value stream | Advantage stream]
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    The mean-subtraction ensures identifiability:
    V and A cannot compensate for each other arbitrarily.
    """

    def __init__(self, state_size: int, n_actions: int,
                 hidden: int = config.HIDDEN_UNITS):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Value stream V(s) — scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage stream A(s,a) — one per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value    = self.value_stream(features)            # (batch, 1)
        advantage= self.advantage_stream(features)        # (batch, n_actions)
        # Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = config.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_size: int,
                 n_actions:  int   = config.N_ACTIONS,
                 lr:         float = config.LEARNING_RATE,
                 gamma:      float = config.GAMMA,
                 eps_start:  float = config.EPSILON_START,
                 eps_end:    float = config.EPSILON_END,
                 eps_decay_frac: float = config.EPSILON_DECAY_FRAC,
                 buffer_size: int  = config.REPLAY_BUFFER_SIZE,
                 batch_size:  int  = config.BATCH_SIZE,
                 target_update: int = config.TARGET_UPDATE_FREQ,
                 total_steps: int  = 100_000):

        self.n_actions    = n_actions
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update= target_update
        self.steps_done   = 0

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay_steps = int(total_steps * eps_decay_frac)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.online_net = DuelingDQN(state_size, n_actions).to(self.device)
        self.target_net = DuelingDQN(state_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — more stable than MSE

    # ── Epsilon ───────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps_done / (self.eps_decay_steps + 1))
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - progress)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return raw Q-values for all actions (for UI display)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.online_net(s).cpu().numpy().flatten()

    # ── Learning ──────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self._update_target()

    def learn(self) -> float:
        if len(self.buffer) < config.MIN_REPLAY_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN: online selects action, target evaluates)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    def _update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "steps_done":  self.steps_done,
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No weights at {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        self.online_net.eval()
        self.target_net.eval()
