# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from config import DQNConfig


class DQN(nn.Module):
    """ """

    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        # super(DQN, self).__init__()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.

    Experience replay breaks the correlation between consecutive samples,
    improving training stability and sample efficiency.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(
            *batch
        )  # transposes the batch structure, grouping by column
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN and DDQN Agents with experience replay and target network.

    Key components:
    1. Policy Network: Current Q-network used for action selection
    2. Target Network: Stabilizes training by providing fixed Q-targets
    3. Experience Replay: Breaks correlation between consecutive samples
    4. Epsilon-Greedy: Balances exploration vs exploitation
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        config: DQNConfig,
    ):
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        # Hyperparameters from config
        self.action_dim = action_dim
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.update_counter = 0

        self.use_double_dqn = config.use_double_dqn
        self.use_soft_update = config.use_soft_update
        self.tau = config.tau

        # Networks
        self.policy_net = DQN(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the module in evaluation mode

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose best action from Q-network (exploitation)
        """
        state = state.astype(np.float32)  # Normalize state

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        """
        Perform one gradient descent step on a batch of transitions.

        DQN Update Rule:
        Loss = (r + γ * max_a' Q_target(s', a') - Q_policy(s, a))^2
        """
        # Replay Buffer Warmup Guard (Preventing sampling crash)
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        # states = torch.FloatTensor(states).to(device=self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Switchable implementation
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                # 1. Select best action using policy network
                next_actions = self.policy_net(next_states).argmax(1)

                # 2. Evaluate that action using target network
                next_q_values = (
                    self.target_net(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        # Update target network
        self.update_counter += 1

        if self.use_soft_update:
            # Soft update (Polyak averaging) - smoother convergence and slightly more stable curves
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Hard update
            if self.update_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
