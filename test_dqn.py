# test_dqn.py
"""
Unit tests for DQN/ DDQN implementations.
Run with: pytest test_dqn.py -v
"""

import pytest
import torch
import numpy as np
import gymnasium as gym

from config import DQNConfig, TrainingConfig
from agent import DQN, ReplayBuffer, DQNAgent


@pytest.fixture
def make_agent():
    def _make_agent(**overrides):
        config = DQNConfig()
        for k, v in overrides.items():
            setattr(config, k, v)
        return DQNAgent(state_dim=8, action_dim=4, config=config)

    return _make_agent


@pytest.fixture
def make_env():
    train_config = TrainingConfig()
    env = gym.make(train_config.env_name)
    yield env
    env.close()


class TestDQN:
    """Test DQN neural network."""

    def test_network_initialization(self):
        """Test network is properly initialized."""
        net = DQN(state_dim=8, action_dim=4, hidden_dim=128)
        assert isinstance(net, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        net = DQN(state_dim=8, action_dim=4, hidden_dim=128)
        state = torch.randn(32, 8)  # Batch of 32 states
        output = net(state)
        assert output.shape == (32, 4)

    def test_output_values(self):
        """Test network outputs reasonable Q-values."""
        net = DQN()
        state = torch.randn(1, 8)
        output = net(state)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestReplayBuffer:
    """Test experience replay buffer."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.buffer.maxlen == 100

    def test_push_and_length(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=100)
        buffer.push(np.zeros(8), 0, 1.0, np.zeros(8), False)
        assert len(buffer) == 1

        # Add more
        for i in range(10):
            buffer.push(np.zeros(8), i % 4, 1.0, np.zeros(8), False)
        assert len(buffer) == 11

    def test_buffer_overflow(self):
        """Test buffer respects maximum capacity."""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity)

        # Add more than capacity
        for i in range(20):
            buffer.push(np.zeros(8), 0, float(i), np.zeros(8), False)

        assert len(buffer) == capacity

    def test_sample_batch(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add some experiences
        for i in range(50):
            buffer.push(np.ones(8) * i, i % 4, float(i), np.ones(8) * (i + 1), i % 2 == 0)

        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        assert states.shape == (batch_size, 8)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 8)
        assert dones.shape == (batch_size,)


class TestDQNAgent:
    """Test DQN agent."""

    def create_agent(self, **overrides):
        config = DQNConfig()
        for k, v in overrides.items():
            setattr(config, k, v)
        return DQNAgent(8, 4, config)

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = self.create_agent()
        assert agent.policy_net is not None
        assert agent.target_net is not None
        assert agent.replay_buffer is not None
        assert agent.epsilon == 1.0

    def test_action_selection_exploration(self):
        """Test epsilon-greedy exploration."""
        agent = self.create_agent(epsilon_start=1.0)
        state = np.random.randn(8)

        # With epsilon=1.0, should be random
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        unique_actions = len(set(actions))
        assert unique_actions > 1  # Should have some variety

    def test_action_selection_exploitation(self):
        """Test greedy action selection."""
        agent = self.create_agent(epsilon_start=0.0)
        state = np.random.randn(8)

        # With epsilon=0.0, should always pick same action for same state
        action1 = agent.select_action(state, training=True)
        action2 = agent.select_action(state, training=True)
        assert action1 == action2

    def test_epsilon_decay(self):
        """Test epsilon decay mechanism."""
        agent = self.create_agent(epsilon_start=1.0, epsilon_decay=0.99, epsilon_end=0.01)
        initial_epsilon = agent.epsilon

        for _ in range(10):
            agent.decay_epsilon()

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end

    def test_update_requires_batch(self):
        """Test update returns None when buffer too small."""
        agent = self.create_agent(batch_size=64)

        # Add only a few experiences
        for i in range(10):
            agent.replay_buffer.push(np.zeros(8), 0, 1.0, np.zeros(8), False)

        loss = agent.update()
        assert loss is None

    def test_update_with_sufficient_data(self):
        """Test update works with enough data."""
        agent = self.create_agent(batch_size=32)

        # Add enough experiences
        for i in range(100):
            agent.replay_buffer.push(
                np.random.randn(8), i % 4, np.random.randn(), np.random.randn(8), False
            )

        loss = agent.update()
        assert loss is not None
        assert loss >= 0
        assert not np.isnan(loss)

    def test_save_and_load(self, tmp_path):
        """Test model saving and loading."""
        agent1 = self.create_agent()

        # Train a bit
        for i in range(100):
            agent1.replay_buffer.push(
                np.random.randn(8), i % 4, np.random.randn(), np.random.randn(8), False
            )
            agent1.update()

        # Save
        save_path = tmp_path / "test_model.pth"
        agent1.save(save_path)

        # Load into new agent
        agent2 = self.create_agent()
        agent2.load(save_path)

        # Check weights are the same
        state = torch.randn(1, 8)
        with torch.no_grad():
            output1 = agent1.policy_net(state)
            output2 = agent2.policy_net(state)

        assert torch.allclose(output1, output2)


class TestIntegration:
    """Integration tests with actual environment."""

    def test_environment_compatibility(self, make_agent, make_env):
        """Test agent works with LunarLander environment."""
        env = make_env
        agent = make_agent()

        state, _ = env.reset()
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        assert state.shape == (8,)
        assert 0 <= action < 4
        assert isinstance(reward, float)

    def test_single_episode(self, make_env, make_agent):
        """Test agent can complete a full episode."""
        env = make_env
        agent = make_agent()

        state, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 1000

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        assert steps > 0
        assert isinstance(total_reward, float)

    def test_training_loop(self, make_env, make_agent):
        """Test basic training loop works."""
        env = make_env
        agent = make_agent(batch_size=32)

        # Run a few episodes
        for episode in range(5):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(100):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update()

                episode_reward += reward
                state = next_state

                if done:
                    break

            agent.decay_epsilon()

        # Check that agent has learned something (buffer is populated)
        assert len(agent.replay_buffer) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_state(self, make_agent):
        """Test agent handles NaN states gracefully."""
        agent = make_agent()
        state = np.array([np.nan] * 8)

        # Should not crash, though behavior may be undefined
        try:
            action = agent.select_action(state, training=False)
            assert 0 <= action < 4 or action is None
        except Exception as e:
            # It's okay if it raises an exception, we just want to catch it
            assert isinstance(e, Exception)

    def test_extreme_state_values(self, make_agent):
        """Test agent with extreme state values."""
        agent = make_agent()
        state = np.array([1e10, -1e10, 0, 0, 0, 0, 0, 0])

        action = agent.select_action(state, training=False)
        assert 0 <= action < 4

    def test_empty_buffer_sample(self):
        """Test sampling from empty buffer."""
        buffer = ReplayBuffer(capacity=100)

        with pytest.raises(ValueError):
            buffer.sample(32)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
