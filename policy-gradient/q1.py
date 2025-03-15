# import gym
import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

gym.register_envs(ale_py)

# Check for MPS (Metal Performance Shaders) availability on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for GPU acceleration")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MPS not available, using device: {device}")


# Neural Network for function approximation
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights uniformly between -0.001 and 0.001 as specified
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.001, 0.001)
                nn.init.uniform_(m.bias, -0.001, 0.001)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Store experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays for batch processing
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Q-learning without replay buffer
class QLearning:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-network
        self.q_network = MLP(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Get Q-values
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Calculate target Q-value using Q-learning update rule
        target = q_values.clone()
        if done:
            target[0, action] = reward
        else:
            target[0, action] = reward + self.gamma * torch.max(next_q_values).item()

        # Update network
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Q-learning with replay buffer
class QLearningWithReplay:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
        buffer_capacity=1_000_000,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Initialize Q-network
        self.q_network = MLP(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # If buffer is too small, don't update yet
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # Get current Q-values for the taken actions
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Get maximum Q-values for next states (Q-learning)
        next_q_values = self.q_network(next_states_tensor).max(1, keepdim=True)[0]

        # Calculate target Q-values
        target_q_values = (
            rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        )

        # Calculate loss and update
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Expected SARSA without replay buffer
class ExpectedSARSA:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-network
        self.q_network = MLP(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def get_expected_q_value(self, q_values):
        # Calculate expected Q-value using epsilon-greedy policy
        best_action = torch.argmax(q_values).item()
        expected_q = 0.0

        # Sum over all possible actions weighted by policy probability
        for a in range(self.action_dim):
            if a == best_action:
                # Probability of selecting best action = (1-ε) + ε/|A|
                prob = (1 - self.epsilon) + (self.epsilon / self.action_dim)
            else:
                # Probability of selecting non-best action = ε/|A|
                prob = self.epsilon / self.action_dim
            expected_q += q_values[0, a].item() * prob

        return expected_q

    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Get Q-values
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Calculate target using expected SARSA
        target = q_values.clone()
        if done:
            target[0, action] = reward
        else:
            # Use expected value over policy instead of max
            expected_q = self.get_expected_q_value(next_q_values)
            target[0, action] = reward + self.gamma * expected_q

        # Update network
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Expected SARSA with replay buffer
class ExpectedSARSAWithReplay:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
        buffer_capacity=1_000_000,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Initialize Q-network
        self.q_network = MLP(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def get_expected_q_values_batch(self, next_q_values_batch):
        # Calculate expected Q-values using epsilon-greedy policy for a batch
        batch_size = next_q_values_batch.size(0)
        expected_q_values = torch.zeros(batch_size, 1)

        for b in range(batch_size):
            next_q_values = next_q_values_batch[b]
            best_action = torch.argmax(next_q_values).item()
            expected_q = 0.0

            # Calculate expected Q-value for each state in batch
            for a in range(self.action_dim):
                if a == best_action:
                    prob = (1 - self.epsilon) + (self.epsilon / self.action_dim)
                else:
                    prob = self.epsilon / self.action_dim
                expected_q += next_q_values[a].item() * prob

            expected_q_values[b, 0] = expected_q

        return expected_q_values

    def update(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # If buffer is too small, don't update yet
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # Get current Q-values for the taken actions
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Get Q-values for next states
        next_q_values = self.q_network(next_states_tensor)

        # Calculate expected Q-values for next states based on policy
        expected_q_values = self.get_expected_q_values_batch(next_q_values)

        # Calculate target Q-values
        target_q_values = (
            rewards_tensor + (1 - dones_tensor) * self.gamma * expected_q_values
        )

        # Calculate loss and update
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Function to run a single trial with a specific algorithm and parameters
def run_trial(
    env_name,
    algorithm_class,
    epsilon,
    learning_rate,
    with_replay=False,
    num_episodes=1000,
    hidden_dim=256,
    seed=None,
):
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Initialize environment with seed
    if seed is not None:
        env = gym.make(env_name, render_mode=None)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    else:
        env = gym.make(env_name)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    agent = algorithm_class(
        state_dim,
        action_dim,
        learning_rate=learning_rate,
        epsilon=epsilon,
        hidden_dim=hidden_dim,
    )

    # Run episodes
    episode_rewards = []

    for episode in range(num_episodes):
        # In newer gymnasium versions, reset returns (state, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Extract state from (state, info) tuple
        else:
            state = reset_result

        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Choose action based on current state
            action = agent.choose_action(state)

            # Take action in environment - newer versions return (next_state, reward, done, truncated, info)
            step_result = env.step(action)
            if len(step_result) == 5:  # New gym API
                next_state, reward, done, truncated, _ = step_result
            else:  # Old gym API
                next_state, reward, done, _ = step_result
                truncated = False

            # Update agent's knowledge
            agent.update(state, action, reward, next_state, done or truncated)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

        # Store episode reward
        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


# Function to run experiments for a specific algorithm and environment
def run_experiments(
    env_name,
    algorithm_class,
    algorithm_with_replay_class,
    epsilons,
    learning_rates,
    num_trials=50,
    num_episodes=1000,
    hidden_dim=256,
):
    results = {}

    # Without replay buffer
    for epsilon in epsilons:
        for lr in learning_rates:
            key = f"no_replay_e{epsilon}_lr{lr}"
            results[key] = []

            for trial in tqdm(
                range(num_trials), desc=f"No Replay: ε={epsilon}, α={lr}"
            ):
                rewards = run_trial(
                    env_name,
                    algorithm_class,
                    epsilon,
                    lr,
                    False,
                    num_episodes,
                    hidden_dim,
                    seed=trial,
                )
                results[key].append(rewards)

    # With replay buffer
    for epsilon in epsilons:
        for lr in learning_rates:
            key = f"with_replay_e{epsilon}_lr{lr}"
            results[key] = []

            for trial in tqdm(
                range(num_trials), desc=f"With Replay: ε={epsilon}, α={lr}"
            ):
                rewards = run_trial(
                    env_name,
                    algorithm_with_replay_class,
                    epsilon,
                    lr,
                    True,
                    num_episodes,
                    hidden_dim,
                    seed=trial,
                )
                results[key].append(rewards)

    return results


# Function to plot results
def plot_results(
    q_results, sarsa_results, epsilons, learning_rates, env_name, with_replay=False
):
    fig, axes = plt.subplots(
        len(epsilons), len(learning_rates), figsize=(15, 10), sharex=True, sharey=True
    )

    # Adjust for single-dimension subplots
    if len(epsilons) == 1 and len(learning_rates) == 1:
        axes = np.array([[axes]])
    elif len(epsilons) == 1:
        axes = axes.reshape(1, -1)
    elif len(learning_rates) == 1:
        axes = axes.reshape(-1, 1)

    for i, epsilon in enumerate(epsilons):
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]

            # Get the right keys
            key_prefix = "with_replay" if with_replay else "no_replay"
            q_key = f"{key_prefix}_e{epsilon}_lr{lr}"
            sarsa_key = f"{key_prefix}_e{epsilon}_lr{lr}"

            # Get the data
            q_data = np.array(q_results[q_key])
            sarsa_data = np.array(sarsa_results[sarsa_key])

            # Calculate mean and std
            q_mean = np.mean(q_data, axis=0)
            q_std = np.std(q_data, axis=0)
            sarsa_mean = np.mean(sarsa_data, axis=0)
            sarsa_std = np.std(sarsa_data, axis=0)

            # Plot
            episodes = np.arange(1, len(q_mean) + 1)
            ax.plot(episodes, q_mean, color="green", label="Q-Learning")
            ax.fill_between(
                episodes, q_mean - q_std, q_mean + q_std, color="green", alpha=0.2
            )
            ax.plot(episodes, sarsa_mean, color="red", label="Expected SARSA")
            ax.fill_between(
                episodes,
                sarsa_mean - sarsa_std,
                sarsa_mean + sarsa_std,
                color="red",
                alpha=0.2,
            )

            # Set title and labels
            ax.set_title(f"ε={epsilon}, α={lr}")
            if i == len(epsilons) - 1:
                ax.set_xlabel("Episode")
            if j == 0:
                ax.set_ylabel("Return")

            # Add legend to the first subplot
            if i == 0 and j == 0:
                ax.legend()

    # Set the main title
    replay_text = "with Replay Buffer" if with_replay else "without Replay Buffer"
    fig.suptitle(f"{env_name} {replay_text}", fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the figure
    os.makedirs("figures", exist_ok=True)
    filename = f"{env_name.replace('/', '_')}_{key_prefix}.png"
    plt.savefig(os.path.join("figures", filename))
    plt.close()


# Main function
def main():
    # Parameters
    epsilons = [0.01, 0.05, 0.1]
    learning_rates = [1 / 4, 1 / 8, 1 / 16]
    num_trials = 10
    num_episodes = 1000
    hidden_dim = 256

    # Run experiments for Acrobot-v1
    print("Running experiments for Acrobot-v1...")
    acrobot_q_results = run_experiments(
        "Acrobot-v1",
        QLearning,
        QLearningWithReplay,
        epsilons,
        learning_rates,
        num_trials,
        num_episodes,
        hidden_dim,
    )

    acrobot_sarsa_results = run_experiments(
        "Acrobot-v1",
        ExpectedSARSA,
        ExpectedSARSAWithReplay,
        epsilons,
        learning_rates,
        num_trials,
        num_episodes,
        hidden_dim,
    )

    # Plot Acrobot-v1 results
    print("Plotting results for Acrobot-v1...")
    plot_results(
        acrobot_q_results,
        acrobot_sarsa_results,
        epsilons,
        learning_rates,
        "Acrobot-v1",
        False,
    )
    plot_results(
        acrobot_q_results,
        acrobot_sarsa_results,
        epsilons,
        learning_rates,
        "Acrobot-v1",
        True,
    )

    # Run experiments for ALE/Assault-ram-v5
    print("Running experiments for ALE/Assault-ram-v5...")
    assault_q_results = run_experiments(
        "ALE/Assault-ram-v5",
        QLearning,
        QLearningWithReplay,
        epsilons,
        learning_rates,
        num_trials,
        num_episodes,
        hidden_dim,
    )

    assault_sarsa_results = run_experiments(
        "ALE/Assault-ram-v5",
        ExpectedSARSA,
        ExpectedSARSAWithReplay,
        epsilons,
        learning_rates,
        num_trials,
        num_episodes,
        hidden_dim,
    )

    # Plot ALE/Assault-ram-v5 results
    print("Plotting results for ALE/Assault-ram-v5...")
    plot_results(
        assault_q_results,
        assault_sarsa_results,
        epsilons,
        learning_rates,
        "ALE/Assault-ram-v5",
        False,
    )
    plot_results(
        assault_q_results,
        assault_sarsa_results,
        epsilons,
        learning_rates,
        "ALE/Assault-ram-v5",
        True,
    )


if __name__ == "__main__":
    main()
