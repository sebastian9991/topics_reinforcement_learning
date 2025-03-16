import os
import random
from collections import deque
from re import template

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

gym.register_envs(ale_py)

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


class BoltzmannPolicy:
    def __init__(
        self,
        state_dim,
        action_dim,
        initial_temperature,
        min_temperature=0.1,
        decay_steps=500,
        hidden_dim=256,
    ):
        self.policy_net = MLP(state_dim, action_dim, hidden_dim)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optim.SGD([torch.tensor(self.temperature)], lr=initial_temperature),
            start_factor=1.0,
            end_factor=min_temperature / initial_temperature,
            total_iters=decay_steps,
        )

    def select_action(self, state, temperature=1.0):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(prob, num_samples=1).item()
        return action, prob[action]

    def decay_temperature(self):
        self.scheduler.step()
        self.temperature = max(
            self.min_temperature, self.scheduler.optimizer.param_groups[0]["lr"]
        )


class ActorCritic:
    def __init__(
        self,
        state_dim,
        action_dim,
        initial_temperature,
        temperature_decay,
        alpha_theta=0.01,
        alpha_w=0.01,
        gamma=0.99,
    ):
        self.gamma = gamma
        self.temperature_decay = temperature_decay
        self.actor = BoltzmannPolicy(
            state_dim, action_dim, initial_temperature=initial_temperature
        )
        self.critic = MLP(state_dim, 1)  # Our state-value function
        self.actor_optimizer = optim.Adam(
            self.actor.policy_net.parameters(), lr=alpha_theta
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha_w)

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # Compute the value estimates from the value-estimate critic network
        value = self.critic(state_tensor)
        next_value = (
            self.critic(next_state_tensor) if not done else 0.0
        )  # For the case of terminal states

        # TD error
        delta = reward_tensor + self.gamma * next_value - value

        # Update the critic (State-Value function)
        critic_loss = delta.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy Gradient Update
        _, prob = self.actor.select_action(state)
        policy_loss = -torch.log(prob) * delta.detach()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Decay temp

        if self.temperature_decay:
            self.actor.decay_temperature()


class Reinforce:
    def __init__(
        self,
        state_dim,
        action_dim,
        initial_temperature,
        temperature_decay,
        alpha_theta=0.01,
        gamma=0.99,
    ):
        self.gamma = gamma
        self.temperature_decay = temperature_decay
        self.actor = BoltzmannPolicy(state_dim, action_dim, initial_temperature)
        self.theta_optimizer = optim.Adam(
            self.actor.policy_net.parameters(), lr=alpha_theta
        )

    def update(self, trajectory):

        G = 0
        for t in reversed(range(len(trajectory))):
            state, _, reward = trajectory[t]
            G = reward + self.gamma * G
            _, prob = self.actor.select_action(state)
            loss = -torch.log(prob) * (self.gamma**t) * G
            self.theta_optimizer.zero_grad()
            loss.backward()
            self.theta_optimizer.step()

        if self.temperature_decay:
            self.actor.decay_temperature()


# Function to run a single trial with a specific algorithm and parameters
def run_trial(
    env_name,
    use_reinforce,
    algorithm_class,
    alpha_theta,
    alpha_w,
    initial_temperature,
    temperature_decay,
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
    if use_reinforce:
        agent = algorithm_class(
            state_dim,
            action_dim,
            alpha_theta=alpha_theta,
            alpha_w=alpha_w,
            initial_temperature=initial_temperature,
            temperature_decay=temperature_decay,
            hidden_dim=hidden_dim,
        )
    else:
        agent = algorithm_class(
            state_dim,
            action_dim,
            alpha_theta=alpha_theta,
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
        trajectory = []

        while not (done or truncated):
            # Choose action based on current state
            action = agent.actor.select_action(state)

            # Take action in environment - newer versions return (next_state, reward, done, truncated, info)
            step_result = env.step(action)
            if len(step_result) == 5:  # New gym API
                next_state, reward, done, truncated, _ = step_result
            else:  # Old gym API
                next_state, reward, done, _ = step_result
                truncated = False

            # Update agent's knowledge
            if use_reinforce:
                trajectory.append((state, action, reward))
            else:
                agent.update(state, action, reward, next_state, done or truncated)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

        if use_reinforce:
            agent.update(trajectory)
        # Store episode reward
        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


# Function to run experiments for a specific algorithm and environment
def run_experiments(
    env_name,
    use_reinforce,
    algorithm_class,
    alpha_theta,
    alpha_w,
    initial_temperature,
    temperature_decay,
    num_trials=50,
    num_episodes=1000,
    hidden_dim=256,
):
    results = {}

    key = f"env:{env_name}_class:{use_reinforce}_temperature:{initial_temperature}_w_decay:{temperature_decay}"
    results[key] = []

    for trial in tqdm(
        range(num_trials),
        desc=f"env:{env_name}_class:{use_reinforce}_temperature:{initial_temperature}_w_decay:{temperature_decay}",
    ):
        rewards = run_trial(
            env_name,
            use_reinforce,
            algorithm_class,
            alpha_theta,
            alpha_w,
            initial_temperature,
            temperature_decay,
            num_episodes,
            hidden_dim,
            seed=trial,
        )
        results[key].append(rewards)

    return results


def main():
    alpha_theta = 0.01
    alpha_w = 0.01
    initial_temperature = 2.0

    print("Running Acrobat-v1 experiement")
    print("Running Acrobot-v1 on Actor-Critic, Fixed Temp.")
    acrobat_actor_fixed_results = run_experiments(
        "Acrobot-v1",
        use_reinforce=False,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=False,
    )

    print("Running Acrobot-v1 on Actor-Critic, Decay Temp.")
    acrobat_actor_decay_results = run_experiments(
        "Acrobot-v1",
        use_reinforce=False,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=True,
    )

    print("Running Acrobot-v1 on REINFORCE, Fixed Temp.")
    acrobat_reinforce_fixed_results = run_experiments(
        "Acrobot-v1",
        use_reinforce=True,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=False,
    )

    print("Running Acrobot-v1 on REINFORCE, Decay Temp.")
    acrobat_reinforce_decay_results = run_experiments(
        "Acrobot-v1",
        use_reinforce=True,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=True,
    )

    print("Running ALE/Assault-ram-v5 experiement.")
    print("Running ALE/Assault-ram-v5 on Actor-Critic, Fixed Temp.")
    assault_actor_fixed_results = run_experiments(
        "ALE/Assault-ram-v5",
        use_reinforce=False,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=False,
    )
    print("Running ALE/Assault-ram-v5 on Actor-Critic, Decay Temp.")
    assault_actor_decay_results = run_experiments(
        "ALE/Assault-ram-v5",
        use_reinforce=False,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=True,
    )

    print("Running ALE/Assault-ram-v5 on REINFORCE, Fixed Temp.")
    assault_reinforce_fixed_results = run_experiments(
        "ALE/Assault-ram-v5",
        use_reinforce=True,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=False,
    )

    print("Running ALE/Assault-ram-v5 on REINFORCE, Decay Temp.")
    assault_reinforce_decay_reults = run_experiments(
        "ALE/Assault-ram-v5",
        use_reinforce=True,
        algorithm_class=ActorCritic,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        initial_temperature=initial_temperature,
        temperature_decay=True,
    )


if __name__ == "__main__":
    main()
