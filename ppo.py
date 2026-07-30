from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor):
        features = self.backbone(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, observations: torch.Tensor):
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values


@dataclass
class Rollout:
    observations: list
    actions: list
    log_probabilities: list
    rewards: list
    dones: list
    values: list

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [], [])


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    next_value = torch.zeros(rewards.shape[1], device=rewards.device)

    for time_index in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[time_index]
        delta = rewards[time_index] + gamma * next_value * non_terminal - values[time_index]
        last_advantage = (
            delta + gamma * gae_lambda * non_terminal * last_advantage
        )
        advantages[time_index] = last_advantage
        next_value = values[time_index]

    returns = advantages + values
    return advantages, returns


class IndependentPPO:
    """One shared policy is trained from all homogeneous agents' experiences."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 4,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ActorCritic(observation_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs

    @torch.no_grad()
    def choose_actions(self, observations: np.ndarray):
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions, log_probabilities, values = self.model.act(tensor)
        return (
            actions.cpu().numpy(),
            log_probabilities.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(self, rollout: Rollout) -> dict[str, float]:
        observations = torch.as_tensor(
            np.asarray(rollout.observations), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(rollout.actions), dtype=torch.long, device=self.device
        )
        old_log_probabilities = torch.as_tensor(
            np.asarray(rollout.log_probabilities), dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            np.asarray(rollout.rewards), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            np.asarray(rollout.dones), dtype=torch.float32, device=self.device
        )
        values = torch.as_tensor(
            np.asarray(rollout.values), dtype=torch.float32, device=self.device
        )

        advantages, returns = compute_gae(
            rewards, dones, values, self.gamma, self.gae_lambda
        )

        # Flatten time and agent dimensions.
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1)
        old_log_probabilities = old_log_probabilities.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {}
        for _ in range(self.update_epochs):
            logits, predicted_values = self.model(observations)
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            unclipped = ratio * advantages
            clipped = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(predicted_values, returns)
            loss = (
                policy_loss
                + self.value_coefficient * value_loss
                - self.entropy_coefficient * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }

        return metrics

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
