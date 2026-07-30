from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch

from environment import CooperativeGridWorld
from ppo import IndependentPPO, Rollout


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    episodes: int = 500,
    num_agents: int = 4,
    grid_size: int = 8,
    max_steps: int = 80,
    seed: int = 42,
):
    set_seed(seed)
    environment = CooperativeGridWorld(
        num_agents=num_agents,
        grid_size=grid_size,
        max_steps=max_steps,
        seed=seed,
    )
    learner = IndependentPPO(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
    )

    episode_rewards = []
    success_history = []
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    for episode in range(1, episodes + 1):
        observations = environment.reset()
        rollout = Rollout.empty()
        total_reward = 0.0
        success = False

        while True:
            actions, log_probabilities, values = learner.choose_actions(observations)
            result = environment.step(actions)

            rollout.observations.append(observations)
            rollout.actions.append(actions)
            rollout.log_probabilities.append(log_probabilities)
            rollout.rewards.append(result.rewards)
            rollout.dones.append(
                np.full(num_agents, float(result.terminated), dtype=np.float32)
            )
            rollout.values.append(values)

            total_reward += float(result.rewards.mean())
            observations = result.observations
            success = result.info["all_collected"]

            if result.terminated:
                break

        metrics = learner.update(rollout)
        episode_rewards.append(total_reward)
        success_history.append(float(success))

        if episode % 25 == 0:
            average_reward = np.mean(episode_rewards[-25:])
            success_rate = np.mean(success_history[-25:])
            print(
                f"Episode {episode:4d} | average reward={average_reward:7.3f} | "
                f"success rate={success_rate:5.1%} | loss={metrics['loss']:.4f}"
            )

    learner.save(str(output_directory / "shared_ppo_policy.pt"))
    history = {
        "episodes": episodes,
        "num_agents": num_agents,
        "grid_size": grid_size,
        "episode_rewards": episode_rewards,
        "success_history": success_history,
    }
    (output_directory / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print("Saved model to outputs/shared_ppo_policy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cooperative multi-agent PPO")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        num_agents=args.agents,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
    )
