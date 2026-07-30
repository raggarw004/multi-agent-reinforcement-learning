from __future__ import annotations
import argparse
import numpy as np
from environment import CooperativeGridWorld
from ppo import IndependentPPO


def evaluate(model_path: str, episodes: int, num_agents: int, grid_size: int):
    environment = CooperativeGridWorld(
        num_agents=num_agents, grid_size=grid_size, seed=100
    )
    learner = IndependentPPO(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
    )
    learner.load(model_path)
    learner.model.eval()

    rewards = []
    successes = []

    for _ in range(episodes):
        observations = environment.reset()
        episode_reward = 0.0

        while True:
            actions, _, _ = learner.choose_actions(observations)
            result = environment.step(actions)
            episode_reward += float(result.rewards.mean())
            observations = result.observations

            if result.terminated:
                rewards.append(episode_reward)
                successes.append(float(result.info["all_collected"]))
                break

    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Success rate: {np.mean(successes):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/shared_ppo_policy.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.agents, args.grid_size)
