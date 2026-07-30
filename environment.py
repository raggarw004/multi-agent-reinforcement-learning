from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ACTIONS = {
    0: np.array([0, 0], dtype=np.int64),   # stay
    1: np.array([-1, 0], dtype=np.int64),  # up
    2: np.array([1, 0], dtype=np.int64),   # down
    3: np.array([0, -1], dtype=np.int64),  # left
    4: np.array([0, 1], dtype=np.int64),   # right
}


@dataclass
class StepResult:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: bool
    info: dict


class CooperativeGridWorld:
    """Agents cooperate to cover targets in a shared grid."""

    def __init__(
        self,
        num_agents: int = 4,
        grid_size: int = 8,
        max_steps: int = 80,
        seed: int = 42,
    ):
        if num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((num_agents, 2), dtype=np.int64)
        self.targets = np.zeros((num_agents, 2), dtype=np.int64)
        self.collected = np.zeros(num_agents, dtype=bool)
        self.steps = 0

    @property
    def observation_size(self) -> int:
        # own position (2), nearest target delta (2), collected ratio (1), time (1)
        return 6

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    def _random_positions(self, count: int) -> np.ndarray:
        cells = self.rng.choice(self.grid_size * self.grid_size, size=count, replace=False)
        return np.stack((cells // self.grid_size, cells % self.grid_size), axis=1)

    def reset(self) -> np.ndarray:
        cells = self._random_positions(self.num_agents * 2)
        self.positions = cells[: self.num_agents].copy()
        self.targets = cells[self.num_agents :].copy()
        self.collected[:] = False
        self.steps = 0
        return self._observations()

    def _observations(self) -> np.ndarray:
        observations = []
        scale = max(self.grid_size - 1, 1)

        for agent_index in range(self.num_agents):
            position = self.positions[agent_index]
            remaining = self.targets[~self.collected]

            if len(remaining):
                distances = np.abs(remaining - position).sum(axis=1)
                nearest = remaining[int(np.argmin(distances))]
                delta = (nearest - position) / scale
            else:
                delta = np.zeros(2, dtype=np.float32)

            observation = np.concatenate([
                position.astype(np.float32) / scale,
                delta.astype(np.float32),
                np.array([
                    self.collected.mean(),
                    self.steps / self.max_steps,
                ], dtype=np.float32),
            ])
            observations.append(observation)

        return np.asarray(observations, dtype=np.float32)

    def step(self, actions: np.ndarray) -> StepResult:
        if len(actions) != self.num_agents:
            raise ValueError("One action is required for every agent")

        old_positions = self.positions.copy()
        for index, action in enumerate(actions):
            movement = ACTIONS[int(action)]
            self.positions[index] = np.clip(
                self.positions[index] + movement, 0, self.grid_size - 1
            )

        rewards = np.full(self.num_agents, -0.01, dtype=np.float32)

        # Penalize collisions to encourage coordination.
        unique, counts = np.unique(self.positions, axis=0, return_counts=True)
        collision_cells = unique[counts > 1]
        for cell in collision_cells:
            colliding = np.all(self.positions == cell, axis=1)
            rewards[colliding] -= 0.08

        newly_collected = 0
        for target_index, target in enumerate(self.targets):
            if self.collected[target_index]:
                continue
            visitors = np.all(self.positions == target, axis=1)
            if visitors.any():
                self.collected[target_index] = True
                newly_collected += 1
                rewards += 0.25          # shared team reward
                rewards[visitors] += 0.75  # individual contribution bonus

        self.steps += 1
        all_collected = bool(self.collected.all())
        timed_out = self.steps >= self.max_steps
        terminated = all_collected or timed_out

        if all_collected:
            rewards += 1.0

        return StepResult(
            observations=self._observations(),
            rewards=rewards,
            terminated=terminated,
            info={
                "targets_collected": int(self.collected.sum()),
                "newly_collected": newly_collected,
                "all_collected": all_collected,
                "moved_agents": int(np.any(old_positions != self.positions, axis=1).sum()),
            },
        )
