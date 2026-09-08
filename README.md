# Multi-Agent Reinforcement Learning

A cooperative multi agent reinforcement learning project using a custom grid
environment and a shared PPO actor critic policy.

Every agent observes its own location, the direction of the nearest uncollected
target, progress, and remaining time. Agents receive shared rewards for target
collection, individual contribution bonuses, and collision penalties.

The implementation uses **parameter sharing**: all homogeneous agents act with
the same neural policy, while each agent contributes its own experience to the
PPO update. This is much more practical than training 100 completely separate
networks.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --episodes 500 --agents 4
python evaluate.py --agents 4
python plot_results.py
```

A larger experiment can use:

```bash
python train.py --episodes 1000 --agents 20 --grid-size 16 --max-steps 150
```

You can set `--agents 100`, but use a larger grid and expect training to take
significantly longer.

## Output

- `outputs/shared_ppo_policy.pt`
- `outputs/training_history.json`
- `outputs/training_curve.png`

This is an educational portfolio implementation of cooperative Independent PPO
with parameter sharing.
