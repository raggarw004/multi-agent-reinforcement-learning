import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

history_path = Path("outputs/training_history.json")
if not history_path.exists():
    raise FileNotFoundError("Run train.py before plotting results.")

history = json.loads(history_path.read_text(encoding="utf-8"))
rewards = np.asarray(history["episode_rewards"], dtype=float)

window = min(25, len(rewards))
moving_average = np.convolve(rewards, np.ones(window) / window, mode="valid")

plt.figure(figsize=(9, 5))
plt.plot(range(window, len(rewards) + 1), moving_average)
plt.xlabel("Episode")
plt.ylabel(f"Average team reward ({window}-episode window)")
plt.title("Multi-Agent PPO Training")
plt.tight_layout()
Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/training_curve.png", dpi=160)
plt.show()
