"""
Load a trained model and watch it fly in a GUI window.

By default, loads whichever model .zip (checkpoints/ or a final
drone_soccer_plainppo_v*.zip) was written most recently — so you can check in on
a run that's still training without waiting for it to finish. Pass a path
explicitly to watch a specific one instead.

Run:
    python watch.py
    python watch.py checkpoints/drone_soccer_plainppo_v1.0_500000_steps.zip
"""

import glob
import os
import sys
import time

from stable_baselines3 import PPO

from drone_soccer_env import DroneSoccerEnv
from yolo_detector import YoloDetector


def find_latest_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    # plainppo only: RecurrentPPO checkpoints in the same folders won't load with PPO.
    candidates = (glob.glob("checkpoints/drone_soccer_plainppo_*.zip")
                  + glob.glob("drone_soccer_plainppo_v*.zip"))
    if not candidates:
        raise FileNotFoundError(
            "No plainppo model .zip found in checkpoints/ or drone_soccer_plainppo_v*.zip. Train first."
        )
    return max(candidates, key=os.path.getmtime)


model_path = find_latest_model()
print(f"Loading {model_path}")
model = PPO.load(model_path)

env = DroneSoccerEnv(detector=YoloDetector(), render_mode="human", chase_probability=1.0)
obs, info = env.reset()

episode_num = 1

for _ in range(5000):
    # deterministic=False: this policy's sampled actions score far better
    # than its mean action, so deterministic=True would look misleadingly broken.
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1 / 60)
    if terminated or truncated:
        reason = info.get("termination_reason")
        print(f"[episode {episode_num}] ended: {reason}")
        env.show_termination_text(str(reason))
        episode_num += 1
        obs, info = env.reset()

env.close()
