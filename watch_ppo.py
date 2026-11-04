import glob
import os
import sys
import time

from stable_baselines3 import PPO

from drone_soccer_env import DroneSoccerEnv
from train_ppo import PPO_ENV_KWARGS
from yolo_detector import YoloDetector


def find_latest_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = (glob.glob("checkpoints/drone_soccer_plainppo_*.zip")
                  + glob.glob("drone_soccer_plainppo_v*.zip"))
    if not candidates:
        raise FileNotFoundError("No plainppo model .zip found. Train first.")
    return max(candidates, key=os.path.getmtime)


model_path = find_latest_model()
print(f"Loading {model_path}")
model = PPO.load(model_path)

env = DroneSoccerEnv(
    detector=YoloDetector(), render_mode="human", chase_probability=1.0, **PPO_ENV_KWARGS
)
obs, _ = env.reset()

episode_num = 1

for _ in range(5000):
    action, _ = model.predict(obs, deterministic=False)
    obs, _, terminated, truncated, info = env.step(action)
    time.sleep(1 / 60)
    if terminated or truncated:
        reason = info.get("termination_reason")
        print(f"[episode {episode_num}] ended: {reason}")
        env.show_termination_text(str(reason))
        episode_num += 1
        obs, _ = env.reset()

env.close()
