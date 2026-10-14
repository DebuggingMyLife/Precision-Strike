import glob
import os
import sys
import time

from sb3_contrib import RecurrentPPO

from drone_soccer_env import DroneSoccerEnv
from yolo_detector import YoloDetector


def find_latest_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = glob.glob("checkpoints/*.zip") + glob.glob("drone_soccer_ppo_v*.zip")
    if not candidates:
        raise FileNotFoundError("No model .zip found. Train first.")
    return max(candidates, key=os.path.getmtime)


model_path = find_latest_model()
print(f"Loading {model_path}")
model = RecurrentPPO.load(model_path)

env = DroneSoccerEnv(detector=YoloDetector(), render_mode="human", chase_probability=1.0)
obs, _ = env.reset()

lstm_states = None
episode_start = True
episode_num = 1

for _ in range(5000):
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_start, deterministic=False
    )
    obs, _, terminated, truncated, info = env.step(action)
    episode_start = terminated or truncated
    time.sleep(1 / 60)
    if episode_start:
        reason = info.get("termination_reason")
        print(f"[episode {episode_num}] ended: {reason}")
        env.show_termination_text(str(reason))
        episode_num += 1
        obs, _ = env.reset()
        lstm_states = None

env.close()
