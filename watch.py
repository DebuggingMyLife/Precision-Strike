"""
Load a trained model and watch it fly in a GUI window.

By default, loads whichever model .zip (checkpoints/ or a final
drone_soccer_ppo_v*.zip) was written most recently — so you can check in on
a run that's still training without waiting for it to finish. Pass a path
explicitly to watch a specific one instead.

Run:
    python watch.py
    python watch.py checkpoints/drone_soccer_ppo_500000_steps.zip
"""

import glob
import os
import sys
import time

from sb3_contrib import RecurrentPPO

from drone_soccer_env import DroneSoccerEnv


def find_latest_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = glob.glob("checkpoints/*.zip") + glob.glob("drone_soccer_ppo_v*.zip")
    if not candidates:
        raise FileNotFoundError(
            "No model .zip found in checkpoints/ or drone_soccer_ppo_v*.zip. Train first."
        )
    return max(candidates, key=os.path.getmtime)


model_path = find_latest_model()
print(f"Loading {model_path}")
model = RecurrentPPO.load(model_path)

env = DroneSoccerEnv(render_mode="human")
obs, info = env.reset()

lstm_states = None
episode_start = True
episode_num = 1

for _ in range(5000):
    # deterministic=False on purpose: this policy's mean action alone
    # flips almost every episode, but sampling from its actual learned
    # distribution (what training itself used) is what actually scores.
    # deterministic=True here would show a misleadingly broken drone.
    # (The ~70% figure once quoted here was from the old single-opponent
    # env's v2 model — see PROGRESS.md for current scored-rate numbers.)
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_start, deterministic=False
    )
    obs, reward, terminated, truncated, info = env.step(action)
    episode_start = terminated or truncated
    time.sleep(1 / 60)
    if episode_start:
        reason = info.get("termination_reason")
        print(f"[episode {episode_num}] ended: {reason}")
        env.show_termination_text(str(reason))
        episode_num += 1
        obs, info = env.reset()
        lstm_states = None

env.close()
