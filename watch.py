"""
Load a trained model and watch it fly in a GUI window.

Run: python watch.py
"""

import time

from sb3_contrib import RecurrentPPO

from drone_soccer_env import DroneSoccerEnv

model = RecurrentPPO.load("drone_soccer_ppo_v0")

env = DroneSoccerEnv(render_mode="human")
obs, info = env.reset()

lstm_states = None
episode_start = True

for _ in range(1000):
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_start, deterministic=True
    )
    obs, reward, terminated, truncated, info = env.step(action)
    episode_start = terminated or truncated
    time.sleep(1 / 60)
    if episode_start:
        obs, info = env.reset()
        lstm_states = None

env.close()
