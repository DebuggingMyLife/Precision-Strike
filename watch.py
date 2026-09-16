"""
Watch a trained model act in a live PyBullet GUI window.

Runs episodes on repeat, keeping the window open so you can freely
rotate/zoom/pan with the mouse. Press Ctrl+C in the terminal to stop
and close the window.

Run:
    python watch.py
"""

from sb3_contrib import RecurrentPPO

from drone_soccer_env import DroneSoccerEnv


if __name__ == "__main__":
    model = RecurrentPPO.load("drone_soccer_ppo_v0")
    env = DroneSoccerEnv(detector=None, render_mode="human")

    obs, info = env.reset()
    lstm_states = None
    episode_start = True

    try:
        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_start = terminated or truncated
            if episode_start:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()
