"""
Starter training script for the drone soccer striker task using RecurrentPPO.

Run:
    python train.py

Once drone_soccer_env.py has real dynamics/rewards filled in, this script
should work as-is. Swap in your trained detector by passing it to
DroneSoccerEnv(detector=your_model).
"""

import time

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import DroneSoccerEnv

# TF32 matmuls: free precision-for-speed tradeoff on Ampere+/Blackwell GPUs,
# and RL gradients are noisy enough that the reduced precision doesn't matter.
torch.set_float32_matmul_precision("high")

# The policy itself is tiny (MLP+LSTM on a 14-number observation), so it
# doesn't need many CPU threads; leave the cores free for the env subprocesses
# instead of letting this main process's torch ops compete for them.
torch.set_num_threads(1)


def make_env(render_mode=None):
    def _init():
        return DroneSoccerEnv(detector=None, render_mode=render_mode)
    return _init


class LivePreviewCallback(BaseCallback):
    """Every `preview_freq` timesteps, open a GUI window and fly the current
    policy live for a bit, so you can watch training progress over time.

    Training itself stays headless (fast) across all N_ENVS; this uses a
    separate, one-off env instance with the model's live weights.
    """

    def __init__(self, preview_freq=50_000, preview_steps=300, verbose=0):
        super().__init__(verbose)
        self.preview_freq = preview_freq
        self.preview_steps = preview_steps
        self._last_preview = 0

    def _on_step(self):
        if self.num_timesteps - self._last_preview >= self.preview_freq:
            self._last_preview = self.num_timesteps
            self._run_preview()
        return True

    def _run_preview(self):
        print(f"\n[preview] timestep {self.num_timesteps}: opening GUI window...")
        preview_env = DroneSoccerEnv(detector=None, render_mode="human")
        obs, _ = preview_env.reset()
        lstm_states = None
        episode_start = True
        for _ in range(self.preview_steps):
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = preview_env.step(action)
            episode_start = terminated or truncated
            time.sleep(1 / 60)
            if episode_start:
                obs, _ = preview_env.reset()
                lstm_states = None
        preview_env.close()


if __name__ == "__main__":
    N_ENVS = 8  # one process per env; tune to your CPU core count

    # SubprocVecEnv runs each env in its own process for real CPU parallelism
    # (PyBullet's physics stepping is CPU-bound, independent of GPU training).
    # Swap back to DummyVecEnv if you need simpler tracebacks while debugging.
    # All training envs stay headless for speed — SB3's VecEnv requires every
    # sub-env to share one render_mode, so a mixed GUI/headless vec env isn't
    # possible. LivePreviewCallback below shows progress instead.
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    policy_kwargs = dict(
        lstm_hidden_size=128,  # Increases memory capacity for trajectory tracking
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # Separate, clean MLP layers before/after the LSTM
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=512,
        batch_size=256,  # bigger batches -> fewer, better-utilized GPU updates
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,  
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=LivePreviewCallback(preview_freq=500_000, preview_steps=300),
    )
    model.save("drone_soccer_ppo_v0")

    print("Training complete. Model saved to drone_soccer_ppo_v0.zip")
