"""
Starter training script for the drone soccer striker task using RecurrentPPO.

Run:
    python train.py

Once drone_soccer_env.py has real dynamics/rewards filled in, this script
should work as-is. Swap in your trained detector by passing it to
DroneSoccerEnv(detector=your_model).
"""

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import DroneSoccerEnv


def make_env():
    def _init():
        return DroneSoccerEnv(detector=None, render_mode=None)
    return _init


if __name__ == "__main__":
    N_ENVS = 4  # bump this up once you move to a machine with more cores/GPU

    # Use SubprocVecEnv for real parallelism once envs are stable;
    # DummyVecEnv is easier to debug (single process, clearer tracebacks).
    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=100_000)
    model.save("drone_soccer_ppo_v0")

    print("Training complete. Model saved to drone_soccer_ppo_v0.zip")
