"""
Continuation training script: loads drone_soccer_ppo_v4 and trains it for
4,000,000 more timesteps, now with the LATERAL_SCALE final-approach shaping
term added in drone_soccer_env.py.

Run:
    python train_continue.py
"""

import time

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import DroneSoccerEnv
from train import ProgressPrintCallback, TerminationReasonCallback, make_env

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)

SOURCE_MODEL = "drone_soccer_ppo_v4"
OUTPUT_MODEL = "drone_soccer_ppo_v5"
ADDITIONAL_TIMESTEPS = 4_000_000


if __name__ == "__main__":
    N_ENVS = 10

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = RecurrentPPO.load(SOURCE_MODEL, env=env)
    # v4 already has ent_coef=0.001 baked in (set explicitly during the
    # v3->v4 run and preserved on save), so no override needed this time —
    # see git history if tuning ent_coef again.

    # name_prefix distinguishes this from the v3->v4 run's checkpoints:
    # reset_num_timesteps=False means this run's step count picks up from
    # SOURCE_MODEL's, so with the same N_ENVS/n_steps, a shared prefix would
    # land on the exact same timestep numbers as the old run's files and
    # silently overwrite them.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="drone_soccer_ppo_v4_continued",
    )

    # See train.py's eval_callback comment: PPO can regress after its peak,
    # so this tracks the best-scoring checkpoint independently of whatever
    # the run happens to end on. deterministic=False for the same reason —
    # this policy's mean action underperforms its sampled distribution.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model_continued_v4/",
        log_path="./eval_logs_continued_v4/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=CallbackList([
            ProgressPrintCallback(print_freq=100_000),
            TerminationReasonCallback(report_freq=50_000),
            checkpoint_callback,
            eval_callback,
        ]),
        reset_num_timesteps=False,
        # Distinct from "RecurrentPPO_v3_continued": reset_num_timesteps=False
        # makes SB3 log continuations into the latest existing run dir for a
        # given name rather than a fresh one — reusing the same name here
        # would interleave the v3->v4 and v4->v5 curves on one tensorboard
        # x-axis.
        tb_log_name="RecurrentPPO_v4_continued",
    )
    model.save(OUTPUT_MODEL)

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
