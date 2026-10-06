"""
Continuation training script: loads SOURCE_MODEL and trains it for
ADDITIONAL_TIMESTEPS more timesteps. See PROGRESS.md for the current model
lineage before changing these.

Naming convention: vX.0 = fresh start (train.py), vX.1/.2/... = later
stages on that lineage. X changes only for a fresh start under different
core physics/observation space; continuations on the same physics bump the
stage number. Update SOURCE_MODEL/OUTPUT_MODEL/tb_log_name/checkpoint+eval
names together before running, or it will overwrite or interleave with an
existing lineage.

Run:
    python train_continue.py
"""

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from train import ProgressPrintCallback, TerminationReasonCallback, make_env

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)

SOURCE_MODEL = "drone_soccer_ppo_v4.0"
OUTPUT_MODEL = "drone_soccer_ppo_v4.1"
# Diagnostic budget (not a full commitment) — v4.0's trend was volatile
# (oscillated 9-70% scored over its last 2M steps) and trained under the
# since-fixed defender-standoff bug (see PROGRESS.md), so check this actually
# climbs under clean signal before extending further.
ADDITIONAL_TIMESTEPS = 2_000_000


if __name__ == "__main__":
    N_ENVS = 12

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix=OUTPUT_MODEL,
    )

    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./checkpoints/best_model_{OUTPUT_MODEL.split('_')[-1]}/",
        log_path=f"./eval_logs_{OUTPUT_MODEL.split('_')[-1]}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,  # this policy's sampled actions score better than its mean
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
        tb_log_name=OUTPUT_MODEL.split("_")[-1],
    )
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
