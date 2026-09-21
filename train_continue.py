"""
Continuation training script: resumes the v5->v6 run from its last checkpoint
(11,211,200 steps) rather than restarting from v5 — that run wasn't flawed,
just running on the wrong device (see device="cpu" below), so no progress is
being discarded here, unlike earlier restarts in this project. Same reward as
v4->v5 (no changes) — checking whether shot accuracy keeps improving with
more training on the lateral shaping term before tuning MISS_PENALTY further.

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

SOURCE_MODEL = "checkpoints/drone_soccer_ppo_v5_continued_11211200_steps"
OUTPUT_MODEL = "drone_soccer_ppo_v6"
# v5->v6 originally targeted 13,000,000 total (v5's 9M + 4,000,000). Resuming
# from the 11,211,200-step checkpoint instead of restarting from v5, so only
# the remaining budget is left to run.
ADDITIONAL_TIMESTEPS = 13_000_000 - 11_211_200


if __name__ == "__main__":
    N_ENVS = 10

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # device="cpu": see train.py's device comment — benchmarked 1.29x faster
    # than the "auto"/cuda default for this tiny policy.
    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")
    # v5 already has ent_coef=0.001 baked in, so no override needed.

    # name_prefix distinguishes this from the v4->v5 run's checkpoints:
    # reset_num_timesteps=False means this run's step count picks up from
    # SOURCE_MODEL's, so with the same N_ENVS/n_steps, a shared prefix would
    # land on the exact same timestep numbers as the old run's files and
    # silently overwrite them.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="drone_soccer_ppo_v5_continued",
    )

    # See train.py's eval_callback comment: PPO can regress after its peak,
    # so this tracks the best-scoring checkpoint independently of whatever
    # the run happens to end on. deterministic=False for the same reason —
    # this policy's mean action underperforms its sampled distribution.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model_continued_v5/",
        log_path="./eval_logs_continued_v5/",
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
        # Distinct from "RecurrentPPO_v4_continued": reset_num_timesteps=False
        # makes SB3 log continuations into the latest existing run dir for a
        # given name rather than a fresh one — reusing the same name here
        # would interleave the v4->v5 and v5->v6 curves on one tensorboard
        # x-axis.
        tb_log_name="RecurrentPPO_v5_continued",
    )
    model.save(OUTPUT_MODEL)

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
