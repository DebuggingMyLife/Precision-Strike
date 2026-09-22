"""
Continuation training script: loads SOURCE_MODEL and trains it for
ADDITIONAL_TIMESTEPS more timesteps. See PROGRESS.md for the current model
lineage before changing these.

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

SOURCE_MODEL = "drone_soccer_ppo_v6"
OUTPUT_MODEL = "drone_soccer_ppo_v7"
ADDITIONAL_TIMESTEPS = 4_000_000


if __name__ == "__main__":
    N_ENVS = 10

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # device="cpu": see train.py's device comment — benchmarked 1.29x faster
    # than the "auto"/cuda default for this tiny policy.
    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")
    # v6 already has ent_coef=0.001 baked in, so no override needed.

    # name_prefix/tb_log_name/eval paths below must be unique per lineage
    # (bump the "v6" to match SOURCE_MODEL whenever you change it above):
    # reset_num_timesteps=False means this run's step count picks up from
    # SOURCE_MODEL's, so with the same N_ENVS/n_steps, a shared prefix/name
    # across different lineages would land on the exact same timestep
    # numbers and either overwrite old checkpoint files or interleave
    # unrelated runs' curves on one tensorboard x-axis. See PROGRESS.md.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="drone_soccer_ppo_v6_continued",
    )

    # See train.py's eval_callback comment: PPO can regress after its peak,
    # so this tracks the best-scoring checkpoint independently of whatever
    # the run happens to end on. deterministic=False for the same reason —
    # this policy's mean action underperforms its sampled distribution.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model_continued_v6/",
        log_path="./eval_logs_continued_v6/",
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
        # Stage_1/2/3 = v3->v4/v4->v5/v5->v6 (see PROGRESS.md) — this v6->v7
        # run is Stage_4. Bump this each time SOURCE_MODEL/OUTPUT_MODEL move
        # to the next pair, to keep tensorboard runs matching the report.
        tb_log_name="Stage_4",
    )
    model.save(OUTPUT_MODEL)

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
