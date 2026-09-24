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

# Source from v7_best, NOT v6.zip or v7.zip — both of those regressed from a
# mid-run peak (see PROGRESS.md's "PPO's non-monotonic regression" note).
# v7_best is the actual best model found so far.
SOURCE_MODEL = "drone_soccer_ppo_v7_best"
OUTPUT_MODEL = "drone_soccer_ppo_v8"
ADDITIONAL_TIMESTEPS = 4_000_000


if __name__ == "__main__":
    N_ENVS = 10

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # device="cpu": see train.py's device comment — benchmarked 1.29x faster
    # than the "auto"/cuda default for this tiny policy.
    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")
    # v7_best already has ent_coef=0.001 baked in, so no override needed.

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
        name_prefix="drone_soccer_ppo_v7_best_continued",
    )

    # See train.py's eval_callback comment: PPO can regress after its peak,
    # so this tracks the best-scoring checkpoint independently of whatever
    # the run happens to end on. deterministic=False for the same reason —
    # this policy's mean action underperforms its sampled distribution.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model_continued_v7_best/",
        log_path="./eval_logs_continued_v7_best/",
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
        # Stage_1-4 = v3->v4/v4->v5/v5->v6/v6->v7 (see PROGRESS.md) — this
        # v7_best->v8 run is Stage_5. Bump this each time SOURCE_MODEL/
        # OUTPUT_MODEL move to the next pair, to keep tensorboard runs
        # matching the report. After this run finishes, check its final
        # save against checkpoints/best_model_continued_v7_best/best_model.zip
        # before trusting it — v6 and v7 both regressed from their mid-run
        # peak, v4 and v5 didn't, so this has to be checked every time, not
        # assumed either way.
        tb_log_name="Stage_5",
    )
    model.save(OUTPUT_MODEL)

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
