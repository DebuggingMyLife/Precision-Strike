"""
Continuation training script: loads SOURCE_MODEL and trains it for
ADDITIONAL_TIMESTEPS more timesteps. See PROGRESS.md for the current model
lineage before changing these.

Naming convention: vX.0 = fresh start (train.py), vX.1/.2/... = later
stages continuing that lineage (this script). X only changes for a fresh
start under different core physics; continuations on the same physics bump
the stage number instead. Set below to continue v2.0 -> v2.1 once v2.0
finishes — update SOURCE_MODEL/OUTPUT_MODEL/tb_log_name/checkpoint+eval
names together before running, or it will silently overwrite or interleave
with an existing lineage (see the naming-collision finding in PROGRESS.md).

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

SOURCE_MODEL = "drone_soccer_ppo_v2.1"
OUTPUT_MODEL = "drone_soccer_ppo_v2.2"
ADDITIONAL_TIMESTEPS = 4_000_000


if __name__ == "__main__":
    N_ENVS = 10

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    # device="cpu": see train.py's device comment — benchmarked 1.29x faster
    # than the "auto"/cuda default for this tiny policy.
    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")

    # name_prefix/tb_log_name/eval paths below must be unique per lineage —
    # reset_num_timesteps=False means this run's step count picks up from
    # SOURCE_MODEL's, so with the same N_ENVS/n_steps, a shared prefix/name
    # across different lineages would land on the exact same timestep
    # numbers and either overwrite old checkpoint files or interleave
    # unrelated runs' curves on one tensorboard x-axis. See PROGRESS.md.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="drone_soccer_ppo_v2.2",
    )

    # See train.py's eval_callback comment: PPO can regress after its peak,
    # so this tracks the best-scoring checkpoint independently of whatever
    # the run happens to end on. deterministic=False for the same reason —
    # this policy's mean action underperforms its sampled distribution.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model_v2.2/",
        log_path="./eval_logs_v2.2/",
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
        # After this run finishes, check its final save against
        # checkpoints/best_model_v2.2/best_model.zip with a matched
        # eval_model.py comparison before trusting it — v1.3/v1.4 both
        # regressed from their mid-run peak by the final save, v1.1/v1.2
        # didn't, so this has to be checked every time, not assumed either
        # way. Also remember to evaluate any older model in a comparison
        # under its own native ATTITUDE_GAIN, not whatever the current file
        # says (see the cross-lineage eval bug finding in PROGRESS.md).
        tb_log_name="v2.2",
    )
    # See train.py's comment on this same pattern: SB3's save() misreads a
    # dotted name's trailing ".N" as an existing extension and skips adding
    # ".zip", so pass it explicitly.
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
