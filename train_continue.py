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

# Update both names before each new stage to avoid overwriting runs
SOURCE_MODEL = "drone_soccer_ppo_v4.0"
OUTPUT_MODEL = "drone_soccer_ppo_v4.1"
ADDITIONAL_TIMESTEPS = 2_000_000


if __name__ == "__main__":
    N_ENVS = 12
    version = OUTPUT_MODEL.split("_")[-1]

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    model = RecurrentPPO.load(SOURCE_MODEL, env=env, device="cpu")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix=OUTPUT_MODEL,
    )

    eval_callback = EvalCallback(
        DummyVecEnv([make_env()]),
        best_model_save_path=f"./checkpoints/best_model_{version}/",
        log_path=f"./eval_logs_{version}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=CallbackList([
            ProgressPrintCallback(),
            TerminationReasonCallback(),
            checkpoint_callback,
            eval_callback,
        ]),
        reset_num_timesteps=False,
        tb_log_name=version,
    )
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Continuation training complete. Model saved to {OUTPUT_MODEL}.zip")
