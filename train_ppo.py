import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import CHASE_PROBABILITY, DroneSoccerEnv
from train import (
    ChaseCurriculumCallback,
    ProgressPrintCallback,
    TerminationReasonCallback,
)

# Plain (non-recurrent) PPO baseline for comparison against RecurrentPPO
OUTPUT_MODEL = "drone_soccer_plainppo_v1.0"

# Defender settings the plain PPO run was trained with (the RecurrentPPO env uses defaults)
PPO_ENV_KWARGS = dict(opponent_speed_mps=1.5, randomize_defender_spawn=True)

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)


def make_env(chase_probability=CHASE_PROBABILITY):
    def _init():
        from yolo_detector import YoloDetector
        env = DroneSoccerEnv(
            detector=YoloDetector(), chase_probability=chase_probability, **PPO_ENV_KWARGS
        )
        return Monitor(env)
    return _init


if __name__ == "__main__":
    N_ENVS = 12
    TOTAL_TIMESTEPS = 12_000_000
    CHASE_CURRICULUM_END_STEP = 3_500_000

    env = SubprocVecEnv([make_env(chase_probability=0.0) for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.001,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix=OUTPUT_MODEL,
    )

    eval_callback = EvalCallback(
        DummyVecEnv([make_env()]),
        best_model_save_path=f"./checkpoints/best_model_{OUTPUT_MODEL}/",
        log_path=f"./eval_logs_{OUTPUT_MODEL}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([
            ProgressPrintCallback(),
            TerminationReasonCallback(),
            ChaseCurriculumCallback(end_step=CHASE_CURRICULUM_END_STEP),
            checkpoint_callback,
            eval_callback,
        ]),
        tb_log_name="plainppo_v1.0",
    )
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Training complete. Model saved to {OUTPUT_MODEL}.zip")
