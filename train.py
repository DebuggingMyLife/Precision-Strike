import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import CHASE_PROBABILITY, DroneSoccerEnv

OUTPUT_MODEL = "drone_soccer_ppo_v4.0"

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)  # leave cores for env subprocesses


def make_env(chase_probability=CHASE_PROBABILITY):
    def _init():
        from yolo_detector import YoloDetector  # loaded per subprocess
        env = DroneSoccerEnv(detector=YoloDetector(), chase_probability=chase_probability)
        return Monitor(env)
    return _init


class ChaseCurriculumCallback(BaseCallback):
    """Ramps chase_probability linearly from 0 to end_prob by end_step."""

    def __init__(self, end_step, end_prob=CHASE_PROBABILITY, update_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.end_step = end_step
        self.end_prob = end_prob
        self.update_freq = update_freq
        self._last_update = -update_freq  # update on first call

    def _on_step(self):
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            prob = min(1.0, self.num_timesteps / self.end_step) * self.end_prob
            self.training_env.env_method("set_chase_probability", prob)
            self.logger.record("curriculum/chase_probability", prob)
        return True


class ProgressPrintCallback(BaseCallback):
    """Prints a progress line every print_freq timesteps."""

    def __init__(self, print_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self):
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            print(f"[progress] timestep {self.num_timesteps}")
        return True


class TerminationReasonCallback(BaseCallback):
    """Logs the termination_reason breakdown every report_freq timesteps."""

    def __init__(self, report_freq=50_000, verbose=0):
        super().__init__(verbose)
        self.report_freq = report_freq
        self._last_report = 0
        self._counts = {}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            reason = info.get("termination_reason")
            if reason is not None:
                self._counts[reason] = self._counts.get(reason, 0) + 1

        if self.num_timesteps - self._last_report >= self.report_freq:
            self._last_report = self.num_timesteps
            total = sum(self._counts.values())
            if total:
                breakdown = ", ".join(
                    f"{k}={v} ({100 * v / total:.0f}%)"
                    for k, v in sorted(self._counts.items())
                )
                print(f"[episode ends] timestep {self.num_timesteps}: {breakdown}")
                for k, v in self._counts.items():
                    self.logger.record(f"episode_end/{k}", v)
            self._counts = {}
        return True


if __name__ == "__main__":
    N_ENVS = 12
    TOTAL_TIMESTEPS = 5_000_000
    CHASE_CURRICULUM_END_STEP = int(TOTAL_TIMESTEPS * 0.7)

    # Training starts with no chasing; eval env uses full difficulty
    env = SubprocVecEnv([make_env(chase_probability=0.0) for _ in range(N_ENVS)])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.001,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cpu",  # faster than cuda for this small policy
    )

    # Frequencies are per vec-env step, so ~50k timesteps
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
        deterministic=False,  # sampled actions outperform the mean action
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
        tb_log_name="v4.0",
    )
    # Explicit .zip: SB3 treats a dotted name's ".0" as an extension
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Training complete. Model saved to {OUTPUT_MODEL}.zip")
