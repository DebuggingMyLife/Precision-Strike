"""
Fresh-start training script for the drone soccer striker task using
plain (non-recurrent) PPO, for comparison against RecurrentPPO. See
train_continue.py to continue an existing model instead.

Run:
    python train.py
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import CHASE_PROBABILITY, DroneSoccerEnv

# v4.0: fourth lineage — single defender (was goalkeeper + 3 fliers) and a
# required YOLO detector (was optional, defaulting to ground-truth opponent
# state), both breaking observation-space compatibility with every prior
# checkpoint. See PROGRESS.md for the full history this naming convention
# tracks.
# plainppo: separate lineage from the RecurrentPPO models (drone_soccer_ppo_v*),
# same env and hyperparameters but no LSTM, so the two can be compared directly.
OUTPUT_MODEL = "drone_soccer_plainppo_v1.0"

torch.set_float32_matmul_precision("high")  # free speed, RL gradients are noisy anyway
torch.set_num_threads(1)  # leave cores free for the env subprocesses


def make_env(render_mode=None, chase_probability=CHASE_PROBABILITY):
    def _init():
        from yolo_detector import YoloDetector  # loaded per-subprocess, not pickled in
        env = DroneSoccerEnv(
            detector=YoloDetector(), render_mode=render_mode,
            chase_probability=chase_probability,
        )
        return Monitor(env)  # Monitor is what lets SB3 log ep_rew_mean/ep_len_mean
    return _init


class ChaseCurriculumCallback(BaseCallback):
    """Ramps the training envs' chase_probability linearly from 0 at step 0
    to `end_prob` by `end_step`, instead of exposing a trained-from-scratch
    policy to an interceptor immediately — see PROGRESS.md's v3.0_abandoned
    entries for what happens when that's skipped."""

    def __init__(self, end_step, end_prob=CHASE_PROBABILITY, update_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.end_step = end_step
        self.end_prob = end_prob
        self.update_freq = update_freq
        self._last_update = -update_freq  # force an update on the first call

    def _on_step(self):
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            progress = min(1.0, self.num_timesteps / self.end_step)
            prob = progress * self.end_prob
            self.training_env.env_method("set_chase_probability", prob)
            self.logger.record("curriculum/chase_probability", prob)
        return True


class ProgressPrintCallback(BaseCallback):
    """Prints a progress line every `print_freq` timesteps."""

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
    """Logs the breakdown of why episodes end (from info["termination_reason"])
    every `report_freq` timesteps, to tensorboard and stdout."""

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
    N_ENVS = 12  # one process per env
    TOTAL_TIMESTEPS = 5_000_000
    CHASE_CURRICULUM_END_STEP = int(TOTAL_TIMESTEPS * 0.7)  # full difficulty by 70% through

    # Training envs start at chase_probability=0 (pure wander); the eval env
    # below stays at the CHASE_PROBABILITY default so EvalCallback measures
    # real target-difficulty performance throughout, not a moving target.
    env = SubprocVecEnv([make_env(chase_probability=0.0) for _ in range(N_ENVS)])

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.001,  # keeps exploration alive without causing std runaway
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cpu",  # benchmarked faster than cuda for this small a policy
    )

    # save_freq/eval_freq count rollout steps, not env timesteps, so divide
    # by N_ENVS to land checkpoints/evals every ~50k real timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix=OUTPUT_MODEL,
    )

    # Tracks the best-scoring checkpoint independently of whatever the run
    # happens to end on, since PPO isn't monotonic. deterministic=False
    # since this policy's sampled actions score better than its mean action.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./checkpoints/best_model_{OUTPUT_MODEL}/",
        log_path=f"./eval_logs_{OUTPUT_MODEL}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([
            ProgressPrintCallback(print_freq=100_000),
            TerminationReasonCallback(report_freq=50_000),
            ChaseCurriculumCallback(end_step=CHASE_CURRICULUM_END_STEP),
            checkpoint_callback,
            eval_callback,
        ]),
        tb_log_name="plainppo_v1.0",
    )
    # SB3's save() misreads a dotted name's trailing ".N" as an existing
    # extension, so ".zip" must be passed explicitly.
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Training complete. Model saved to {OUTPUT_MODEL}.zip")
