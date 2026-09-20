"""
Starter training script for the drone soccer striker task using RecurrentPPO.

Run:
    python train.py

Once drone_soccer_env.py has real dynamics/rewards filled in, this script
should work as-is. Swap in your trained detector by passing it to
DroneSoccerEnv(detector=your_model).
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_soccer_env import DroneSoccerEnv

# TF32 matmuls: free precision-for-speed tradeoff on Ampere+/Blackwell GPUs,
# and RL gradients are noisy enough that the reduced precision doesn't matter.
torch.set_float32_matmul_precision("high")

# The policy itself is tiny (MLP+LSTM on a 14-number observation), so it
# doesn't need many CPU threads; leave the cores free for the env subprocesses
# instead of letting this main process's torch ops compete for them.
torch.set_num_threads(1)


def make_env(render_mode=None):
    def _init():
        # Monitor tracks per-episode reward/length so SB3 can log
        # rollout/ep_rew_mean to tensorboard — without it there's no way to
        # tell whether the policy is actually improving at the task.
        return Monitor(DroneSoccerEnv(detector=None, render_mode=render_mode))
    return _init


class LivePreviewCallback(BaseCallback):
    """Every `preview_freq` timesteps, open a GUI window and fly the current
    policy live for a bit, so you can watch training progress over time.

    Training itself stays headless (fast) across all N_ENVS; this uses a
    separate, one-off env instance with the model's live weights.
    """

    def __init__(self, preview_freq=50_000, preview_steps=300, verbose=0):
        super().__init__(verbose)
        self.preview_freq = preview_freq
        self.preview_steps = preview_steps
        self._last_preview = 0

    def _on_step(self):
        if self.num_timesteps - self._last_preview >= self.preview_freq:
            self._last_preview = self.num_timesteps
            self._run_preview()
        return True

    def _run_preview(self):
        print(f"\n[preview] timestep {self.num_timesteps}: opening GUI window...")
        preview_env = DroneSoccerEnv(detector=None, render_mode="human")
        obs, _ = preview_env.reset()
        lstm_states = None
        episode_start = True
        for _ in range(self.preview_steps):
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = preview_env.step(action)
            episode_start = terminated or truncated
            time.sleep(1 / 60)
            if episode_start:
                obs, _ = preview_env.reset()
                lstm_states = None
        preview_env.close()


class ProgressPrintCallback(BaseCallback):
    """Prints a lightweight progress line every `print_freq` timesteps."""

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
    """Tracks why episodes end (crashed/flipped/scored/missed_goal/
    out_of_bounds/max_steps, from drone_soccer_env's info["termination_reason"])
    and reports the breakdown every `report_freq` timesteps — both printed
    and logged to tensorboard, so you can see e.g. the flipped rate drop or
    the scored rate climb over training, not just the aggregate reward.
    """

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
    N_ENVS = 10  # one process per env; tune to your CPU core count

    # SubprocVecEnv runs each env in its own process for real CPU parallelism
    # (PyBullet's physics stepping is CPU-bound, independent of GPU training).
    # Swap back to DummyVecEnv if you need simpler tracebacks while debugging.
    # All training envs stay headless for speed — SB3's VecEnv requires every
    # sub-env to share one render_mode, so a mixed GUI/headless vec env isn't
    # possible. LivePreviewCallback below shows progress instead.
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    policy_kwargs = dict(
        lstm_hidden_size=128,  # Increases memory capacity for trajectory tracking
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # Separate, clean MLP layers before/after the LSTM
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=512,
        batch_size=256,  # bigger batches -> fewer, better-utilized GPU updates
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        # Defaults to 0.0 (no entropy bonus). With a fully deterministic env
        # (fixed spawn/goal/opponent phase each episode), a policy with no
        # exploration pressure can lock onto one repeated trajectory and
        # stop improving — which is exactly what happened last run. A small
        # entropy bonus keeps some exploration alive throughout training.
        #
        # 0.01 turned out too high once FLIP_PENALTY/MISS_PENALTY were added:
        # over a 4M-step continuation, action std climbed monotonically from
        # ~1.4 to ~5.5 (log this to confirm if tuning again) and never
        # plateaued — the entropy bonus's guaranteed reward outweighed the
        # policy-gradient signal, especially while the value function was
        # still adjusting to the new reward scale. With actions clipped to
        # [-1, 1], a std that large means most sampled actions are noise
        # slammed against the clip boundary, not a refined policy. 0.001
        # keeps some exploration pressure without letting it dominate.
        ent_coef=0.001,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    # save_freq counts calls to _on_step(), which fires once per rollout
    # step across all N_ENVS in parallel (num_timesteps advances by N_ENVS
    # each call) — divide by N_ENVS so checkpoints land every ~50k real
    # timesteps regardless of env count.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="drone_soccer_ppo",
    )

    # PPO isn't monotonic — a policy can peak then regress (entropy
    # collapse, a bad batch of updates knocking it off a good optimum), so
    # the last checkpoint isn't necessarily the best one. EvalCallback runs
    # deterministic-free rollouts on a separate single-env instance every
    # eval_freq*N_ENVS timesteps and keeps the highest-scoring policy as
    # best_model.zip, independent of whatever train.learn() happens to end
    # on. deterministic=False to match watch.py's finding for this policy:
    # its mean action flips almost every episode, while sampling its actual
    # learned distribution scores far more often — evaluating the mean
    # action here would just measure the wrong thing.
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best_model/",
        log_path="./eval_logs/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=1_000_000,
        # LivePreviewCallback dropped for max training speed: it pauses to
        # open a real-time GUI window every preview_freq steps, which costs
        # both the render time itself and the overhead of spinning up a
        # separate PyBullet client each time.
        callback=CallbackList([
            ProgressPrintCallback(print_freq=100_000),
            TerminationReasonCallback(report_freq=50_000),
            checkpoint_callback,
            eval_callback,
        ]),
    )
    model.save("drone_soccer_ppo_v3")

    print("Training complete. Model saved to drone_soccer_ppo_v3.zip")
