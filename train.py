"""
Fresh-start training script for the drone soccer striker task using
RecurrentPPO. Trains a brand-new model from scratch — see train_continue.py
to continue training an existing one instead. Swap in a trained detector by
passing it to DroneSoccerEnv(detector=your_model).

Run:
    python train.py
"""

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

# Naming convention (see PROGRESS.md): vX.0 = fresh start, vX.1/.2/... =
# later stages on that lineage. X changes only for a fresh start under
# different core physics (ATTITUDE_GAIN so far); the stage number bumps for
# continuations that don't change physics. This must be changed before
# running again, or it will silently overwrite an existing model.
#
# v2.0: second lineage, ATTITUDE_GAIN=0.6 (v1.x was 0.4) — fine-tuning
# v1.4 (the v1.x lineage's best) under the new gain (v1.5_abandoned, +4M
# steps) fell well short of v1.4's 27.3% and showed no sign of closing the
# gap by the end (best checkpoint 11.3%, final save even lower at 9.3%),
# suggesting the fine-tuned policy was stuck adapting rather than slowly
# converging. Testing from scratch instead: the reward shaping
# (FLIP_PENALTY, MISS_PENALTY, LATERAL_SCALE, ent_coef=0.001) is already
# mature at this point, unlike when v1.0 started, so this shouldn't need to
# re-discover any of that — just learn flight/navigation under the new
# torque response without v1.4's 0.4-tuned habits to unlearn first.
#
# v2.0 replaced (not bumped to v3.0): its first attempt never got
# meaningful training (stopped twice, max 500k/10M steps) before the env
# changed further underneath it — DRONE_MASS to the real Tello EDU spec
# (100g incl. cage, was 105g), real hoop dimensions (45cm outer / 3.5cm
# border, ring vs opening radius properly separated), field resized
# 3x3m -> 2.09x2.09x2.09m (also adds a ceiling that didn't exist before),
# and out_of_bounds changed from terminal to a per-step penalty (the
# boundary is a net, not a wall). Reusing the v2.0 name and starting over
# rather than treating this as a new lineage, per instruction — the old
# 500k-step v2.0 checkpoints are gone (see PROGRESS.md for the historical
# note); this run's own checkpoints will land on the same step-count
# filenames from scratch.
OUTPUT_MODEL = "drone_soccer_ppo_v2.0"

# TF32 matmuls: free precision-for-speed tradeoff on Ampere+/Blackwell GPUs,
# and RL gradients are noisy enough that the reduced precision doesn't matter.
torch.set_float32_matmul_precision("high")

# The policy itself is tiny (MLP+LSTM on a 26-number observation), so it
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

    # SubprocVecEnv runs each env in its own process. Note: benchmarked
    # against DummyVecEnv (all envs sequential in one process) and found no
    # real difference (1.05x) — the actual bottleneck is the PPO update
    # phase (LSTM gradient steps), not environment stepping, for this small
    # a policy. Swap to DummyVecEnv if you want simpler tracebacks while
    # debugging; it won't cost meaningful throughput.
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
        # Benchmarked: device="cpu" beat "cuda" (the "auto" default) by 1.29x
        # for this policy (64x64 MLP + 128 LSTM) — it's small enough that GPU
        # kernel-launch/PCIe-transfer overhead per minibatch outweighs the
        # compute it saves. Confirmed via bench_device.py: 130.6 fps (cpu) vs
        # 101.4 fps (cuda), same N_ENVS/SubprocVecEnv/source model.
        device="cpu",
    )

    # save_freq counts calls to _on_step(), which fires once per rollout
    # step across all N_ENVS in parallel (num_timesteps advances by N_ENVS
    # each call) — divide by N_ENVS so checkpoints land every ~50k real
    # timesteps regardless of env count.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix=OUTPUT_MODEL,
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
        best_model_save_path=f"./checkpoints/best_model_{OUTPUT_MODEL}/",
        log_path=f"./eval_logs_{OUTPUT_MODEL}/",
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=False,
    )

    model.learn(
        total_timesteps=4_000_000,
        callback=CallbackList([
            ProgressPrintCallback(print_freq=100_000),
            TerminationReasonCallback(report_freq=50_000),
            checkpoint_callback,
            eval_callback,
        ]),
        tb_log_name="v2.0",
    )
    # SB3's save() only appends ".zip" if the path has no extension at all —
    # for a dotted name like "v2.0" it sees ".0" and assumes one's already
    # there, silently saving with no extension at all (hit this for real:
    # drone_soccer_ppo_v2.0's final save came out as a zip archive literally
    # named "drone_soccer_ppo_v2.0", not "...v2.0.zip", so watch.py's *.zip
    # glob couldn't find it). Passing ".zip" explicitly sidesteps the
    # ambiguity regardless of how many dots OUTPUT_MODEL has.
    model.save(f"{OUTPUT_MODEL}.zip")

    print(f"Training complete. Model saved to {OUTPUT_MODEL}.zip")
