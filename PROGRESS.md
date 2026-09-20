# Progress Notes

Running log of milestones, decisions, and fixes for the drone soccer striker
task. Chronological within each section; newest at the bottom.

## Model lineage

| Model | Steps | Env version | Notes |
|---|---|---|---|
| `v1` | 1M (fresh) | old, single opponent | First baseline. |
| `v2` | +2M (from v1), 3M total | old, single opponent | Reached ~69% scored. **Incompatible with current env** — obs space was 14-dim (1 opponent) vs current 26-dim (4 opponents), so it can no longer be loaded against `drone_soccer_env.py`. |
| — | — | env rewrite | Added goalkeeper + 3 random-flying opponents (`N_OPPONENTS=4`), reworked drone mass/thrust-to-weight to DJI Tello specs, added potential-based progress shaping, loosened `MAX_TILT_RAD`. |
| `v3` | 1M (fresh) | current (4 opponents) | Fresh run under the rewritten env. Mostly failing at this point (flip/miss dominated, near-zero scored) — expected, undertrained. |
| `v4` | +4M (from v3), 5M total | current | See fixes below. Final breakdown (last ~1M steps, 3125 episodes): missed_goal 52%, flipped 37%, out_of_bounds 6%, scored 5%, crashed 0.4%. |
| `v5` | +4M (from v4), in progress | current + lateral shaping | See "final-approach lateral shaping" below. |

## Key findings & fixes

- **Flip/miss had no reward penalty.** `_check_done()` could end an episode as `"flipped"` or `"missed_goal"`, but `_compute_reward()` had no matching branch — those outcomes scored the same as an ordinary step, so the policy had no direct incentive to avoid them beyond losing future progress reward. Added `FLIP_PENALTY=-30`, `MISS_PENALTY=-10`.
- **`ent_coef=0.01` caused an entropy runaway once those penalties were added.** Continuing v3→v4 with the old ent_coef, action `std` climbed monotonically from ~1.4 to ~5.5 over 4M steps (never plateaued) — the entropy bonus outweighed the policy-gradient signal while the value function was still adjusting to the new reward scale. With actions clipped to `[-1, 1]`, that std means most sampled actions are just noise slammed against the clip boundary. Fixed by lowering to `ent_coef=0.001`; re-ran v3→v4 clean (std settled to ~0.7).
- **`out_of_bounds` has the same missing-penalty gap** as flip/miss did — not yet fixed, and crept up to 6% in v4's final stretch. Candidate follow-up.
- **Flip mechanics diagnostic** (60 headless episodes against v4): even non-flip episodes reach ~80° tilt on average (task requires very aggressive flight near the tilt limit as normal operation, not just in failure cases). In the last 15 steps before an actual flip, angular velocity *accelerates* (7.5 → 14 rad/s) while action magnitude stays high/near-saturated the whole time — the policy is actively fighting the spin, not ignoring it; available roll/pitch torque authority (`ATTITUDE_GAIN=0.4`) just isn't enough to arrest it once started. Suggests raising `ATTITUDE_GAIN` or `angularDamping` as a control-side fix, separate from reward tuning. **Not yet implemented** — deprioritized after the large-sample data below showed missing, not flipping, is the bigger failure mode.
- **Large-sample check (3125 episodes) showed `missed_goal` (52%) is the dominant failure mode, not `flipped` (37%)** — the 60-episode diagnostic above was noisy on this point. Root cause: `PROGRESS_SCALE` shapes toward raw 3D distance to `goal_pos`, which doesn't specifically reward staying centered on the hoop's Y-Z opening — a drone can reduce net distance while drifting off-axis and only find out at the goal plane.
- **Final-approach lateral shaping added** to address the above: `LATERAL_SCALE=10.0` potential-based reward for reducing Y-Z distance to the hoop center, gated to `drone_pos[0] >= FINAL_APPROACH_X (2.0)` so it doesn't fight goalkeeper-dodging earlier in the flight. This is what `v4`→`v5` is training with.
- **Checkpoint/tensorboard naming collisions**, twice: `train_continue.py`'s `CheckpointCallback` name_prefix and `tb_log_name` need to be unique per lineage (`v3_continued`, `v4_continued`, ...) — `reset_num_timesteps=False` makes SB3 reuse the *latest existing* run directory for a given name rather than incrementing, so reusing a name across different training lineages silently interleaves unrelated runs' curves or overwrites old checkpoint files at the same step numbers.

## Hardware / throughput notes

- Training is CPU-bound (PyBullet physics stepped across `SubprocVecEnv` workers), not GPU-bound — the policy net (tiny MLP+LSTM) barely uses the GPU (~13% util observed).
- Current machine: Ryzen 5 7500F (6 cores/12 threads). `N_ENVS` raised from 8 → 10 (oversubscribing SMT threads, leaving 2 free for main process/OS) — safe and effective in practice.
- If moving to a Ryzen 7 9700X (8 cores/16 threads): recommend `N_ENVS≈13-14` (same oversubscription ratio), not just matching physical core count. Estimated ~5-5.5h for a run that takes 8h here (more parallel envs + faster per-core Zen5 performance).

## Open items / next steps

- `v5` result (lateral shaping) — check `missed_goal` rate drops without hurting `scored`/collision rates.
- `out_of_bounds` still has no explicit penalty.
- Flip-recovery control authority (`ATTITUDE_GAIN`/`angularDamping`) not yet tuned — worth revisiting if flip rate is still high after `v5`.
- Detector is still `None` (ground-truth opponent positions) — swapping in a real detector later will introduce observation noise the current policy has never trained against; expect a regression when that happens.
