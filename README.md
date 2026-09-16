# Drone Soccer RL Starter

A minimal skeleton to get you moving: a PyBullet-backed Gymnasium environment,
wired up for drone state + camera/depth-based opponent detection, and a
RecurrentPPO training script.

## Setup

```bash
pip install -r requirements.txt
```

## Files

- **drone_soccer_env.py** — the environment. Run it directly for a quick
  sanity check (`python drone_soccer_env.py`) — it steps a few random actions
  using ground-truth opponent position (no detector needed yet) so you can
  confirm the sim, action application, and reward loop all connect.
- **train.py** — trains RecurrentPPO on the env. Uses `DummyVecEnv` by default
  (easier to debug); switch to `SubprocVecEnv` once things are stable for
  real parallelism.

## What's a placeholder vs. what's real

**Already wired up (should work as-is):**
- Gymnasium `reset()`/`step()` loop
- Camera rendering + PyBullet depth buffer → real-world distance conversion
- Observation vector assembly (drone state + detection features)
- RecurrentPPO training loop

**You need to fill in:**
- `_load_scene()` — swap the placeholder sphere for a real drone URDF (check
  `gym-pybullet-drones` on GitHub for a maintained one) plus field/goal geometry
- `_apply_action()` — replace the simple force application with real per-motor
  thrust → force/torque mapping for a quadrotor
- `_compute_reward()` — replace the placeholder distance-based reward with
  your actual scoring task reward
- `_check_done()` — add real termination conditions (goal scored, drone
  crashed, out of bounds, etc.)
- Plug in your trained detector: `DroneSoccerEnv(detector=your_yolo_model)` —
  the detector should be callable as `detector(rgb_image)` returning a list of
  `(x1, y1, x2, y2, confidence, class)` boxes

## Suggested order

1. Get `python drone_soccer_env.py` running with the placeholder scene (verifies
   your PyBullet install and the env plumbing).
2. Fill in `_load_scene()` with a real drone URDF, still using placeholder
   reward/action logic — confirm it loads and simulates correctly.
3. Fill in `_apply_action()` with real motor dynamics.
4. Fill in `_compute_reward()` and `_check_done()` for your actual task.
5. Run `train.py` with vanilla PPO first (swap `RecurrentPPO` →
   `PPO`/`MlpPolicy` temporarily) to validate the reward signal before
   adding recurrence.
6. Switch back to RecurrentPPO once the vanilla baseline is learning
   something sensible.
7. Plug in your object detector, replacing the ground-truth fallback.
