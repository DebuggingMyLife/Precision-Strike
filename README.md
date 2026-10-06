# Drone Soccer RL — Precision Strike

A PyBullet/Gymnasium environment for a drone soccer striker task, trained
with RecurrentPPO (`sb3_contrib`). The drone must fly from a spawn line
through a hoop, avoiding a patrolling goalkeeper and 3 randomly-moving
opponents. See **PROGRESS.md** for the full development history, reward
design decisions, and model lineage (`v3` baseline through `v7_best`,
currently the best model).

## Setup

```bash
pip install -r requirements.txt
```

**Windows note:** pybullet has no prebuilt Windows wheel on PyPI, so pip
compiles it from source — this requires the
[MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(C++ workload) to be installed first, and can take 30-60+ minutes.

**GPU:** not needed — see PROGRESS.md's hardware notes. This policy is
small enough that `device="cpu"` is actually faster than GPU (benchmarked
1.29x), so both `train.py` and `train_continue.py` use it by default.

## Files

- **drone_soccer_env.py** — the environment.
- **train.py** — trains a brand-new `RecurrentPPO` model from scratch.
- **train_continue.py** — continues training an existing model. This is
  what you want most of the time (see below).
- **watch.py** — loads a model and flies it live in a GUI window:
  `python watch.py <path-to-model.zip>`.
- **PROGRESS.md** — full history: what changed at each stage, why, and what
  the actual results were (including cases where a "final" checkpoint
  regressed from a mid-run peak — see below).

## Continuing training

`train_continue.py` has `SOURCE_MODEL`/`OUTPUT_MODEL`/`ADDITIONAL_TIMESTEPS`
constants at the top — check these point at the model you actually want to
build on (see PROGRESS.md's model lineage table for the current best) before
running:

```bash
python train_continue.py
```

**Important — always verify the result before trusting it.** PPO isn't
guaranteed to improve monotonically: a run's final save can be *worse* than
a checkpoint from partway through. Every continuation run already has an
`EvalCallback` that tracks the best-scoring checkpoint independently
(saved to `checkpoints/best_model_continued_<name>/best_model.zip`) — after
a run finishes, compare its final save against that best checkpoint with a
matched evaluation (see `PROGRESS.md`'s "PPO's non-monotonic regression"
note for exactly how `v6` and `v7` were caught doing this, while `v4`/`v5`
were fine). If the mid-run checkpoint wins, copy it to a clearly-named file
(e.g. `drone_soccer_ppo_v8_best.zip`) rather than using the raw final save,
and update `PROGRESS.md` + `.gitignore` accordingly.

Each new continuation run should also get its own unique
`name_prefix`/`tb_log_name`/eval paths (bump the version number in each) —
reusing a previous run's names will silently overwrite its checkpoints or
merge unrelated runs' curves in TensorBoard. `train_continue.py`'s comments
walk through why.

## Watching a model fly

```bash
python watch.py drone_soccer_ppo_v7_best.zip
```

Opens a live GUI window and prints each episode's outcome
(scored/flipped/missed/etc.) to the console.

## Viewing training curves

```bash
python -m tensorboard.main --logdir tb_logs
```

Then open http://localhost:6006/. Runs are labeled `Baseline`, `Stage_1`
through `Stage_5`, matching the stages described in PROGRESS.md.
