"""
Matched evaluation: loads a model, runs N headless episodes with
deterministic=False (see PROGRESS.md — this policy's mean action
underperforms its sampled distribution, same reason EvalCallback and
watch.py both use deterministic=False), and tallies termination_reason.

Run:
    python eval_model.py <model_path> [n_episodes]

Or import evaluate() directly for use in other scripts.
"""

import sys
from collections import Counter

from sb3_contrib import RecurrentPPO

from drone_soccer_env import DroneSoccerEnv


def evaluate(model_path, n_episodes=150, seed=0):
    model = RecurrentPPO.load(model_path, device="cpu")
    env = DroneSoccerEnv()

    reasons = Counter()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        lstm_states = None
        episode_start = True
        for _ in range(1000):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start,
                deterministic=False,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_start = terminated or truncated
            if episode_start:
                reasons[info.get("termination_reason")] += 1
                break
    env.close()

    total = sum(reasons.values())
    scored = reasons.get("scored", 0)
    missed = reasons.get("missed_goal", 0)
    shots = scored + missed
    return {
        "model_path": model_path,
        "n_episodes": total,
        "counts": dict(reasons),
        "scored_pct": 100 * scored / total if total else 0.0,
        "shot_accuracy_pct": 100 * scored / shots if shots else 0.0,
    }


def format_result(result):
    counts_str = ", ".join(
        f"{k}={v} ({100 * v / result['n_episodes']:.1f}%)"
        for k, v in sorted(result["counts"].items())
    )
    return (
        f"{result['model_path']} ({result['n_episodes']} episodes)\n"
        f"  {counts_str}\n"
        f"  scored={result['scored_pct']:.1f}%  "
        f"shot_accuracy={result['shot_accuracy_pct']:.1f}%"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_model.py <model_path> [n_episodes]")
        sys.exit(1)
    path = sys.argv[1]
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    print(f"Evaluating {path} over {n_eps} episodes...")
    result = evaluate(path, n_episodes=n_eps)
    print(format_result(result))
