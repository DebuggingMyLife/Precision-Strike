import sys
from collections import Counter

from stable_baselines3 import PPO

from drone_soccer_env import DroneSoccerEnv
from eval_model import format_result
from train_ppo import PPO_ENV_KWARGS
from yolo_detector import YoloDetector


def evaluate(model_path, n_episodes=150, seed=0):
    model = PPO.load(model_path, device="cpu")
    env = DroneSoccerEnv(detector=YoloDetector(), **PPO_ENV_KWARGS)

    reasons = Counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(1000):
            # Sampled actions outperform the mean action for this policy
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                reasons[info.get("termination_reason")] += 1
                break
    env.close()

    total = sum(reasons.values())
    scored = reasons.get("scored", 0)
    shots = scored + reasons.get("missed_goal", 0)
    return {
        "model_path": model_path,
        "n_episodes": total,
        "counts": dict(reasons),
        "scored_pct": 100 * scored / total if total else 0.0,
        "shot_accuracy_pct": 100 * scored / shots if shots else 0.0,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_model_ppo.py <model_path> [n_episodes]")
        sys.exit(1)
    path = sys.argv[1]
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    print(f"Evaluating {path} over {n_eps} episodes...")
    print(format_result(evaluate(path, n_episodes=n_eps)))
