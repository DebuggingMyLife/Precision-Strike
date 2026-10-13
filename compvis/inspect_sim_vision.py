"""Offline smoke check: simulator camera -> pretrained MobileNet features."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch

from drone_soccer_env import DroneSoccerEnv
from benchmark_mobilenet import load_encoder
from compvis.visual_env import VisualDroneSoccerEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual', action='store_true', help='Check the separate RGB/telemetry environment.')
    options = parser.parse_args()
    output = ROOT / 'runs' / ('visual_env' if options.visual else 'vision_review')
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(threads=4, device='cpu', warmup=20)
    encoder, transform, device = load_encoder(args)
    env = VisualDroneSoccerEnv(render_mode='rgb_array') if options.visual else DroneSoccerEnv(img_size=256)
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        if options.visual:
            rgb = observation['rgb']
            assert rgb.shape == (240, 320, 3)
            np.testing.assert_array_equal(env.render(), rgb)
        else:
            rgb, depth = env._render_camera()
            assert rgb.shape == (256, 256, 3) and np.isfinite(depth).all()
        assert rgb.dtype == np.uint8
        with torch.inference_mode():
            image = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            features = encoder(image)
        assert features.shape == (1, 576) and torch.isfinite(features).all().item()
        assert cv2.imwrite(str(output / 'spawn_camera.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        checked_steps = 0
        for _ in range(10):
            observation, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            assert env.observation_space.contains(observation) and np.isfinite(reward)
            checked_steps += 1
            if terminated or truncated:
                break
        summary = {
            'seed': 0, 'environment': type(env).__name__,
            'observation_shape': {key: list(value.shape) for key, value in observation.items()}
                if options.visual else list(observation.shape),
            'action_shape': list(env.action_space.shape),
            'camera_shape': list(rgb.shape), 'feature_shape': list(features.shape),
            'finite_features': True, 'checked_steps': checked_steps,
            'purpose': 'Offline integration smoke check; not training or a throughput benchmark.',
        }
        if options.visual:
            summary['telemetry_order'] = list(env.TELEMETRY_NAMES)
            summary['telemetry_after_steps'] = observation['telemetry'].tolist()
            summary['camera'] = {
                'vertical_fov_degrees': env.vertical_fov,
                'mount_offset_body_m': env.camera_offset.tolist(),
                'calibrated_to_tello': False,
            }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        print(f'Camera image: {output / "spawn_camera.png"}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
