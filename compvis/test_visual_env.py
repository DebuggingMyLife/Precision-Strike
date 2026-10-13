"""Real headless simulation checks for the visual observation contract."""

import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p
from gymnasium.utils.env_checker import check_env

from drone_soccer_env import DroneSoccerEnv
from compvis.visual_env import VisualDroneSoccerEnv


class VisualEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = VisualDroneSoccerEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_contract(self):
        env = self.make_env(width=80, height=60)
        check_env(env, skip_render_check=True)

    def test_observation_has_no_privileged_detection_fields_and_no_aliasing(self):
        env = self.make_env()
        with patch.object(env, '_get_detection_features', side_effect=AssertionError('privileged detections')), \
                patch.object(env, '_get_drone_state', side_effect=AssertionError('legacy state')):
            obs, _ = env.reset(seed=0)
            self.assertEqual(set(obs), {'rgb', 'telemetry'})
            self.assertEqual(obs['rgb'].shape, (240, 320, 3))
            self.assertEqual(obs['rgb'].dtype, np.uint8)
            self.assertEqual(obs['telemetry'].shape, (8,))
            self.assertTrue(env.observation_space.contains(obs))
            # The initial scene contains visible red pixels from the hoop.
            pixels = obs['rgb'].astype(np.int16)
            self.assertTrue(np.any((pixels[:, :, 0] > pixels[:, :, 1] + 50) &
                                   (pixels[:, :, 0] > pixels[:, :, 2] + 50)))
            saved = env.render()
            obs['rgb'][:] = 0
            np.testing.assert_array_equal(env.render(), saved)
            env.render()[:] = 0
            np.testing.assert_array_equal(env.render(), saved)
            next_obs, *_ = env.step(np.zeros(4, dtype=np.float32))
            self.assertTrue(env.observation_space.contains(next_obs))

    def test_telemetry_body_axes_normalization_and_clipping(self):
        env = self.make_env(width=80, height=60)
        env.reset(seed=1)
        orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        p.resetBasePositionAndOrientation(env.drone_id, [0.5, 0.5, 1.5], orientation,
                                          physicsClientId=env._client)
        p.resetBaseVelocity(env.drone_id, linearVelocity=[0, 2, 0],
                            physicsClientId=env._client)
        telemetry = env._get_telemetry()
        np.testing.assert_allclose(telemetry, [0.5, 0.4, 0, 0, 0, 1, 0, 1], atol=1e-6)
        # Moving world X/Y must not expose global location in the telemetry.
        p.resetBasePositionAndOrientation(env.drone_id, [2, -1, 1.5], orientation,
                                          physicsClientId=env._client)
        p.resetBaseVelocity(env.drone_id, linearVelocity=[0, 2, 0],
                            physicsClientId=env._client)
        np.testing.assert_allclose(env._get_telemetry(), telemetry, atol=1e-6)
        p.resetBasePositionAndOrientation(env.drone_id, [0, 0, 4], [0, 0, 0, 1],
                                          physicsClientId=env._client)
        p.resetBaseVelocity(env.drone_id, linearVelocity=[20, -20, 0],
                            physicsClientId=env._client)
        np.testing.assert_allclose(env._get_telemetry()[:4], [1, 1, -1, 0])

    def test_camera_mount_rotates_with_body(self):
        env = self.make_env(width=80, height=60, camera_offset=(0.1, 0, 0))
        env.reset(seed=0)
        orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        p.resetBasePositionAndOrientation(env.drone_id, [1, 0, 1], orientation,
                                          physicsClientId=env._client)
        with patch('compvis.visual_env.p.computeViewMatrix', wraps=p.computeViewMatrix) as view:
            rgb, _ = env._render_camera()
        eye, target, up = view.call_args.args
        np.testing.assert_allclose(eye, [1.1, 0, 1], atol=1e-6)
        np.testing.assert_allclose(target, [2.1, 0, 1], atol=1e-6)
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-6)
        self.assertEqual(rgb.shape, (60, 80, 3))

    def test_visual_variant_preserves_baseline_dynamics_rewards_and_termination(self):
        baseline = DroneSoccerEnv()
        self.addCleanup(baseline.close)
        visual = self.make_env(width=80, height=60)
        original, _ = baseline.reset(seed=7)
        visual.reset(seed=7)
        self.assertEqual(original.shape, (26,))
        for action in [np.zeros(4, dtype=np.float32), np.full(4, 0.1, dtype=np.float32)] * 5:
            old = baseline.step(action)
            new = visual.step(action)
            self.assertAlmostEqual(old[1], new[1], places=7)
            self.assertEqual(old[2:], new[2:])
            np.testing.assert_allclose(
                baseline._get_drone_state(), visual._get_drone_state(), atol=1e-7)
        baseline.max_steps = baseline.step_count + 1
        visual.max_steps = visual.step_count + 1
        self.assertEqual(baseline.step(np.zeros(4))[2:], visual.step(np.zeros(4))[2:])
        baseline.reset(seed=7)
        visual.reset(seed=7)
        for env in (baseline, visual):
            p.resetBasePositionAndOrientation(env.drone_id, [3.1, 0, 1], [0, 0, 0, 1],
                                              physicsClientId=env._client)
        self.assertEqual(baseline.step(np.zeros(4))[1:], visual.step(np.zeros(4))[1:])


if __name__ == '__main__':
    unittest.main()
