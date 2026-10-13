"""Image/telemetry observation variant of the existing soccer environment.

This is a simulation observation contract, not a calibrated Tello adapter.
Physics, rewards, opponents, action semantics and episode duration are inherited.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from drone_soccer_env import DroneSoccerEnv


class VisualDroneSoccerEnv(DroneSoccerEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}
    TELEMETRY_NAMES = (
        "height_over_3m", "forward_velocity_over_5mps",
        "left_velocity_over_5mps", "up_velocity_over_5mps",
        "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
    )
    HEIGHT_SCALE_M = 3.0
    VELOCITY_SCALE_MPS = 5.0

    def __init__(self, render_mode=None, width=320, height=240,
                 vertical_fov=90.0, camera_offset=(0.06, 0.0, 0.0)):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'.")
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 32:
            raise ValueError("Camera width and height must be integers of at least 32 pixels.")
        if not 0 < vertical_fov < 180:
            raise ValueError("Vertical field of view must be between 0 and 180 degrees.")
        offset = np.asarray(camera_offset, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Camera offset must contain three finite body-frame coordinates.")
        self.camera_width = width
        self.camera_height = height
        self.vertical_fov = float(vertical_fov)
        self.camera_offset = offset.copy()
        self._last_rgb = None
        self._closed = False
        super().__init__(detector=None, render_mode=render_mode)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
            "telemetry": spaces.Box(
                low=np.array([0, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
                high=np.ones(8, dtype=np.float32), dtype=np.float32,
            ),
        })

    def _get_obs(self):
        # Do not call the parent's flat observation/detection path. In particular,
        # no opponent positions, depth or goal coordinates enter this dictionary.
        rgb, _ = self._render_camera()
        self._last_rgb = rgb.copy()
        return {"rgb": rgb, "telemetry": self._get_telemetry()}

    def _get_telemetry(self):
        position, orientation = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        world_velocity, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self._client)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(world_velocity)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)
        # Simulation body axes: +X forward, +Y left, +Z up. Height is above the
        # fixed z=0 ground plane. These scales are design choices, not Tello limits.
        telemetry = np.array([
            np.clip(position[2] / self.HEIGHT_SCALE_M, 0, 1),
            *np.clip(body_velocity / self.VELOCITY_SCALE_MPS, -1, 1),
            np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch),
        ], dtype=np.float32)
        return telemetry

    def _render_camera(self):
        position, orientation = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        eye = np.asarray(position) + rotation @ self.camera_offset
        target = eye + rotation[:, 0]
        up = rotation[:, 2]
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        projection = p.computeProjectionMatrixFOV(
            fov=self.vertical_fov, aspect=self.camera_width / self.camera_height,
            nearVal=0.02, farVal=20.0,
        )
        _, _, rgba, depth, _ = p.getCameraImage(
            self.camera_width, self.camera_height, view, projection,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self._client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(
            self.camera_height, self.camera_width, 4
        )[:, :, :3].copy()
        depth = np.asarray(depth).reshape(self.camera_height, self.camera_width)
        return rgb, depth

    def render(self):
        """Return the most recent observation image without another physics step."""
        if self._last_rgb is None:
            raise RuntimeError("Call reset() before requesting a camera preview.")
        return self._last_rgb.copy()

    def close(self):
        if not self._closed:
            super().close()
            self._closed = True
            self._last_rgb = None
