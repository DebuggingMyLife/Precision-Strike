"""
Starter Gymnasium environment for the drone soccer striker task.

This is a skeleton, not a finished simulation. Fill in:
  - self._load_scene(): load your drone URDF + opponent + goal/field
  - self._apply_action(): map policy output to drone motor commands
  - self._compute_reward(): your reward function
  - self._check_done(): episode termination logic

The observation pipeline is already wired up:
  drone state + camera RGB/depth -> object detector -> distance/bearing to opponent
  -> flat feature vector fed to the (Recurrent) PPO policy.

During training use p.DIRECT (headless, fast). Switch to p.GUI only when you
want to watch an episode locally.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class DroneSoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, detector=None, render_mode=None, img_size=84):
        super().__init__()
        self.render_mode = render_mode
        self.img_size = img_size
        self.detector = detector  # plug in your trained YOLO/other detector here

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- Observation space ---
        # [drone pos(3), drone vel(3), drone orientation(4 quat)] = 10
        # + [opponent_detected(1), rel_x(1), rel_y(1), distance(1)] = 4
        # Adjust as your feature encoding grows (e.g. add velocity of opponent).
        obs_dim = 10 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space ---
        # 4 continuous motor thrust commands, normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.drone_id = None
        self.opponent_id = None
        self.step_count = 0
        self.max_steps = 1000

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self._load_scene()
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation(physicsClientId=self._client)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(physicsClientId=self._client)

    # ------------------------------------------------------------------ #
    # Scene / action / reward — fill these in for your specific task
    # ------------------------------------------------------------------ #
    def _load_scene(self):
        """Load drone, opponent drone, goal, field boundaries."""
        # Placeholder: using a simple sphere as a stand-in body.
        # Replace with your actual drone URDF (e.g. from gym-pybullet-drones).
        start_pos = [0, 0, 1]
        self.drone_id = p.loadURDF("sphere2.urdf", start_pos, globalScaling=0.2)

        opponent_pos = [2, 0, 1]
        self.opponent_id = p.loadURDF(
            "sphere2.urdf", opponent_pos, globalScaling=0.2
        )

    def _apply_action(self, action):
        """Map normalized action [-1, 1]^4 to drone motor forces/torques."""
        # Placeholder: apply a simple external force in x/y/z based on action.
        # Replace with real quadrotor dynamics (per-motor thrust -> force/torque).
        force = [float(action[0]) * 5, float(action[1]) * 5, float(action[2]) * 5]
        p.applyExternalForce(
            self.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME
        )

    def _compute_reward(self):
        """Placeholder reward: encourage closing distance to opponent."""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        opp_pos, _ = p.getBasePositionAndOrientation(self.opponent_id)
        dist = np.linalg.norm(np.array(drone_pos) - np.array(opp_pos))
        return -dist  # replace with your actual scoring/task reward

    def _check_done(self):
        """Placeholder termination: none yet."""
        return False

    # ------------------------------------------------------------------ #
    # Observation: drone state + camera -> detector -> feature vector
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        drone_state = self._get_drone_state()
        det_features = self._get_detection_features()
        return np.concatenate([drone_state, det_features]).astype(np.float32)

    def _get_drone_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, _ = p.getBaseVelocity(self.drone_id)
        return np.array(list(pos) + list(vel) + list(orn), dtype=np.float32)

    def _get_detection_features(self):
        """
        Renders the drone's onboard camera, runs the detector (if provided),
        and returns [detected_flag, rel_x, rel_y, distance].

        With no detector attached, this uses PyBullet's ground-truth depth
        buffer directly (useful for early sim-only training/debugging before
        your real detector is plugged in).
        """
        rgb, depth = self._render_camera()

        if self.detector is not None:
            # Expect detector(rgb) -> list of (x1, y1, x2, y2, conf, cls)
            detections = self.detector(rgb)
            if len(detections) == 0:
                return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            x1, y1, x2, y2, conf, cls = detections[0]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            distance = self._depth_to_distance(depth[cy, cx])
            rel_x = (cx / self.img_size) * 2 - 1  # normalize to [-1, 1]
            rel_y = (cy / self.img_size) * 2 - 1
            return np.array([1.0, rel_x, rel_y, distance], dtype=np.float32)

        # Fallback: ground-truth relative position (sim-only sanity check)
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        opp_pos, _ = p.getBasePositionAndOrientation(self.opponent_id)
        rel = np.array(opp_pos) - np.array(drone_pos)
        distance = np.linalg.norm(rel)
        return np.array([1.0, rel[0], rel[1], distance], dtype=np.float32)

    def _render_camera(self):
        """Render an onboard camera image + depth buffer from the drone's pose."""
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id)
        rot_matrix = p.getMatrixFromQuaternion(drone_orn)
        forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        cam_target = [drone_pos[i] + forward_vec[i] for i in range(3)]

        view_matrix = p.computeViewMatrix(drone_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.05, farVal=20
        )
        _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
            self.img_size, self.img_size, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.reshape(rgb_raw, (self.img_size, self.img_size, 4))[:, :, :3]
        depth = np.reshape(depth_raw, (self.img_size, self.img_size))
        return rgb, depth

    @staticmethod
    def _depth_to_distance(depth_buffer_value, near=0.05, far=20):
        """Convert PyBullet's normalized depth buffer value to real distance (metres)."""
        return far * near / (far - (far - near) * depth_buffer_value)


if __name__ == "__main__":
    # Quick sanity check: random actions, no detector, ground-truth obs.
    env = DroneSoccerEnv(render_mode=None)
    obs, info = env.reset()
    print("Initial obs:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f} obs={obs}")
    env.close()
