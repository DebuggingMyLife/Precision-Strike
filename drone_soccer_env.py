"""
Starter Gymnasium environment for the drone soccer striker task.

This is a skeleton, not a finished simulation. Fill in:
  - self._load_scene(): load your drone URDF + goal ring/field
  - self._apply_action(): map policy output to drone motor commands
  - self._compute_reward(): your reward function
  - self._check_done(): episode termination logic

The observation pipeline is already wired up:
  drone state + camera RGB/depth -> object detector -> distance/bearing to goal
  -> flat feature vector fed to the (Recurrent) PPO policy.

During training use p.DIRECT (headless, fast). Switch to p.GUI only when you
want to watch an episode locally.
"""
#testing josh git push

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
        # + [goal_detected(1), rel_x(1), rel_y(1), distance(1)] = 4
        # Adjust as your feature encoding grows.
        obs_dim = 10 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space ---
        # CHANGED: SWAPPED RAW XYZ THRUST FOR DIRECTIONAL FLIGHT CONTROL IN THE DRONE'S OWN FACING DIRECTION
        # [forward/backward, left/right, up/down], normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.drone_id = None
        self.goal_pos = None
        self.goal_ids = []
        self.step_count = 0
        self.max_steps = 1000

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81)
        #CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        p.loadURDF("plane.urdf", physicsClientId=self._client)

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
        """Load drone and goal ring."""
        # CHANGED: REPLACED THE PLACEHOLDER SPHERE WITH A DJI TELLO-SIZED QUADCOPTER BODY
        start_pos = [0, 0, 1]
        self.drone_id = self._load_drone(start_pos)

        # CHANGED: REPLACED THE OPPONENT SPHERE WITH A RED RING GOAL POST
        self.goal_pos = np.array([2, 0, 1], dtype=np.float32)
        self.goal_ids = self._load_goal_ring(self.goal_pos)

    def _load_drone(self, start_pos):
        """Build a simple DJI Tello-sized quadcopter: a flat rectangular body
        (real Tello: ~98x92.5x41mm, ~87g) with four rotor discs at the corners.
        This is a primitive-shape approximation, not an actual Tello mesh/URDF
        (PyBullet ships no built-in drone model) -- swap in a real mesh later
        if you want exact visuals."""
        body_half_extents = [0.049, 0.046, 0.02]
        body_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=body_half_extents, physicsClientId=self._client
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=body_half_extents, rgbaColor=[1, 1, 1, 1],
            physicsClientId=self._client,
        )

        rotor_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.03, length=0.006, rgbaColor=[0.1, 0.1, 0.1, 1],
            physicsClientId=self._client,
        )
        rotor_offsets = [
            [0.06, 0.06, 0.02], [0.06, -0.06, 0.02],
            [-0.06, 0.06, 0.02], [-0.06, -0.06, 0.02],
        ]
        num_rotors = len(rotor_offsets)

        drone_id = p.createMultiBody(
            baseMass=0.087,  # ~Tello's real-world weight in kg
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=start_pos,
            linkMasses=[0] * num_rotors,
            linkCollisionShapeIndices=[-1] * num_rotors,
            linkVisualShapeIndices=[rotor_visual] * num_rotors,
            linkPositions=rotor_offsets,
            linkOrientations=[[0, 0, 0, 1]] * num_rotors,
            linkInertialFramePositions=[[0, 0, 0]] * num_rotors,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * num_rotors,
            linkParentIndices=[0] * num_rotors,
            linkJointTypes=[p.JOINT_FIXED] * num_rotors,
            linkJointAxis=[[0, 0, 0]] * num_rotors,
            physicsClientId=self._client,
        )
        return drone_id

    def _load_goal_ring(self, center, radius=0.5, tube_radius=0.03, num_segments=16):
        """Build a red ring (drone-soccer-style goal post) out of small static
        cylinder segments, since PyBullet has no built-in torus primitive.
        The ring lies in the Y-Z plane so a drone flies through it along X."""
        segment_length = 2 * np.pi * radius / num_segments
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=tube_radius, height=segment_length,
            physicsClientId=self._client,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=tube_radius, length=segment_length,
            rgbaColor=[1, 0, 0, 1], physicsClientId=self._client,
        )

        segment_ids = []
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            offset = np.array([0, radius * np.cos(theta), radius * np.sin(theta)])
            position = center + offset
            # Rotating the cylinder's default z-axis about X by theta aligns it
            # tangent to the circle at this point.
            orientation = p.getQuaternionFromEuler([theta, 0, 0])
            segment_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position.tolist(),
                baseOrientation=orientation,
                physicsClientId=self._client,
            )
            segment_ids.append(segment_id)
        return segment_ids

    def _apply_action(self, action):
        """Map normalized action [-1, 1]^3 to forward/backward, left/right, up/down thrust.
        Still a placeholder force model, not real per-motor quadrotor dynamics --
        replace with real thrust/torque mixing when you're ready."""
        # CHANGED: action IS NOW [forward/back, left/right, up/down] INSTEAD OF RAW WORLD-FRAME XYZ.
        # p.LINK_FRAME MEANS THIS FORCE VECTOR IS INTERPRETED IN THE DRONE'S OWN LOCAL AXES, SO
        # "FORWARD" ALWAYS MEANS THE DIRECTION THE DRONE IS CURRENTLY FACING, REGARDLESS OF ITS ORIENTATION.
        thrust_scale = 5
        forward_back = float(action[0]) * thrust_scale  # local +x = forward, -x = backward
        left_right = float(action[1]) * thrust_scale     # local +y = left, -y = right
        up_down = float(action[2]) * thrust_scale         # local +z = up, -z = down
        force = [forward_back, left_right, up_down]
        p.applyExternalForce(
            self.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME, physicsClientId=self._client
        )

    def _compute_reward(self):
        """Placeholder reward: encourage closing distance to the goal ring."""
        # CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        # CHANGED: THE GOAL RING IS STATIC, SO ITS POSITION IS JUST THE STORED goal_pos, NO PYBULLET QUERY NEEDED
        dist = np.linalg.norm(np.array(drone_pos) - self.goal_pos)
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
        # CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self._client)
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

        # Fallback: ground-truth relative position of the goal ring (sim-only sanity check)
        # CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        # CHANGED: USES THE STATIC goal_pos INSTEAD OF QUERYING A REMOVED OPPONENT BODY
        rel = self.goal_pos - np.array(drone_pos)
        distance = np.linalg.norm(rel)
        return np.array([1.0, rel[0], rel[1], distance], dtype=np.float32)

    def _render_camera(self):
        """Render an onboard camera image + depth buffer from the drone's pose."""
        # CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        rot_matrix = p.getMatrixFromQuaternion(drone_orn)
        forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        cam_target = [drone_pos[i] + forward_vec[i] for i in range(3)]

        view_matrix = p.computeViewMatrix(drone_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.05, farVal=20
        )
        # CHANGED: ADDED physicsClientId=self._client SO EACH PARALLEL ENV TALKS TO ITS OWN PYBULLET CLIENT
        _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
            self.img_size, self.img_size, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self._client,
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