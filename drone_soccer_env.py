"""
Starter Gymnasium environment for the drone soccer striker task.

This is a skeleton, not a finished simulation. Fill in:
  - self._load_scene(): drone body is real (see below); still need goal/field geometry
  - self._apply_action(): real per-motor thrust/torque now wired up (see below)
  - self._compute_reward(): your reward function
  - self._check_done(): episode termination logic

The observation pipeline is already wired up:
  drone state + camera RGB/depth -> object detector -> distance/bearing to opponent
  -> flat feature vector fed to the (Recurrent) PPO policy.

During training use p.DIRECT (headless, fast). Switch to p.GUI only when you
want to watch an episode locally.
"""

import os
import time

# Each SubprocVecEnv worker is a separate process that imports this module
# fresh; without this, NumPy's BLAS backend can spin up multiple threads per
# process, oversubscribing the CPU once several env processes run at once.
# Must be set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DRONE_URDF = os.path.join(ASSETS_DIR, "cf2x.urdf")

# Bitcraze Crazyflie 2.X physical constants, read from cf2x.urdf's <properties>
# tag (vendored from github.com/learnsyslab/gym-pybullet-drones, see
# assets/NOTICE.md). Mirrors that project's BaseAviary hover/thrust math.
DRONE_MASS = 0.1  # kg
DRONE_KF = 3.16e-10  # thrust coefficient, N / (rad/s)^2
DRONE_KM = 7.94e-12  # torque coefficient, N*m / (rad/s)^2
DRONE_THRUST2WEIGHT = 2.25
GRAVITY_ACCEL = 9.8  # m/s^2

_GRAVITY_FORCE = GRAVITY_ACCEL * DRONE_MASS
HOVER_RPM = (_GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5
MAX_RPM = (DRONE_THRUST2WEIGHT * _GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5

SCORE_REWARD = 50.0
CRASH_PENALTY = -50.0
COLLISION_PENALTY = -20.0


class DroneSoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, detector=None, render_mode=None, img_size=84):
        super().__init__()
        self.render_mode = render_mode
        self.img_size = img_size
        self.detector = detector  # plug in your trained YOLO/other detector here

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )

        # --- Observation space ---
        # [drone pos(3), drone vel(3), drone orientation(4 quat)] = 10
        # + [opponent_detected(1), rel_x(1), rel_y(1), distance(1)] = 4
        # Adjust as your feature encoding grows (e.g. add velocity of opponent).
        obs_dim = 10 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space ---
        # 4 continuous commands: [roll, pitch, throttle, yaw] velocities normalized to [-1, 1]
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
        p.setGravity(0, 0, -GRAVITY_ACCEL, physicsClientId=self._client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

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
        # Crazyflie 2.X quadrotor for both your drone and the opponent.
        # TODO: add field boundary geometry.
        start_pos = [0, 0, 1]
        self.drone_id = p.loadURDF(
            DRONE_URDF, start_pos, physicsClientId=self._client
        )

        opponent_pos = [2, 0, 1]
        self.opponent_id = p.loadURDF(
            DRONE_URDF, opponent_pos, physicsClientId=self._client
        )

        # Goal: a red hoop to fly through, facing down the field's X axis.
        self.goal_pos = [3, 0, 1]
        self.goal_radius = 0.5
        self.goal_ids = self._create_hoop(self.goal_pos, radius=self.goal_radius)

    def _create_hoop(self, center, radius=0.5, tube_radius=0.03, segments=16,
                      rgba=(1, 0, 0, 1)):
        """Build a static red hoop out of capsule segments arranged in a ring.

        PyBullet has no built-in torus primitive, so the hoop is approximated
        by `segments` short capsules placed edge-to-edge around a circle.
        Each is a separate mass=0 (static) body, so the hoop floats in place
        and the drone can fly through its open centre. The ring lies in the
        world Y-Z plane, centred at `center` — i.e. you fly through it along
        the X axis. Returns the list of segment body IDs.
        """
        seg_length = 2 * np.pi * radius / segments
        collision_shape = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=tube_radius, height=seg_length,
            physicsClientId=self._client,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CAPSULE, radius=tube_radius, length=seg_length,
            rgbaColor=list(rgba), physicsClientId=self._client,
        )

        segment_ids = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            position = [
                center[0],
                center[1] + radius * np.cos(angle),
                center[2] + radius * np.sin(angle),
            ]
            # Aligns each capsule's long (local Z) axis with the ring's
            # tangent direction at this angle: a pure rotation about world X.
            orientation = p.getQuaternionFromEuler([angle, 0, 0])
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self._client,
            )
            segment_ids.append(body_id)
        return segment_ids
    
    def _apply_action(self, action):
        """Map normalized action [-1, 1]^4 to high-level Tello velocities.
        
        action[0] = Roll (Left/Right)
        action[1] = Pitch (Forward/Backward)
        action[2] = Throttle (Up/Down)
        action[3] = Yaw (Rotation)
        """
        # Ensure the inputs are safe
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to realistic physical speeds in meters per second (m/s)
        # and radians per second for rotation
        max_linear_vel = 2.0   # Tello moves at roughly 2 m/s max
        max_angular_vel = 1.5  # Comfortable turning speed
        
        # Map actions to direction vectors
        # action[1] = Forward/Backward (X axis)
        # action[0] = Left/Right (Y axis)
        # action[2] = Up/Down (Z axis)
        linear_velocity = [
            action[1] * max_linear_vel, 
            action[0] * max_linear_vel, 
            action[2] * max_linear_vel
        ]
        
        # action[3] = Turn clockwise/counter-clockwise around Z axis
        angular_velocity = [0.0, 0.0, action[3] * max_angular_vel]
        
        # Directly override body physics to move the drone like a Tello autopilot
        p.resetBaseVelocity(
            self.drone_id, 
            linearVelocity=linear_velocity, 
            angularVelocity=angular_velocity,
            physicsClientId=self._client
        )
    def _compute_reward(self):
        # 1. Extract drone positions and orientation from PyBullet
        drone_pos, drone_quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        drone_euler = p.getEulerFromQuaternion(drone_quat)  # [roll, pitch, yaw]
        
        # Extract target position (e.g., the hoop goal)
        goal_pos = np.array(self.goal_pos)
        current_pos = np.array(drone_pos)
        
        # 2. PHASE 1: Stabilization & Hover rewards (Crucial for early training)
        # Target a safe altitude (e.g., 1.0 meter)
        target_z = 1.0
        z_error = abs(current_pos[2] - target_z)
        reward_hover = np.exp(-2.0 * z_error)  # Max 1.0 when perfectly at 1.0m altitude
        
        # Severe penalty for tilting (keeps it flat like a Tello autopilot would)
        # Tello's internal IMU prevents flips, so we must force the RL to stay flat
        tilt_penalty = -(drone_euler[0]**2 + drone_euler[1]**2)

        # 3. PHASE 2: Navigation / Striker rewards
        # Calculate Euclidean distance to the hoop center
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        reward_dist = np.exp(-1.0 * dist_to_goal)  # Spikes close to 1.0 as it nears goal

        # 4. CONDITIONAL SHAPING MIX
        if z_error > 0.3 or abs(drone_euler[0]) > 0.4 or abs(drone_euler[1]) > 0.4:
            # Drone is unstable or on the ground. Focus exclusively on stabilization.
            reward = (1.5 * reward_hover) + (0.5 * tilt_penalty)
        else:
            # Drone is stable and airborne! Heavily incentivize moving towards the goal.
            reward = (0.3 * reward_hover) + (2.5 * reward_dist) + (0.2 * tilt_penalty)

        # 5. Terminal sparse rewards (matching your constants)
        if current_pos[2] < 0.15:  # Ground crash
            reward += CRASH_PENALTY
        elif dist_to_goal < self.goal_radius and abs(current_pos[0] - goal_pos[0]) < 0.1:
            reward += SCORE_REWARD  # Through the hoop!

        return float(reward)

    def _scored(self, drone_pos):
        """True once the drone has flown through the goal hoop's opening."""
        through_plane = abs(drone_pos[0] - self.goal_pos[0])
        lateral_dist = np.linalg.norm(
            np.array(drone_pos[1:]) - np.array(self.goal_pos[1:])
        )
        return through_plane < 0.1 and lateral_dist < self.goal_radius

    def _crashed(self):
        """True if the drone is touching the ground plane."""
        contacts = p.getContactPoints(
            bodyA=self.drone_id, bodyB=self.plane_id, physicsClientId=self._client
        )
        return len(contacts) > 0

    def _collided_with_opponent(self):
        """True if the drone is touching the opponent drone."""
        contacts = p.getContactPoints(
            bodyA=self.drone_id, bodyB=self.opponent_id, physicsClientId=self._client
        )
        return len(contacts) > 0

    def _check_done(self):
        drone_pos, drone_quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self._client)
        drone_euler = p.getEulerFromQuaternion(drone_quat)
        
        # Terminate if the drone hits the floor
        if drone_pos[2] < 0.15:
            return True
            
        # Terminate if the drone flips past 60 degrees (unrecoverable state)
        if abs(drone_euler[0]) > 1.05 or abs(drone_euler[1]) > 1.05:
            return True
            
        # Terminate only if it actually flew through the hoop's opening —
        # not just past the goal's X plane, which would also fire for a
        # drone that flew wide of the hoop entirely.
        if self._scored(drone_pos):
            return True

        return False

    # ------------------------------------------------------------------ #
    # Observation: drone state + camera -> detector -> feature vector
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        drone_state = self._get_drone_state()
        det_features = self._get_detection_features()
        return np.concatenate([drone_state, det_features]).astype(np.float32)

    def _get_drone_state(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self._client)
        return np.array(list(pos) + list(vel) + list(orn), dtype=np.float32)

    def _get_detection_features(self):
        """
        Renders the drone's onboard camera, runs the detector (if provided),
        and returns [detected_flag, rel_x, rel_y, distance].

        With no detector attached, this uses PyBullet's ground-truth depth
        buffer directly (useful for early sim-only training/debugging before
        your real detector is plugged in). The camera is only rendered when
        a detector is actually attached — otherwise it'd be dead work, and
        camera rendering is often the most expensive part of a step.
        """
        if self.detector is not None:
            rgb, depth = self._render_camera()
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
        drone_pos, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        opp_pos, _ = p.getBasePositionAndOrientation(
            self.opponent_id, physicsClientId=self._client
        )
        rel = np.array(opp_pos) - np.array(drone_pos)
        distance = np.linalg.norm(rel)
        return np.array([1.0, rel[0], rel[1], distance], dtype=np.float32)

    def _render_camera(self):
        """Render an onboard camera image + depth buffer from the drone's pose."""
        drone_pos, drone_orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        rot_matrix = p.getMatrixFromQuaternion(drone_orn)
        forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        cam_target = [drone_pos[i] + forward_vec[i] for i in range(3)]

        view_matrix = p.computeViewMatrix(drone_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.05, farVal=20
        )
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
    env = DroneSoccerEnv(render_mode="human")
    obs, info = env.reset()
    print("Initial obs:", obs)
    for _ in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.3f} obs={obs}")
        time.sleep(1 / 60)
    env.close()
