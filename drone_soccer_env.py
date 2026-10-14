import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

# Paths
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DRONE_URDF = os.path.join(ASSETS_DIR, "cf2x.urdf")

# Drone Physics
DRONE_MASS = 0.100
DRONE_KF = 3.16e-10
DRONE_KM = 7.94e-12
DRONE_THRUST2WEIGHT = 1.7
GRAVITY_ACCEL = 9.81

# Timing
PHYSICS_HZ = 240
DETECTOR_HZ = 30
DETECTOR_STEP_INTERVAL = PHYSICS_HZ // DETECTOR_HZ

# Inertia
_CF2X_BASE_MASS = 0.027
_CF2X_BASE_INERTIA = (1.4e-5, 1.4e-5, 2.17e-5)
_INERTIA_SCALE = DRONE_MASS / _CF2X_BASE_MASS
DRONE_INERTIA = tuple(i * _INERTIA_SCALE for i in _CF2X_BASE_INERTIA)

_GRAVITY_FORCE = GRAVITY_ACCEL * DRONE_MASS
HOVER_RPM = (_GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5
MAX_RPM = (DRONE_THRUST2WEIGHT * _GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5

# Rewards
SCORE_REWARD = 50.0
CRASH_PENALTY = -50.0
COLLISION_PENALTY = -20.0
FLIP_PENALTY = -30.0
MISS_PENALTY = -10.0
OUT_OF_BOUNDS_PENALTY = -20.0
PROGRESS_SCALE = 10.0
LATERAL_SCALE = 10.0
FINAL_APPROACH_X_FRAC = 2 / 3

# Attitude
ATTITUDE_GAIN = 0.6
MAX_TILT_RAD = 1.5
FLIP_GRACE_STEPS = 2 * PHYSICS_HZ

# Field Config
FIELD_X_MIN = 0.0
FIELD_X_MAX = 2.09
FIELD_Y_HALF = 2.09 / 2
FIELD_Z_MAX = 2.09
GOAL_Z = 2.09 / 2
FINAL_APPROACH_X = FIELD_X_MAX * FINAL_APPROACH_X_FRAC

# Hoop Configs
GOAL_OUTER_DIAMETER = 0.45
GOAL_BORDER_THICKNESS = 0.035
GOAL_TUBE_RADIUS = GOAL_BORDER_THICKNESS / 2
GOAL_RING_RADIUS = GOAL_OUTER_DIAMETER / 2 - GOAL_TUBE_RADIUS
GOAL_OPENING_RADIUS = GOAL_RING_RADIUS - GOAL_TUBE_RADIUS

# Defender Config
OPPONENT_SPEED_MPS = 1.5
CHASE_PROBABILITY = 0.5
DEFENDER_SPAWN_POS = (FIELD_X_MAX - 0.3, 0.0, GOAL_Z)
_DEFENDER_X_RANGE = (FIELD_X_MIN + 0.3, FIELD_X_MAX - 0.3)
_DEFENDER_Y_RANGE = (-(FIELD_Y_HALF - 0.2), FIELD_Y_HALF - 0.2)
_DEFENDER_Z_RANGE = (GOAL_Z - 0.3, GOAL_Z + 0.3)


class DroneSoccerEnv(gym.Env):
    TIME_PENALTY = -0.02
    metadata = {"render_modes": ["human", None]}

    def __init__(self, detector, render_mode=None, img_size=160,
                 chase_probability=CHASE_PROBABILITY,
                 opponent_speed_mps=OPPONENT_SPEED_MPS):
        super().__init__()
        if detector is None:
            raise ValueError("detector is required")
        self.render_mode = render_mode
        self.img_size = img_size
        self.detector = detector
        self.chase_probability = chase_probability
        self.opponent_speed = opponent_speed_mps / PHYSICS_HZ

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )
        self._configure_physics()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.max_steps = 1000

    def set_chase_probability(self, p):
        self.chase_probability = p

    def _configure_physics(self):
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            contactBreakingThreshold=0.005,
            physicsClientId=self._client,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -GRAVITY_ACCEL, physicsClientId=self._client)
        self._configure_physics()
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._load_scene()
        self.step_count = 0
        self._flip_start_step = None
        self._last_detection = None

        drone_pos, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        self._prev_dist_to_goal = np.linalg.norm(
            np.array(drone_pos) - np.array(self.goal_pos)
        )
        self._prev_lateral_dist = np.linalg.norm(
            np.array(drone_pos[1:]) - np.array(self.goal_pos[1:])
        )

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._apply_action(action)
        self._update_defender()
        p.stepSimulation(physicsClientId=self._client)
        self.step_count += 1

        obs = self._get_obs()
        termination_reason = self._check_done()
        reward = self._compute_reward(termination_reason)
        terminated = termination_reason is not None
        truncated = self.step_count >= self.max_steps
        if terminated:
            info = {"termination_reason": termination_reason}
        elif truncated:
            info = {"termination_reason": "max_steps"}
        else:
            info = {}

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(physicsClientId=self._client)

    def show_termination_text(self, text, pause=1.5):
        if self.render_mode != "human":
            return
        drone_pos, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        p.addUserDebugText(
            text, [drone_pos[0], drone_pos[1], drone_pos[2] + 0.3],
            textColorRGB=[1, 0.2, 0.2], textSize=1.8,
            physicsClientId=self._client,
        )
        time.sleep(pause)

    def _load_scene(self):
        start_pos = [FIELD_X_MIN + 0.1, 0, GOAL_Z]
        self.drone_id = p.loadURDF(
            DRONE_URDF, start_pos, physicsClientId=self._client
        )
        p.changeDynamics(
            self.drone_id, linkIndex=-1,
            mass=DRONE_MASS,
            localInertiaDiagonal=DRONE_INERTIA,
            restitution=0.8,
            lateralFriction=0.1,
            linearDamping=0.6,
            angularDamping=0.8,
            physicsClientId=self._client,
        )

        self.goal_pos = [FIELD_X_MAX, 0, GOAL_Z]
        self.goal_radius = GOAL_OPENING_RADIUS
        self._create_hoop(
            self.goal_pos, radius=GOAL_RING_RADIUS, tube_radius=GOAL_TUBE_RADIUS
        )

        self._defender_chasing = self.np_random.random() < self.chase_probability
        self._defender_pos = np.array(DEFENDER_SPAWN_POS, dtype=np.float64)
        self._defender_target = self._defender_waypoint()
        self.opponent_id = p.loadURDF(
            DRONE_URDF, DEFENDER_SPAWN_POS, physicsClientId=self._client
        )
        p.changeDynamics(
            self.opponent_id, linkIndex=-1,
            mass=DRONE_MASS, localInertiaDiagonal=DRONE_INERTIA,
            restitution=0.8, lateralFriction=0.1,
            physicsClientId=self._client,
        )
        self._update_defender()

    def _defender_waypoint(self):
        return np.array([
            self.np_random.uniform(*_DEFENDER_X_RANGE),
            self.np_random.uniform(*_DEFENDER_Y_RANGE),
            self.np_random.uniform(*_DEFENDER_Z_RANGE),
        ], dtype=np.float64)

    def _create_hoop(self, center, radius, tube_radius, segments=16,
                     rgba=(1, 0, 0, 1)):
        seg_length = 2 * np.pi * radius / segments
        collision_shape = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=tube_radius, height=seg_length,
            physicsClientId=self._client,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CAPSULE, radius=tube_radius, length=seg_length,
            rgbaColor=list(rgba), physicsClientId=self._client,
        )

        for i in range(segments):
            angle = 2 * np.pi * i / segments
            position = [
                center[0],
                center[1] + radius * np.cos(angle),
                center[2] + radius * np.sin(angle),
            ]
            orientation = p.getQuaternionFromEuler([angle, 0, 0])
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self._client,
            )

    def _update_defender(self):
        if self._defender_chasing:
            if self._collided_with_opponent():
                target = self._defender_pos
            else:
                drone_pos, _ = p.getBasePositionAndOrientation(
                    self.drone_id, physicsClientId=self._client
                )
                target = np.array(drone_pos, dtype=np.float64)
        else:
            target = self._defender_target

        to_target = target - self._defender_pos
        dist = np.linalg.norm(to_target)
        if dist <= self.opponent_speed:
            self._defender_pos = target
            if not self._defender_chasing:
                self._defender_target = self._defender_waypoint()
        else:
            self._defender_pos = self._defender_pos + to_target / dist * self.opponent_speed
        self._place_opponent(self.opponent_id, self._defender_pos)

    def _place_opponent(self, body_id, position):
        p.resetBasePositionAndOrientation(
            body_id, position, [0, 0, 0, 1], physicsClientId=self._client
        )
        p.resetBaseVelocity(
            body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0],
            physicsClientId=self._client,
        )

    def _apply_action(self, action):
        act_roll, act_pitch, act_throttle, act_yaw = np.clip(action, -1.0, 1.0)

        rpm_range = MAX_RPM - HOVER_RPM
        base_rpm = HOVER_RPM + act_throttle * rpm_range
        attitude_range = rpm_range * ATTITUDE_GAIN

        rpm_0 = base_rpm + attitude_range * (-act_roll + act_pitch + act_yaw)
        rpm_1 = base_rpm + attitude_range * (-act_roll - act_pitch - act_yaw)
        rpm_2 = base_rpm + attitude_range * (act_roll - act_pitch + act_yaw)
        rpm_3 = base_rpm + attitude_range * (act_roll + act_pitch - act_yaw)
        rpms = np.clip(np.array([rpm_0, rpm_1, rpm_2, rpm_3]), 0, MAX_RPM)

        forces = (rpms ** 2) * DRONE_KF
        torques = (rpms ** 2) * DRONE_KM
        total_torque_z = torques[0] - torques[1] + torques[2] - torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.drone_id, linkIndex=i,
                forceObj=[0, 0, forces[i]], posObj=[0, 0, 0],
                flags=p.LINK_FRAME, physicsClientId=self._client,
            )
        p.applyExternalTorque(
            self.drone_id, linkIndex=-1,
            torqueObj=[0, 0, total_torque_z],
            flags=p.LINK_FRAME, physicsClientId=self._client,
        )

    def _compute_reward(self, termination_reason):
        drone_pos, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        dist_to_goal = np.linalg.norm(np.array(drone_pos) - np.array(self.goal_pos))
        lateral_dist = np.linalg.norm(
            np.array(drone_pos[1:]) - np.array(self.goal_pos[1:])
        )

        if termination_reason == "crashed":
            reward = CRASH_PENALTY
        elif termination_reason == "flipped":
            reward = FLIP_PENALTY
        elif self._collided_with_opponent():
            reward = COLLISION_PENALTY
        elif self._out_of_bounds(drone_pos):
            reward = OUT_OF_BOUNDS_PENALTY
        elif termination_reason == "scored":
            reward = SCORE_REWARD
        elif termination_reason == "missed_goal":
            reward = MISS_PENALTY
        else:
            progress = self._prev_dist_to_goal - dist_to_goal
            reward = PROGRESS_SCALE * progress + self.TIME_PENALTY
            if drone_pos[0] >= FINAL_APPROACH_X:
                lateral_progress = self._prev_lateral_dist - lateral_dist
                reward += LATERAL_SCALE * lateral_progress

        self._prev_dist_to_goal = dist_to_goal
        self._prev_lateral_dist = lateral_dist
        return float(reward)

    def _scored(self, drone_pos):
        through_plane = drone_pos[0] >= self.goal_pos[0]
        lateral_dist = np.linalg.norm(
            np.array(drone_pos[1:]) - np.array(self.goal_pos[1:])
        )
        return through_plane and lateral_dist <= self.goal_radius

    def _collided_with_opponent(self):
        return len(p.getContactPoints(
            bodyA=self.drone_id, bodyB=self.opponent_id, physicsClientId=self._client
        )) > 0

    def _touching_ground(self):
        return len(p.getContactPoints(
            bodyA=self.drone_id, bodyB=self.plane_id, physicsClientId=self._client
        )) > 0

    def _out_of_bounds(self, drone_pos):
        return (
            drone_pos[0] < FIELD_X_MIN
            or abs(drone_pos[1]) > FIELD_Y_HALF
            or drone_pos[2] > FIELD_Z_MAX
        )

    def _check_done(self):
        drone_pos, drone_orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )

        if self._touching_ground():
            return "crashed"

        drone_euler = p.getEulerFromQuaternion(drone_orn)
        tilted = (
            abs(drone_euler[0]) > MAX_TILT_RAD or abs(drone_euler[1]) > MAX_TILT_RAD
        )
        if tilted:
            if self._flip_start_step is None:
                self._flip_start_step = self.step_count
            elif self.step_count - self._flip_start_step >= FLIP_GRACE_STEPS:
                return "flipped"
        else:
            self._flip_start_step = None

        if drone_pos[0] >= self.goal_pos[0]:
            return "scored" if self._scored(drone_pos) else "missed_goal"

        return None

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
        if self._last_detection is not None and self.step_count % DETECTOR_STEP_INTERVAL != 0:
            return self._last_detection

        rgb, depth = self._render_camera()
        detections = sorted(self.detector(rgb), key=lambda d: d[4], reverse=True)
        if not detections:
            self._last_detection = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        else:
            x1, y1, x2, y2 = detections[0][:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            distance = self._depth_to_distance(depth[cy, cx])
            rel_x = (cx / self.img_size) * 2 - 1
            rel_y = (cy / self.img_size) * 2 - 1
            self._last_detection = np.array([1.0, rel_x, rel_y, distance], dtype=np.float32)
        return self._last_detection

    def _render_camera(self):
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
        rgb = np.reshape(rgb_raw, (self.img_size, self.img_size, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth_raw, (self.img_size, self.img_size))
        return rgb, depth

    @staticmethod
    def _depth_to_distance(depth_buffer_value, near=0.05, far=20):
        return far * near / (far - (far - near) * depth_buffer_value)


if __name__ == "__main__":
    env = DroneSoccerEnv(detector=lambda rgb: [], render_mode="human")
    obs, _ = env.reset()
    print("Initial obs:", obs)
    for _ in range(150):
        obs, reward, *_ = env.step(env.action_space.sample())
        print(f"reward={reward:.3f} obs={obs}")
        time.sleep(1 / 60)
    env.close()
