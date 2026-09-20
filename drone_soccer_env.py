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

# DJI Tello drone soccer specs (overrides the cf2x.urdf body's own mass/thrust
# tags below, applied via changeDynamics in _load_scene). No real Tello URDF
# on hand, so cf2x.urdf's geometry is kept as a stand-in body shape.
DRONE_MASS = 0.105            # 80g Tello + 25g soccer cage
DRONE_KF = 3.16e-10           # thrust coefficient, N / (rad/s)^2 (kept from URDF tag for RPM mixing)
DRONE_KM = 7.94e-12           # torque coefficient, N*m / (rad/s)^2
DRONE_THRUST2WEIGHT = 1.7     # Tello has a tight ~1.7:1 TWR ratio (not the cf2x's 2.25)
GRAVITY_ACCEL = 9.81  # m/s^2

# cf2x.urdf's own base_link <inertial> tag: mass=0.027kg,
# ixx=iyy=1.4e-5, izz=2.17e-5 (kg*m^2). changeDynamics only overrides mass
# below, not inertia, so without rescaling, the sim would keep a 27g-sized
# inertia tensor on a 105g body — unrealistically easy to tilt/spin for its
# mass. Scale linearly with the mass ratio (same assumed body geometry/
# radius of gyration, just heavier).
_CF2X_BASE_MASS = 0.027
_CF2X_BASE_INERTIA = (1.4e-5, 1.4e-5, 2.17e-5)  # ixx, iyy, izz
_INERTIA_SCALE = DRONE_MASS / _CF2X_BASE_MASS
DRONE_INERTIA = tuple(i * _INERTIA_SCALE for i in _CF2X_BASE_INERTIA)

_GRAVITY_FORCE = GRAVITY_ACCEL * DRONE_MASS
HOVER_RPM = (_GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5
MAX_RPM = (DRONE_THRUST2WEIGHT * _GRAVITY_FORCE / (4 * DRONE_KF)) ** 0.5

SCORE_REWARD = 50.0
CRASH_PENALTY = -50.0
COLLISION_PENALTY = -20.0
# Previously missing entirely: _check_done() ends the episode on "flipped"
# (tilt past MAX_TILT_RAD) but _compute_reward() had no matching branch, so
# a flip that didn't also touch the ground scored the same as any ordinary
# step. That left the policy with no direct incentive to avoid tipping over
# beyond the indirect cost of losing future progress reward — flip rate
# stayed at 75-90% of terminations across training regardless. -30 sits
# between COLLISION_PENALTY and CRASH_PENALTY: worse than bumping an
# opponent, but a ground crash is still the least recoverable failure.
FLIP_PENALTY = -30.0
# Missing is a worse *shot*, not a worse *failure* — the drone still flew
# the course successfully, it just didn't thread the hoop, so this stays
# lighter than the failure-mode penalties above.
MISS_PENALTY = -10.0

# Potential-based shaping: reward the change in distance to the goal each
# step, not the absolute distance. An absolute-distance penalty charges the
# same cost per step regardless of progress, which makes ending the episode
# early (e.g. tipping over on purpose) reward-optimal whenever the policy
# can't reliably reach the goal. Rewarding the delta instead means standing
# still costs ~0, not a growing negative total, removing that incentive.
PROGRESS_SCALE = 10.0

# Raw 3D distance-to-goal progress (above) rewards closing X-distance just as
# much as centering on the hoop's Y-Z opening, so a drone can look like it's
# "making progress" while drifting off-axis — and only find out it's
# mis-aligned when it crosses the goal plane. missed_goal became the single
# largest failure mode (52% of terminations) once FLIP_PENALTY/MISS_PENALTY
# fixed the flip-tolerance problem, more common than flipping itself. This
# adds dedicated potential-based shaping for lateral (Y-Z) alignment to the
# hoop centre, same style as PROGRESS_SCALE, but only once the drone is on
# final approach — applying it for the whole flight would fight the
# goalkeeper-avoidance detours needed earlier in the field.
LATERAL_SCALE = 10.0
FINAL_APPROACH_X = 2.0  # last third of the 0-3m field, past most dodging

# Fraction of the full throttle RPM range given to roll/pitch/yaw (see
# _apply_action). At 1.0, three simultaneous max-deflection axes could sum
# to +/-3x rpm_range before clipping — very easy to snap-roll past the tilt
# threshold before the policy can level out. Cutting this down makes the
# drone less twitchy so it has a real chance to recover instead of flipping
# every attempt.
ATTITUDE_GAIN = 0.4

# Tilt angle (radians) past which the drone is considered unrecoverable and
# the episode ends. Loosened twice now: 1.05 rad (~60 deg) originally, then
# 1.4 rad (~80 deg), now 1.5 rad (~86 deg) — nearly on its side — for even
# more room to recover before giving up on it.
MAX_TILT_RAD = 1.5  # ~86 degrees

# 3x3m field: X runs from the spawn line to the goal plane (see goal_pos in
# _load_scene), Y is centred on the spawn/goal line. Flying outside this
# footprint ends the episode (see _check_done) rather than being physically
# walled off.
FIELD_X_MIN = 0.0
FIELD_Y_HALF = 1.5

# Total opponents on the field, including the goalkeeper: opponent 0 always
# patrols in front of the hoop (unchanged); the rest fly randomly around the
# field as extra obstacles.
N_OPPONENTS = 4
RANDOM_OPPONENT_SPEED = 0.02  # metres moved per env step toward its current waypoint
# Random fliers stay inset from the true field edges so they don't spend all
# their time clipped against a boundary they just re-targeted past.
_RANDOM_OPPONENT_X_RANGE = (FIELD_X_MIN + 0.3, 2.7)
_RANDOM_OPPONENT_Y_RANGE = (-(FIELD_Y_HALF - 0.2), FIELD_Y_HALF - 0.2)


class DroneSoccerEnv(gym.Env):
    TIME_PENALTY = -0.02  # per-step cost, pushes the policy toward scoring quickly
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
        self._configure_physics()

        # --- Observation space ---
        # [drone pos(3), drone vel(3), drone orientation(4 quat)] = 10
        # + N_OPPONENTS * [opponent_detected(1), rel_x(1), rel_y(1), distance(1)]
        # one block per opponent, in self.opponent_ids order (goalkeeper
        # first, then the random fliers) — a fixed order so the observation
        # shape/layout stays consistent across episodes even though the
        # opponents' actual positions change.
        obs_dim = 10 + 4 * N_OPPONENTS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space ---
        # 4 continuous commands: [roll, pitch, throttle, yaw] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.drone_id = None
        self.opponent_ids = []
        self.step_count = 0
        self.max_steps = 1000

    def _configure_physics(self):
        """240Hz stepping + solver/velocity limits tuned for the Tello body.

        resetSimulation() drops engine-level parameters back to their
        defaults, so this needs to be called again after every reset, not
        just once in __init__.
        """
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            numSolverIterations=100,   # high precision contact constraints
            contactBreakingThreshold=0.005,
            physicsClientId=self._client,
        )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -GRAVITY_ACCEL, physicsClientId=self._client)
        self._configure_physics()
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._load_scene()
        self.step_count = 0

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
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)
        self._update_opponents()
        p.stepSimulation(physicsClientId=self._client)
        self.step_count += 1

        obs = self._get_obs()
        # Computed once and handed to _compute_reward() rather than each
        # deriving crash/scored independently — both used to check
        # drone_pos[2] < 0.05 and _scored() separately, which was redundant
        # and, worse, meant _compute_reward() couldn't see the "flipped"
        # outcome that only _check_done() knew about.
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
        """Overlay `text` above the drone in the GUI window and pause briefly
        so it's readable — resetSimulation() (called by reset()) wipes all
        debug items immediately, so without the pause the text would vanish
        the instant the caller resets. No-op outside render_mode="human"
        since addUserDebugText has no effect in DIRECT mode.
        """
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

    # ------------------------------------------------------------------ #
    # Scene / action / reward — fill these in for your specific task
    # ------------------------------------------------------------------ #
    def _load_scene(self):
        """Load drone, opponent drone, goal, field boundaries."""
        # Crazyflie 2.X body for both drones, re-tuned to Tello mass/handling
        # via changeDynamics below (no Tello URDF on hand, so this is the
        # closest stand-in geometry).
        # Spawn with a small margin off the FIELD_X_MIN line — sitting
        # exactly on the boundary means sub-millimetre physics noise (not
        # real movement) can trip the out-of-bounds check on step 1.
        start_pos = [FIELD_X_MIN + 0.1, 0, 1]
        self.drone_id = p.loadURDF(
            DRONE_URDF, start_pos, physicsClientId=self._client
        )
        p.changeDynamics(
            self.drone_id, linkIndex=-1,
            mass=DRONE_MASS,
            localInertiaDiagonal=DRONE_INERTIA,
            restitution=0.8,        # springy drone-soccer cage bounce
            lateralFriction=0.1,    # don't lock up against opponent frames
            linearDamping=0.6,      # mimics Tello's VPS/optical-flow hold
            angularDamping=0.8,
            physicsClientId=self._client,
        )

        # Goal: a red hoop to fly through, facing down the field's X axis.
        self.goal_pos = [3, 0, 1]
        self.goal_radius = 0.25
        self.goal_ids = self._create_hoop(self.goal_pos, radius=self.goal_radius)

        # Opponent 0 patrols left/right just in front of the hoop, like a
        # goalkeeper, blocking the striker's approach. opponent_ids[0] is
        # always this goalkeeper; the rest (see below) fly randomly.
        self.opponent_x = self.goal_pos[0] - 0.3
        self.opponent_z = 1.0
        self.opponent_amplitude = 0.6  # metres either side of centre
        self.opponent_angular_speed = 0.05  # radians per env step
        opponent_pos = [self.opponent_x, 0, self.opponent_z]
        goalkeeper_id = p.loadURDF(
            DRONE_URDF, opponent_pos, physicsClientId=self._client
        )
        p.changeDynamics(
            goalkeeper_id, linkIndex=-1,
            mass=DRONE_MASS, localInertiaDiagonal=DRONE_INERTIA,
            restitution=0.8, lateralFriction=0.1,
            physicsClientId=self._client,
        )
        self.opponent_ids = [goalkeeper_id]

        # Remaining opponents fly randomly around the field: each has its
        # own random waypoint and steps toward it at a constant speed each
        # frame, picking a new random waypoint on arrival (see
        # _update_opponents). Kinematic, same as the goalkeeper — these are
        # obstacles for the policy to learn around, not something trained.
        self.random_opponents = []
        for _ in range(N_OPPONENTS - 1):
            x = self.np_random.uniform(*_RANDOM_OPPONENT_X_RANGE)
            y = self.np_random.uniform(*_RANDOM_OPPONENT_Y_RANGE)
            z = self.np_random.uniform(0.7, 1.3)
            body_id = p.loadURDF(
                DRONE_URDF, [x, y, z], physicsClientId=self._client
            )
            p.changeDynamics(
                body_id, linkIndex=-1,
                mass=DRONE_MASS, localInertiaDiagonal=DRONE_INERTIA,
                restitution=0.8, lateralFriction=0.1,
                physicsClientId=self._client,
            )
            self.opponent_ids.append(body_id)
            self.random_opponents.append({
                "pos": np.array([x, y, z], dtype=np.float64),
                "target": self._random_opponent_waypoint(),
            })

        self._update_opponents()

    def _random_opponent_waypoint(self):
        """A random [x, y, z] target within the random fliers' fly zone."""
        return np.array([
            self.np_random.uniform(*_RANDOM_OPPONENT_X_RANGE),
            self.np_random.uniform(*_RANDOM_OPPONENT_Y_RANGE),
            self.np_random.uniform(0.7, 1.3),
        ], dtype=np.float64)

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

    def _update_opponents(self):
        """Kinematically move every opponent for this step: the goalkeeper
        sweeps left/right in front of the hoop, the rest fly randomly.

        Like `_apply_action`'s thrust override, this bypasses real physics
        (gravity would otherwise just drop their mass to the floor) and
        directly places each one every step, since these are scripted
        obstacles rather than something the policy needs to learn.
        """
        angle = self.step_count * self.opponent_angular_speed
        y = self.opponent_amplitude * np.sin(angle)
        goalkeeper_pos = [self.opponent_x, y, self.opponent_z]
        self._place_opponent(self.opponent_ids[0], goalkeeper_pos)

        for body_id, state in zip(self.opponent_ids[1:], self.random_opponents):
            to_target = state["target"] - state["pos"]
            dist = np.linalg.norm(to_target)
            if dist <= RANDOM_OPPONENT_SPEED:
                # Reached (or would overshoot) this waypoint — snap to it
                # and head somewhere new next step.
                state["pos"] = state["target"]
                state["target"] = self._random_opponent_waypoint()
            else:
                state["pos"] = state["pos"] + to_target / dist * RANDOM_OPPONENT_SPEED
            self._place_opponent(body_id, state["pos"])

    def _place_opponent(self, body_id, position):
        p.resetBasePositionAndOrientation(
            body_id, position, [0, 0, 0, 1], physicsClientId=self._client
        )
        p.resetBaseVelocity(
            body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0],
            physicsClientId=self._client,
        )

    def _apply_action(self, action):
        """Map normalized action [-1, 1]^4 to real per-motor thrust/torque.

        action[0] = Roll
        action[1] = Pitch
        action[2] = Throttle
        action[3] = Yaw
        """
        act_roll, act_pitch, act_throttle, act_yaw = np.clip(action, -1.0, 1.0)

        # Base throttle maps around the Tello's hover RPM.
        rpm_range = MAX_RPM - HOVER_RPM
        base_rpm = HOVER_RPM + act_throttle * rpm_range

        # Standard Crazyflie/Quad-X motor mixing. Roll/pitch/yaw are scaled
        # by a fraction (ATTITUDE_GAIN) of the same RPM authority as
        # throttle — without any scaling, a raw action in [-1, 1] shifts
        # RPM by at most 1 out of a base around 28,500+, i.e. ~0.003%
        # control authority, not enough to tilt the drone at all.
        attitude_range = rpm_range * ATTITUDE_GAIN
        rpm_0 = base_rpm + attitude_range * (-act_roll + act_pitch + act_yaw)
        rpm_1 = base_rpm + attitude_range * (-act_roll - act_pitch - act_yaw)
        rpm_2 = base_rpm + attitude_range * (act_roll - act_pitch + act_yaw)
        rpm_3 = base_rpm + attitude_range * (act_roll + act_pitch - act_yaw)

        rpms = np.clip(np.array([rpm_0, rpm_1, rpm_2, rpm_3]), 0, MAX_RPM)

        # RPM -> physical forces (N) and torques (N*m).
        forces = (rpms ** 2) * DRONE_KF
        torques = (rpms ** 2) * DRONE_KM

        total_torque_z = torques[0] - torques[1] + torques[2] - torques[3]

        # LINK_FRAME forces/torques are given in the drone's own body frame
        # and PyBullet rotates them into world space using its current
        # orientation. Each rotor's thrust is applied at its own prop link
        # (prop0_link..prop3_link, offset from the body's centre per
        # cf2x.urdf) rather than summed at the centre of mass — otherwise
        # the lever-arm is lost and differential thrust can never produce
        # roll/pitch torque, leaving the drone unable to translate.
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
        """Potential-based progress shaping toward the hoop, plus sparse
        score/crash/flip/collision/miss bonuses and penalties.

        Shaping rewards the change in distance to the goal each step, not
        the absolute distance — see PROGRESS_SCALE's comment for why.
        _prev_dist_to_goal/_prev_lateral_dist must be updated every call
        (including on the terminal branches, since collisions don't end the
        episode) so the next step's delta is measured against the right
        baseline.

        termination_reason comes from _check_done() (already computed once
        by step()) rather than being re-derived here, so this can see
        outcomes like "flipped" that aren't otherwise visible from
        drone_pos/_scored() alone.
        """
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
        elif termination_reason == "scored":
            reward = SCORE_REWARD
        elif termination_reason == "missed_goal":
            reward = MISS_PENALTY
        else:
            progress = self._prev_dist_to_goal - dist_to_goal
            reward = PROGRESS_SCALE * progress + self.TIME_PENALTY
            # See LATERAL_SCALE's comment: only on final approach, so this
            # doesn't fight goalkeeper-dodging earlier in the field.
            if drone_pos[0] >= FINAL_APPROACH_X:
                lateral_progress = self._prev_lateral_dist - lateral_dist
                reward += LATERAL_SCALE * lateral_progress

        self._prev_dist_to_goal = dist_to_goal
        self._prev_lateral_dist = lateral_dist
        return float(reward)

    def _scored(self, drone_pos):
        """True once the drone has flown through the goal hoop's opening."""
        through_plane = drone_pos[0] >= self.goal_pos[0]
        lateral_dist = np.linalg.norm(
            np.array(drone_pos[1:]) - np.array(self.goal_pos[1:])
        )
        return through_plane and lateral_dist <= self.goal_radius

    def _crashed(self):
        """True if the drone is touching the ground plane."""
        contacts = p.getContactPoints(
            bodyA=self.drone_id, bodyB=self.plane_id, physicsClientId=self._client
        )
        return len(contacts) > 0

    def _collided_with_opponent(self):
        """True if the drone is touching any opponent drone."""
        return any(
            len(p.getContactPoints(
                bodyA=self.drone_id, bodyB=opp_id, physicsClientId=self._client
            )) > 0
            for opp_id in self.opponent_ids
        )

    def _check_done(self):
        """Episode termination: scored, crashed, or flown past the hoop's plane.

        Returns a reason string once terminated ("crashed", "flipped",
        "scored", "missed_goal", "out_of_bounds"), or None while still
        in-flight — lets callers (step()'s info dict, watch.py, logging)
        report why an episode ended, not just that it did.
        """
        drone_pos, drone_orn = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )

        if drone_pos[2] < 0.05:
            return "crashed"

        drone_euler = p.getEulerFromQuaternion(drone_orn)
        if abs(drone_euler[0]) > MAX_TILT_RAD or abs(drone_euler[1]) > MAX_TILT_RAD:
            return "flipped"

        # Terminate once the drone has flown past the hoop's X plane at all,
        # whether it threaded the opening (see _scored, used for the score
        # bonus in _compute_reward) or missed wide — either way the shot is over.
        if drone_pos[0] >= self.goal_pos[0]:
            return "scored" if self._scored(drone_pos) else "missed_goal"

        # Out of bounds: flew behind the spawn line or off the side of the
        # 3x3m field. No physical wall — flying out just ends the episode.
        if drone_pos[0] < FIELD_X_MIN or abs(drone_pos[1]) > FIELD_Y_HALF:
            return "out_of_bounds"

        return None

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
        and returns N_OPPONENTS blocks of [detected_flag, rel_x, rel_y,
        distance], one per opponent, concatenated in self.opponent_ids order.

        With no detector attached, this uses PyBullet's ground-truth
        position of every opponent directly (useful for early sim-only
        training/debugging before your real detector is plugged in). The
        camera is only rendered when a detector is actually attached —
        otherwise it'd be dead work, and camera rendering is often the most
        expensive part of a step.
        """
        if self.detector is not None:
            rgb, depth = self._render_camera()
            # Expect detector(rgb) -> list of (x1, y1, x2, y2, conf, cls),
            # one entry per detected opponent. A real detector has no
            # persistent per-opponent identity across frames, so this just
            # takes up to N_OPPONENTS detections (highest confidence first)
            # and pads any remaining slots as "not detected".
            detections = sorted(self.detector(rgb), key=lambda d: d[4], reverse=True)
            blocks = []
            for i in range(N_OPPONENTS):
                if i >= len(detections):
                    blocks.append(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
                    continue
                x1, y1, x2, y2, conf, cls = detections[i]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                distance = self._depth_to_distance(depth[cy, cx])
                rel_x = (cx / self.img_size) * 2 - 1  # normalize to [-1, 1]
                rel_y = (cy / self.img_size) * 2 - 1
                blocks.append(np.array([1.0, rel_x, rel_y, distance], dtype=np.float32))
            return np.concatenate(blocks)

        # Fallback: ground-truth relative position of every opponent
        # (sim-only sanity check)
        drone_pos, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self._client
        )
        blocks = []
        for opp_id in self.opponent_ids:
            opp_pos, _ = p.getBasePositionAndOrientation(
                opp_id, physicsClientId=self._client
            )
            rel = np.array(opp_pos) - np.array(drone_pos)
            distance = np.linalg.norm(rel)
            blocks.append(np.array([1.0, rel[0], rel[1], distance], dtype=np.float32))
        return np.concatenate(blocks)

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
