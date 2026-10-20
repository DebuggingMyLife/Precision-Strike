"""Run the trained YOLO + recurrent PPO pair against a live Tello stream.

The default mode is observation-only: it predicts and displays RC commands but
never sends them. Add --enable-control to permit commands, then press T to
take off. Press L to land and Q/Esc to stop.

The real-drone observation and action adapters are necessarily approximate:
the PPO policy was trained with simulated world state and motor actions, while
Tello exposes telemetry and velocity-based RC commands. Use this runner first
to log and inspect the adapter output before enabling flight control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from sb3_contrib import RecurrentPPO
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = ROOT / "models" / "drone_soccer_ppo_v4.1.zip"
DEFAULT_WEIGHTS = ROOT / "models" / "tello_defender_goal_best.pt"


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """Convert radians in roll/pitch/yaw order to the simulator's quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def safe_telemetry(drone, method: str, default: float = 0.0) -> float:
    try:
        value = getattr(drone, method)()
        return float(value)
    except Exception:
        return default


@dataclass
class Detection:
    label: str
    present: float
    rel_x: float
    rel_y: float
    distance_m: float
    box: tuple[int, int, int, int] | None
    confidence: float


def choose_policy_target(result, width: int, height: int, target_width_m: float,
                         goal_width_m: float, focal_length_px: float) -> Detection:
    """Select the highest-confidence defender/goal box like the simulator."""
    boxes = result.boxes
    names = result.names or {}
    best = None
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index])
        label = str(names.get(class_id, class_id)).lower() if isinstance(names, dict) else str(class_id)
        if label not in ("defender", "goal"):
            continue
        confidence = float(boxes.conf[index])
        if best is None or confidence > best[0]:
            best = (confidence, label, boxes.xyxy[index].tolist())

    if best is None:
        return Detection("none", 0.0, 0.0, 0.0, -1.0, None, 0.0)

    confidence, label, raw_box = best
    x1, y1, x2, y2 = (int(round(value)) for value in raw_box)
    cx = max(0, min(width - 1, (x1 + x2) // 2))
    cy = max(0, min(height - 1, (y1 + y2) // 2))
    rel_x = 2.0 * cx / max(width, 1) - 1.0
    rel_y = 2.0 * cy / max(height, 1) - 1.0
    pixel_width = max(abs(x2 - x1), 1)
    physical_width = target_width_m if label == "defender" else goal_width_m
    distance_m = max(0.05, focal_length_px * physical_width / pixel_width)
    return Detection(label, 1.0, rel_x, rel_y, distance_m, (x1, y1, x2, y2), confidence)


class TelloObservationAdapter:
    """Approximate the 14-value simulator observation from Tello telemetry."""

    def __init__(self, drone):
        self.drone = drone
        self.last_time = time.monotonic()
        self.x_m = 0.1
        self.y_m = 0.0

    def observation(self, detection: Detection) -> np.ndarray:
        now = time.monotonic()
        dt = min(max(now - self.last_time, 0.0), 0.25)
        self.last_time = now

        # Tello speeds are cm/s in the body frame. Rotate horizontal velocity
        # into the local world frame using the reported yaw before integrating.
        vx = safe_telemetry(self.drone, "get_speed_x") / 100.0
        vy = safe_telemetry(self.drone, "get_speed_y") / 100.0
        vz = safe_telemetry(self.drone, "get_speed_z") / 100.0
        yaw = math.radians(safe_telemetry(self.drone, "get_yaw"))
        self.x_m += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
        self.y_m += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt

        height_m = max(safe_telemetry(self.drone, "get_height") / 100.0, 0.05)
        roll = math.radians(safe_telemetry(self.drone, "get_roll"))
        pitch = math.radians(safe_telemetry(self.drone, "get_pitch"))
        quaternion = euler_to_quaternion(roll, pitch, yaw)
        return np.asarray(
            [
                self.x_m, self.y_m, height_m,
                vx, vy, vz,
                *quaternion,
                detection.present, detection.rel_x, detection.rel_y,
                detection.distance_m,
            ],
            dtype=np.float32,
        )


def action_to_rc(action: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    """Map simulator [roll, pitch, throttle, yaw] to Tello RC channels.

    The simulator's negative pitch produces forward +X motion. Tello's
    forward/backward channel uses the opposite sign, so pitch is inverted here.
    """
    roll, pitch, throttle, yaw = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return tuple(
        int(np.clip(round(value * scale), -100, 100))
        for value in (roll, -pitch, throttle, yaw)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--target-width-m", type=float, default=0.17,
                        help="Physical defender width used for monocular distance.")
    parser.add_argument("--goal-width-m", type=float, default=0.45,
                        help="Physical goal width used for monocular distance.")
    parser.add_argument("--focal-length-px", type=float, default=550.0,
                        help="Calibrated camera focal length in pixels.")
    parser.add_argument("--rc-scale", type=float, default=20.0,
                        help="Maximum RC velocity generated by a full policy action.")
    parser.add_argument("--takeoff-height-cm", type=float, default=150.0,
                        help="Target height after takeoff in centimeters.")
    parser.add_argument("--ascent-speed", type=int, default=20,
                        help="Upward RC speed used only during initial ascent.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Policy/RC update rate; Tello's control channel is 20 Hz.")
    parser.add_argument("--enable-control", action="store_true",
                        help="Permit policy RC commands after pressing T to take off.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample PPO actions instead of using deterministic actions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = args.policy.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if (args.hz <= 0 or args.rc_scale <= 0
            or args.takeoff_height_cm <= 0 or not 1 <= args.ascent_speed <= 100
            or args.target_width_m <= 0 or args.goal_width_m <= 0
            or args.focal_length_px <= 0):
        raise ValueError("invalid timing, RC, camera, or calibration argument")

    try:
        from djitellopy import Tello
    except ImportError as error:
        raise RuntimeError("Install djitellopy before running this test.") from error

    detector = YOLO(str(weights_path))
    policy = RecurrentPPO.load(str(policy_path), device="cpu")
    drone = Tello()
    airborne = False
    takeoff_time = None
    reader = None
    try:
        drone.connect()
        print(f"Battery: {drone.get_battery()}%")
        drone.streamon()
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        adapter = TelloObservationAdapter(drone)
        recurrent_state = None
        episode_start = True
        next_tick = time.monotonic()
        print(f"Policy: {policy_path}")
        print(f"YOLO:   {weights_path}")
        print("Observation-only mode: predicted RC commands are displayed, not sent.")
        if args.enable_control:
            print("Control enabled: T=takeoff to target altitude, L=land.")
        else:
            print("Add --enable-control only after validating the adapter output.")

        while True:
            frame_rgb = reader.frame
            if frame_rgb is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                time.sleep(0.02)
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame_bgr.shape[:2]
            result = detector.predict(frame_bgr, conf=args.confidence, device="cpu", verbose=False)[0]
            detection = choose_policy_target(
                result, width, height, args.target_width_m, args.goal_width_m,
                args.focal_length_px
            )
            observation = adapter.observation(detection)
            action, recurrent_state = policy.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=not args.stochastic,
            )
            episode_start = False
            predicted_rc = action_to_rc(action, args.rc_scale)
            ascending = (
                airborne and takeoff_time is not None
                and safe_telemetry(drone, "get_height") < args.takeoff_height_cm - 5.0
                and time.monotonic() - takeoff_time < 12.0
            )
            if ascending:
                rc_command = (0, 0, args.ascent_speed, 0)
            else:
                rc_command = predicted_rc
            if airborne and args.enable_control:
                drone.send_rc_control(*rc_command)

            annotated = result.plot()
            status = "CONTROL" if args.enable_control else "DRY RUN"
            if airborne:
                status += " AIRBORNE"
            else:
                status += " GROUNDED"
            cv2.putText(annotated, f"{status} | RC {rc_command}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated, f"{detection.label}={detection.confidence:.2f} dist={detection.distance_m:.2f}m",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Tello YOLO + PPO live test", np.asarray(annotated))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("t"), ord("T")) and args.enable_control and not airborne:
                drone.takeoff()
                airborne = True
                takeoff_time = time.monotonic()
                recurrent_state = None
                episode_start = True
            elif key in (ord("l"), ord("L")) and airborne:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                airborne = False
                takeoff_time = None
                recurrent_state = None
                episode_start = True

            next_tick += 1.0 / args.hz
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        try:
            if airborne:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
        finally:
            try:
                drone.streamoff()
            except Exception:
                pass
            drone.end()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
