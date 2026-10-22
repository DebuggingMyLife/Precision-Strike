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
import csv
from dataclasses import dataclass
from collections import deque
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
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "policy_trace.csv"


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


def wrapped_yaw_delta_deg(current_deg: float, reference_deg: float) -> float:
    """Return the signed shortest yaw difference in degrees."""
    return (current_deg - reference_deg + 180.0) % 360.0 - 180.0


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
        self.yaw0_deg = safe_telemetry(drone, "get_yaw")

    def reset_episode(self) -> None:
        """Reset the sim-like state and yaw reference at takeoff."""
        self.last_time = time.monotonic()
        self.x_m = 0.1
        self.y_m = 0.0
        self.yaw0_deg = safe_telemetry(self.drone, "get_yaw")

    def observation(self, detection: Detection) -> np.ndarray:
        now = time.monotonic()
        dt = min(max(now - self.last_time, 0.0), 0.25)
        self.last_time = now

        # Tello speeds are cm/s in the body frame. Rotate horizontal velocity
        # into the local world frame using the reported yaw before integrating.
        vx = safe_telemetry(self.drone, "get_speed_x") / 100.0
        vy = safe_telemetry(self.drone, "get_speed_y") / 100.0
        vz = safe_telemetry(self.drone, "get_speed_z") / 100.0
        tello_yaw_deg = safe_telemetry(self.drone, "get_yaw")
        # Tello yaw is clockwise-positive; the simulator's yaw is
        # counter-clockwise-positive. Use a takeoff-relative simulator yaw.
        yaw = -math.radians(wrapped_yaw_delta_deg(tello_yaw_deg, self.yaw0_deg))
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


def action_to_rc(action: np.ndarray, scale: float,
                 forward_scale: float = 1.0) -> tuple[int, int, int, int]:
    """Map simulator [roll, pitch, throttle, yaw] to Tello RC channels.

    The simulator's negative pitch produces forward +X motion, so pitch is
    inverted for Tello. The simulator and Tello use opposite yaw signs, so yaw
    is inverted as well. ``forward_scale`` adjusts only the forward channel.
    """
    roll, pitch, throttle, yaw = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return (
        int(np.clip(round(roll * scale), -100, 100)),
        int(np.clip(round(-pitch * scale * forward_scale), -100, 100)),
        int(np.clip(round(throttle * scale), -100, 100)),
        int(np.clip(round(-yaw * scale), -100, 100)),
    )


def signed_direction(value: float, positive: str, negative: str,
                     threshold: float) -> str:
    if value > threshold:
        return positive
    if value < -threshold:
        return negative
    return "hold"


def format_agent_direction(rc: tuple[int, int, int, int]) -> str:
    """Describe the requested Tello-frame direction from the policy RC output."""
    roll, pitch, throttle, yaw = rc
    parts = [
        f"{signed_direction(roll, 'RIGHT', 'LEFT', 2)} {abs(roll)}",
        f"{signed_direction(pitch, 'FWD', 'BACK', 2)} {abs(pitch)}",
        f"{signed_direction(throttle, 'UP', 'DOWN', 2)} {abs(throttle)}",
        f"{signed_direction(yaw, 'CW', 'CCW', 2)} {abs(yaw)}",
    ]
    return " | ".join(parts)


def format_dominant_roll(roll_values: deque[float]) -> str:
    """Display the average roll direction over the recent display window."""
    if not roll_values:
        return "DOMINANT: hold 0.0"
    mean_roll = float(np.mean(roll_values))
    direction = signed_direction(mean_roll, "RIGHT", "LEFT", 2.0)
    return f"DOMINANT: {direction} {mean_roll:+.1f}"


def format_drone_direction(speed_x: float, speed_y: float, speed_z: float,
                           yaw_step_deg: float) -> str:
    """Describe observed Tello telemetry directions and values."""
    return " | ".join([
        f"{signed_direction(speed_x, 'FWD', 'BACK', 1.0)} {speed_x:+.1f}",
        f"{signed_direction(speed_y, 'RIGHT', 'LEFT', 1.0)} {speed_y:+.1f}",
        f"{signed_direction(speed_z, 'UP', 'DOWN', 1.0)} {speed_z:+.1f}",
        f"{signed_direction(yaw_step_deg, 'CW', 'CCW', 0.5)} {yaw_step_deg:+.1f}deg",
    ])


def altitude_hold_command(target_cm: float, height_cm: float,
                          deadband_cm: float, gain: float,
                          max_speed: float) -> int:
    """Return a Tello up/down command that holds the requested height."""
    error_cm = target_cm - height_cm
    if abs(error_cm) <= deadband_cm:
        return 0
    return int(np.clip(round(error_cm * gain), -max_speed, max_speed))


def close_range_rc(rc: tuple[int, int, int, int], detection: Detection,
                   close_range_m: float, minimum_scale: float) -> tuple[int, int, int, int]:
    """Slow forward approach while preserving RPPO lateral and yaw commands."""
    if detection.label != "goal" or detection.distance_m <= 0 or close_range_m <= 0:
        return rc
    scale = np.clip(detection.distance_m / close_range_m, minimum_scale, 1.0)
    roll, forward, vertical, yaw = rc
    return roll, int(round(forward * scale)), vertical, yaw


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
    parser.add_argument("--forward-scale", type=float, default=3.0,
                        help="Multiplier for the forward RC channel only.")
    parser.add_argument("--takeoff-height-cm", type=float, default=105.0,
                        help="Target height after takeoff in centimeters.")
    parser.add_argument("--ascent-speed", type=int, default=15,
                        help="Upward RC speed used only during initial ascent.")
    parser.add_argument("--altitude-band-cm", type=float, default=6.0,
                        help="Height deadband in which the vertical command is zero.")
    parser.add_argument("--altitude-gain", type=float, default=0.8,
                        help="Proportional gain for vertical height correction.")
    parser.add_argument("--close-range-m", type=float, default=0.8,
                        help="Goal distance at which approach and yaw commands begin slowing.")
    parser.add_argument("--close-range-min-scale", type=float, default=0.25,
                        help="Minimum horizontal/yaw scale at close range.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Policy/RC update rate; Tello's control channel is 20 Hz.")
    parser.add_argument("--enable-control", action="store_true",
                        help="Permit policy RC commands after pressing T to take off.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample PPO actions instead of using deterministic actions.")
    parser.add_argument("--csv-log", type=Path, default=None,
                        help="CSV trace path; defaults to lab-live-test/logs/policy_trace.csv and overwrites it each run.")
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
            or args.focal_length_px <= 0 or args.altitude_band_cm < 0
            or args.altitude_gain <= 0
            or args.close_range_m <= 0
            or not 0 < args.close_range_min_scale <= 1
            or args.forward_scale <= 0):
        raise ValueError("invalid timing, RC, camera, or calibration argument")

    log_path = args.csv_log.expanduser() if args.csv_log else DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
    log_file = None
    try:
        drone.connect()
        print(f"Battery: {drone.get_battery()}%")
        drone.streamon()
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        log_file = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "timestamp",
                "raw_rel_x",
                "rel_x",
                "raw_action_roll",
                "raw_action_pitch",
                "raw_action_throttle",
                "raw_action_yaw",
                "rc_roll",
                "rc_pitch",
                "rc_throttle",
                "rc_yaw",
                "yaw_minus_yaw0",
                "airborne",
            ],
        )
        log_writer.writeheader()
        log_file.flush()
        adapter = TelloObservationAdapter(drone)
        recurrent_state = None
        episode_start = True
        yaw0_deg = None
        previous_yaw_deg = None
        dominant_roll_window: deque[float] = deque(maxlen=max(1, round(args.hz)))
        next_tick = time.monotonic()
        print(f"Policy: {policy_path}")
        print(f"YOLO:   {weights_path}")
        print(f"CSV:    {log_path}")
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
            raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
            if raw_action.size != 4:
                raise ValueError(f"policy returned {raw_action.size} action values; expected 4")
            predicted_rc = action_to_rc(action, args.rc_scale, args.forward_scale)
            dominant_roll_window.append(float(predicted_rc[0]))
            approach_rc = close_range_rc(
                predicted_rc,
                detection,
                args.close_range_m,
                args.close_range_min_scale,
            )
            current_height_cm = safe_telemetry(drone, "get_height")
            ascending = (
                airborne and takeoff_time is not None
                and current_height_cm < args.takeoff_height_cm - args.altitude_band_cm
                and time.monotonic() - takeoff_time < 12.0
            )
            if ascending:
                rc_command = (0, 0, args.ascent_speed, 0)
            else:
                vertical_rc = altitude_hold_command(
                    args.takeoff_height_cm,
                    current_height_cm,
                    args.altitude_band_cm,
                    args.altitude_gain,
                    args.rc_scale,
                )
                rc_command = (approach_rc[0], approach_rc[1], vertical_rc, approach_rc[3])

            current_yaw_deg = safe_telemetry(drone, "get_yaw")
            speed_x = safe_telemetry(drone, "get_speed_x")
            speed_y = safe_telemetry(drone, "get_speed_y")
            speed_z = safe_telemetry(drone, "get_speed_z")
            yaw_step_deg = (
                0.0 if previous_yaw_deg is None
                else wrapped_yaw_delta_deg(current_yaw_deg, previous_yaw_deg)
            )
            previous_yaw_deg = current_yaw_deg
            yaw_minus_yaw0 = "" if yaw0_deg is None else (
                wrapped_yaw_delta_deg(current_yaw_deg, yaw0_deg)
            )
            log_writer.writerow({
                "timestamp": time.time(),
                "raw_rel_x": detection.rel_x,
                "rel_x": detection.rel_x,
                "raw_action_roll": raw_action[0],
                "raw_action_pitch": raw_action[1],
                "raw_action_throttle": raw_action[2],
                "raw_action_yaw": raw_action[3],
                "rc_roll": rc_command[0],
                "rc_pitch": rc_command[1],
                "rc_throttle": rc_command[2],
                "rc_yaw": rc_command[3],
                "yaw_minus_yaw0": yaw_minus_yaw0,
                "airborne": int(airborne),
            })
            log_file.flush()
            if airborne and args.enable_control:
                drone.send_rc_control(*rc_command)

            annotated = result.plot()
            status = "CONTROL" if args.enable_control else "DRY RUN"
            if airborne:
                status += " AIRBORNE"
            else:
                status += " GROUNDED"
            cv2.putText(annotated, f"AGENT: {format_agent_direction(predicted_rc)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)
            cv2.putText(annotated, format_dominant_roll(dominant_roll_window), (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)
            cv2.putText(annotated, f"DRONE: {format_drone_direction(speed_x, speed_y, speed_z, yaw_step_deg)}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 0), 2)
            cv2.putText(annotated, f"{status} | H {current_height_cm:.0f}/{args.takeoff_height_cm:.0f}cm | RC {rc_command}", (10, 25),
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
                adapter.reset_episode()
                yaw0_deg = adapter.yaw0_deg
                previous_yaw_deg = yaw0_deg
                recurrent_state = None
                episode_start = True
            elif key in (ord("l"), ord("L")) and airborne:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                airborne = False
                takeoff_time = None
                yaw0_deg = None
                previous_yaw_deg = None
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
            if log_file is not None:
                log_file.close()
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
