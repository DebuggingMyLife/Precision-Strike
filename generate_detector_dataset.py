"""
Generates a YOLO-format auto-labeled dataset for the defender detector by
driving the sim with random actions and projecting the defender's known 3D
position into the onboard camera's pixel space — no manual annotation
needed. Writes images/{train,val} + labels/{train,val} + dataset.yaml under
OUT_DIR, ready for train_detector.py.

Run:
    python generate_detector_dataset.py
"""

import os
import numpy as np
import pybullet as p
from PIL import Image

from drone_soccer_env import DroneSoccerEnv

FOV_DEG = 90  # must match _render_camera's fov
# Approximate bounding radius for labeling, not a real physics constant —
# cf2x's rotor arm length (its own "arm" property), used as a stand-in
# silhouette size since the frame isn't a sphere with one true radius.
DRONE_BOUNDING_RADIUS = 0.04
OUT_DIR = "detector_dataset"
N_TRAIN = 4000
N_VAL = 500
MAX_EPISODE_STEPS = 300  # shorter than training's max_steps; just need variety


def _project_to_pixels(world_pos, view_matrix, proj_matrix, img_size):
    """World-space point -> (px, py, in_front) via PyBullet's OpenGL-style
    column-major view/projection matrices."""
    view = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    proj = np.array(proj_matrix, dtype=np.float64).reshape(4, 4, order="F")
    clip = proj @ (view @ np.array([*world_pos, 1.0]))
    if clip[3] <= 0:
        return None, None, False
    ndc = clip[:3] / clip[3]
    px = (ndc[0] * 0.5 + 0.5) * img_size
    py = (1 - (ndc[1] * 0.5 + 0.5)) * img_size
    return px, py, True


def _label_frame(env, view_matrix, proj_matrix):
    """Returns (x1, y1, x2, y2) in pixels for the defender, or None if it's
    not in view — angular size from the pinhole model, since the cage is a
    sphere of known radius."""
    drone_pos, _ = p.getBasePositionAndOrientation(
        env.drone_id, physicsClientId=env._client
    )
    defender_pos = env._defender_pos
    distance = float(np.linalg.norm(np.array(defender_pos) - np.array(drone_pos)))
    cx, cy, in_front = _project_to_pixels(
        defender_pos, view_matrix, proj_matrix, env.img_size
    )
    if not in_front or not (0 <= cx <= env.img_size and 0 <= cy <= env.img_size):
        return None

    half_fov_rad = np.radians(FOV_DEG / 2)
    angular_radius = np.arcsin(min(DRONE_BOUNDING_RADIUS / distance, 1.0))
    pixel_radius = (angular_radius / half_fov_rad) * (env.img_size / 2)
    x1, y1 = cx - pixel_radius, cy - pixel_radius
    x2, y2 = cx + pixel_radius, cy + pixel_radius
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(env.img_size, x2), min(env.img_size, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _write_yolo_label(path, box, img_size):
    if box is None:
        open(path, "w").close()  # empty label = no object in this frame
        return
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / img_size, (y1 + y2) / 2 / img_size
    w, h = (x2 - x1) / img_size, (y2 - y1) / img_size
    with open(path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def generate_split(env, split, n_frames):
    img_dir = os.path.join(OUT_DIR, "images", split)
    label_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    env.reset()
    episode_step = 0
    n_with_defender = 0
    for i in range(n_frames):
        action = env.action_space.sample()
        env._apply_action(action)
        env._update_defender()
        p.stepSimulation(physicsClientId=env._client)
        env.step_count += 1
        episode_step += 1

        rgb, _ = env._render_camera()
        drone_pos, drone_orn = p.getBasePositionAndOrientation(
            env.drone_id, physicsClientId=env._client
        )
        rot_matrix = p.getMatrixFromQuaternion(drone_orn)
        forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        cam_target = [drone_pos[j] + forward_vec[j] for j in range(3)]
        view_matrix = p.computeViewMatrix(drone_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=FOV_DEG, aspect=1.0, nearVal=0.05, farVal=20
        )
        box = _label_frame(env, view_matrix, proj_matrix)
        if box is not None:
            n_with_defender += 1

        Image.fromarray(rgb.astype(np.uint8)).save(
            os.path.join(img_dir, f"{split}_{i:06d}.png")
        )
        _write_yolo_label(
            os.path.join(label_dir, f"{split}_{i:06d}.txt"), box, env.img_size
        )

        done = env._check_done() is not None
        if done or episode_step >= MAX_EPISODE_STEPS:
            env.reset()
            episode_step = 0

        if (i + 1) % 500 == 0:
            print(f"[{split}] {i + 1}/{n_frames} frames "
                  f"({n_with_defender} with defender visible)")

    print(f"[{split}] done: {n_frames} frames, "
          f"{n_with_defender} with defender visible "
          f"({100 * n_with_defender / n_frames:.1f}%)")


def write_dataset_yaml():
    yaml_path = os.path.join(OUT_DIR, "dataset.yaml")
    abs_out = os.path.abspath(OUT_DIR)
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {abs_out}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: defender\n"
        )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    env = DroneSoccerEnv(detector=lambda rgb: [], render_mode=None)
    generate_split(env, "train", N_TRAIN)
    generate_split(env, "val", N_VAL)
    env.close()
    write_dataset_yaml()
