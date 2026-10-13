# Visual observation environment

`VisualDroneSoccerEnv` provides camera RGB and an eight-value telemetry vector.
It is a separate subclass; `train.py` still uses the original 26-number environment.
No PPO training or GRU training is introduced by this change.

From the project root:

```python
from compvis.visual_env import VisualDroneSoccerEnv

env = VisualDroneSoccerEnv(render_mode="rgb_array")
try:
    obs, info = env.reset(seed=0)
    image = obs["rgb"]          # uint8, HWC, RGB, shape (240, 320, 3)
    telemetry = obs["telemetry"]  # float32, shape (8,)
    preview = env.render()     # independent copy of the latest camera observation
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
finally:
    env.close()
```

The observation contains no opponent coordinates, goal coordinates, global X/Y
position, depth map, or detector boxes. Simulator ground truth still determines
rewards, collisions and episode completion. The telemetry uses ideal simulated
sensors; real sensor noise and delays have not been added.

| Index | Value | Convention / normalization |
|---|---|---|
| 0 | Height above ground | World Z above the fixed z=0 floor, divided by 3 m, clipped to [0,1] |
| 1–3 | Body-relative linear velocity | World velocity rotated into body axes; divided by 5 m/s, clipped to [-1,1] |
| 4–5 | sin(roll), cos(roll) | Roll from simulator quaternion, radians |
| 6–7 | sin(pitch), cos(pitch) | Pitch from simulator quaternion, radians |

Body axes are +X forward, +Y left, +Z up. The 3 m and 5 m/s scales are explicit
design choices, not claims about Tello sensor limits. Yaw is excluded from the
telemetry. A real Tello adapter must validate telemetry axis/sign/unit conventions
and reference height before supplying equivalent values. Simulator velocity and
height must not be assumed to equal raw Tello SDK values.

The 320x240 image preserves a 4:3 aspect ratio. The camera looks along body +X,
with body +Z as up, so the view rolls and pitches with the drone. Its default
mount offset is (0.06, 0, 0) m in body coordinates. Resolution, vertical FOV
(90 degrees by default), and mount offset are configurable. These are provisional
simulation settings, not a calibrated Tello camera model. The renderer uses a
0.02 m near plane and 20 m far plane. RGB is explicitly uint8 across PyBullet builds.

Use the same MobileNet pretrained resize/crop/normalization as the live benchmark.
That center crop removes peripheral pixels, so the visual policy's final
preprocessing needs separate evaluation. A future SB3 adapter must also handle
HWC-to-CHW conversion and avoid normalizing pixels twice.

Physics, opponents, rewards and action semantics are inherited without changes:
one `step()` is 1/240 simulated second; actions mix roll/pitch/throttle/yaw into
motor thrust. This milestone does not yet implement a 20–24 Hz controller or
Tello RC command mapping. Existing 26-input checkpoints cannot be loaded against
this new observation space as-is.

## Offline verification and preview

No drone connection is needed. Using the project's existing environment:

```powershell
.\.venv\Scripts\python.exe -m unittest compvis.test_visual_env -v
.\.venv\Scripts\python.exe .\compvis\inspect_sim_vision.py --visual
```

The diagnostic checks simulator RGB -> pretrained MobileNet -> 576 finite features
and ten simulation steps. It saves `runs/visual_env/spawn_camera.png` and
`runs/visual_env/summary.json`. It uses cached MobileNet weights when available.
Tests check the Gymnasium contract, telemetry coordinate transforms and clipping,
camera rotation, observation isolation, and equivalence of baseline physics,
rewards and termination. These checks establish plumbing, not task learning.
