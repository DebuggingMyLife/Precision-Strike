# Vision integration review — 2 October 2026

Reviewed the local `compVis` checkout, including the actual training/environment
code and installed sb3-contrib 2.9.0. The README describes an older skeleton;
the current environment already contains a hoop, opponents and motor mixing.

## Verified starting point

- `train.py` uses `sb3_contrib.RecurrentPPO`, `MlpLstmPolicy`, 128 LSTM units,
  and `detector=None`. It does not currently train on camera images.
- Observations contain 26 floats: position (3), velocity (3), quaternion (4),
  and four opponent blocks (4 each). Opponent positions come from the simulator,
  including opponents outside the camera view. There is no explicit visual hoop input.
- The detector path expects boxes plus simulator depth, so it cannot accept a
  MobileNet feature vector as a drop-in detector. Its relative coordinates also
  have different meanings from the ground-truth fallback.
- `_render_camera()` already produces RGB and depth using TinyRenderer. Its
  default resolution is 84x84; the smoke check explicitly used 256x256 and then
  MobileNet's pretrained 224x224 center-crop preprocessing.
- Actions are roll/pitch/throttle/yaw inputs mixed into motor RPM/thrust. They
  are not the Tello SDK's lateral/forward/vertical/yaw RC interface.
- Each `step()` advances physics once at 1/240 simulated second. The physical
  camera benchmark processed about 24 frames/second. Training currently assumes
  a much faster observation/action cadence than the measured live pipeline.
- The installed recurrent PPO implementation stores LSTM hidden AND cell states
  for actor/critic and uses them in sequence training, terminal bootstrapping and
  rollout buffers. Replacing the LSTM with `nn.GRU` alone is insufficient.

## Completed smoke check

`inspect_sim_vision.py` resets a seeded headless simulation, renders the spawn
camera, extracts 576 finite MobileNet features, and checks ten physics steps.
Artifacts: `runs/vision_review/spawn_camera.png` and `summary.json`.
The image shows the red hoop and distant drone silhouettes against a simple
checkerboard scene. This verifies connectivity, not visual task performance.

The first check exposed RGB pixels returned as int64 on this PyBullet build.
`_render_camera()` now explicitly returns uint8 RGB, preserving pixel values and
making the image usable by PIL/MobileNet. The corrected check passed.
No training was launched and no checkpoint was changed.

## Proposed implementation sequence

1. **Add a separate visual observation environment.** Preserve the 26-float
   baseline. Return an observation dictionary containing RGB and a clearly
   defined telemetry subset. Exclude privileged opponent positions and depth
   from the deployable policy. Simulator truth remains valid for reward and
   evaluation. World position is not directly available from ordinary Tello
   telemetry; define units, coordinate frames and normalization before choosing
   the final telemetry inputs. This step is independent of GRU versus LSTM.
2. **Verify camera geometry and timing.** Match real image aspect ratio/FOV and
   mount position; make the camera up vector rotate with the body (currently
   world-up). Check hoop visibility and peripheral cropping. Retain 240 Hz
   physics but choose an explicit slower policy cadence, initially around
   20–24 Hz for experiments. Reapply forces and update opponents during physics
   substeps. Audit episode duration, reward aggregation and discounting when
   changing the control interval. Measured processed FPS is not sensor FPS.
3. **Build the visual learning baseline.** Initially freeze the pretrained
   MobileNet encoder, then feed its features plus telemetry to the recurrent
   policy. For the requested GRU path, implement compatible recurrent PPO
   state storage, contiguous sequences, padding masks, episode resets and
   timeout bootstrapping, with state-continuity and gradient checks. An optional
   MobileNet + existing LSTM control experiment can isolate the impact of vision
   with less trainer modification. Do not stack an untrained GRU before the LSTM.
4. **Run a small training smoke test, then a curriculum.** Start with a single
   stationary hoop and configurable opponents disabled. Use separate output
   names and new visual-policy checkpoints; the existing 26-input checkpoints
   are not directly compatible. Confirm learning on varied starts before adding
   opponents. Evaluate scoring, collisions, completion time and multiple seeds.
5. **Prepare physical transfer.** Introduce realistic appearances, camera noise,
   delays and frame gaps. Resolve the motor-control versus Tello RC mismatch
   using a suitable simulated controller/action interface before physical policy
   execution. Benchmark the entire policy and measure actual camera delay.

The live CPU benchmark (MobileNet + untrained GRU-128) achieved 24.22 processed
FPS, 9.39 ms mean preprocessing + model time, 11.06 ms p95, and 0.17 ms mean GRU
time. This supports further experiments but establishes neither learned memory,
collision avoidance nor physical control feasibility.

## Visual observation milestone

Implemented `compvis.visual_env.VisualDroneSoccerEnv`: a separate RGB + eight-value
telemetry dictionary, with no privileged opponent/goal/depth observations. The
camera uses a configurable 4:3 image, a forward mount offset and body-relative up.
See `VISUAL_ENV.md` for the exact telemetry and camera contract and commands.
The original training entry points still use the 26-number environment.

The next implementation milestone is aligning the control cadence and action
interface before wiring up recurrent visual-policy training. Camera calibration,
real telemetry mapping and recurrence training remain separate outstanding work.
