# Capture Tello images for YOLO annotation

This standalone script saves original-resolution camera frames as lossless PNGs.
It needs OpenCV, NumPy and DJITelloPy (already installed in the project environment).
It does not load YOLO, predict labels or upload images. Flight commands are enabled
only with `--fly`; capture-only mode remains the default.
PNG avoids additional lossy compression; the Tello video itself is already compressed.

## Run

Power on the grounded Tello, connect the laptop to its Wi-Fi, and close other
Tello applications. Internet is not needed. In PowerShell:

```powershell
cd "C:\Users\ADMIN\OneDrive\Documents\programming\university\RL\a3_group\Precision-Strike"
.\.venv\Scripts\python.exe .\compvis\capture_tello_dataset.py --session hoop-room1
```

Click the preview to focus it:

- **S or Space**: save one fresh image. If no frame is available, the request waits
  for the next decoded frame rather than saving a stale preview.
- **A**: turn automatic capture on/off (default target: 2 images/second).
- **Q or Esc**, or close the window: stop and save the session summary.
- **Ctrl+C** in the terminal also stops and finalizes the session.

Start automatic capture at two images/second for one minute:

```powershell
.\.venv\Scripts\python.exe .\compvis\capture_tello_dataset.py --session drones-room1 --auto --fps 2 --seconds 60 --notes "Indoor lighting, cages fitted, varied distances"
```

Use `--max-images 200` to stop at a count. `--seconds 0` (the default) runs until
you quit. `--no-display` requires `--auto`; Ctrl+C remains available. `--output`
chooses another parent folder. No folders or images from previous runs are overwritten.

## Flying while capturing

Flight mode uses pygame (already installed here) for held-key control and the live
camera preview in one window. Close other drone controllers, connect to Tello Wi-Fi,
and start grounded:

```powershell
.\.venv\Scripts\python.exe .\compvis\capture_tello_dataset.py --fly --speed 100 --session flight-session
```

Click the flight preview to focus it. **Flight-mode keys differ from capture-only mode**:

| Key | Action |
|---|---|
| T | Take off (requires a fresh camera frame) |
| L | Land |
| W / S | Forward / backward |
| A / D | Left / right |
| Up / Down arrows | Ascend / descend |
| Left / Right arrows | Rotate left / right |
| Space | Hover while held |
| C | Save one clean snapshot |
| R | Toggle automatic image capture |
| Q / Esc / window close | Attempt landing, then quit |

The default `--speed 100` is the maximum RC channel magnitude, **not 100 m/s**.
You can choose 1–100 with `--speed`. Movement stops on key release. The loop aims
for 20 control updates per second, although image encoding and UI work can add
delay. Space, loss of window focus, or video older than 0.5 seconds sends zero RC
input. A prolonged stream timeout, Ctrl+C, image limit, duration limit or exception
attempts landing before disconnecting. Landing requires a working drone link; a
forced process kill cannot run this cleanup. This script does not automatically
take off: press T explicitly. It does not provide autonomous collision avoidance.

Add `--auto --fps 2` to start saving automatically; otherwise use C and R.
`--no-display` is unavailable in flight mode. Saved PNGs remain free of UI overlays.

## Files

Each run creates its own unique session directory under `datasets/tello_raw`:

```text
<UTC timestamp>_<session name>_<unique suffix>/
    images/
        frame_000001.png
        frame_000002.png
    frames.jsonl
    session.json
```

`frames.jsonl` records each image's dimensions, manual/automatic trigger, frame
index and timestamps. Timestamps describe host dequeue/save time, not camera
exposure time. `session.json` records notes, starting battery, settings, final
count and stop reason. Preview overlays are drawn on a separate copy and are
never included in saved images. RGB/BGR conversion preserves correct colours.

The single-frame queue discards old decoded frames. Each dequeued frame can be
saved at most once. Timed capture is a maximum target rate; encoding, disk speed
and frame availability can reduce the actual rate. It does not save bursts to
catch up after delays. Stream loss beyond `--frame-timeout` (default 10 seconds)
ends the run with an error; earlier images remain available. Abrupt power loss or
process termination may leave metadata marked `capturing`; inspect the images
and manifest before using such a session. A `.tmp` file is not a finished image.

These images are **unlabeled**, not negative examples. The script deliberately
does not create empty YOLO label files or invent class IDs. Keep session groups
when splitting train/validation/test. If an annotation tool requires a flat image
directory, prefix filenames with their session IDs to avoid name collisions.

## Collection checklist

Vary drone/hoop distance, angle, background, lighting and partial occlusion.
Include multiple objects and empty scenes. Move targets or reposition the grounded
camera between shots rather than collecting hundreds of nearly identical frames.
Keep representative blur and difficult examples; do not save only frames that an
existing detector recognises. Annotate all relevant visible objects consistently.
Agree with the training handler on class IDs and whether hoop boxes cover the
outer structure or the opening before labeling. Reserve entire sessions for testing.

Data stays in your chosen local directory. The default directory is inside this
OneDrive workspace and may sync through your existing OneDrive settings. Use
`--output` outside OneDrive if you want a folder outside that sync location.
The default raw dataset directory is excluded from Git.

Offline tests (no drone required):

```powershell
.\.venv\Scripts\python.exe -m unittest compvis.test_capture_tello_dataset -v
```
