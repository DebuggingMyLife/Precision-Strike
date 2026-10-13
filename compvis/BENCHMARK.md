# MobileNet on Tello video

Run from the Precision-Strike directory using its existing Python environment.
The script extracts 576 features using pretrained MobileNetV3-Small, with no
classification head. This measures compute speed, not hoop/drone recognition.

While connected to the internet, download weights and check inference:

```powershell
.\.venv\Scripts\python.exe .\compvis\benchmark_mobilenet.py --prepare
```

Then power on the Tello, keep it grounded, connect the computer to its Wi-Fi,
and close other Tello applications. Run:

```powershell
.\.venv\Scripts\python.exe .\compvis\benchmark_mobilenet.py --seconds 60
```

The preview shows the camera image and processing time, without detection boxes.
Press Q in the preview or Ctrl+C in the terminal to stop early. The script sends
SDK connection and camera commands; it never requests takeoff or movement.

Results are saved as timestamped CSV and JSON files under `runs/mobilenet`.
Weights are cached under `models/torch`. Both directories are already ignored by Git.

- `preprocess_ms`: RGB image conversion, official resize/crop/normalization,
  and device transfer (synchronized on CUDA).
- `inference_ms`: encoder forward pass, synchronized on CUDA.
- `gru_ms`: recurrent forward pass and state update, synchronized on CUDA
  (included only with `--gru`). `inference_ms` still means MobileNet alone.
- `processing_ms`: preprocessing plus MobileNet and, when enabled, GRU inference.
- `display_ms`: preview conversion, drawing and window update.
- `processed_fps`: distinct dequeued frames processed per measured wall-clock
  second, including waiting and display. Setup, warmup and cleanup are excluded.
- `processing_capacity_fps`: inverse mean processing time; excludes frame waiting
  and display, so it is not the observed video rate.
- `p95`: 95% of measured frames took this long or less.

A one-frame queue keeps the latest decoded frame and prevents processing the same
decoded frame repeatedly. Older frames can be discarded; this script cannot count
all dropped frames or measure the camera frame rate. It has no sensor timestamps,
so camera exposure, Wi-Fi delay, decoder buffering and frame age are not measured.
Do not label these numbers camera-to-action latency.

Defaults: CPU, four PyTorch threads, FP32, batch size one, 20 synthetic warmup
iterations, standard pretrained preprocessing (resize shorter edge to 256, center
crop 224, ImageNet normalization). Center cropping removes peripheral content;
this is a repeatable speed baseline, not a final navigation preprocessing design.

Use `--no-display` for a separate run without preview overhead; `--threads 1` or
`--threads 8` to compare CPU settings. Use `--device cuda` only with a CUDA-enabled
PyTorch installation and supported GPU. Compare repeated runs with similar scenes,
power settings and background load. Keep the per-run configuration in each report.

Dependencies: torch, torchvision, numpy, Pillow, opencv-python, djitellopy, av.
They are already present in the project's environment at setup time.

## MobileNet + GRU overhead experiment

Check both models without connecting to the drone:

```powershell
.\.venv\Scripts\python.exe .\compvis\benchmark_mobilenet.py --gru --prepare
```

Connect to the grounded Tello's Wi-Fi and run the same 60-second test:

```powershell
.\.venv\Scripts\python.exe .\compvis\benchmark_mobilenet.py --gru --seconds 60 --no-display
```

The optional GRU has one unidirectional layer, 576 input features, and 128 hidden
units. It processes one frame at a time, carrying its state to the next processed
frame. It starts from zero state each run, after discarding warmup state. Initialization
uses seed 0. `--hidden-size` and `--seed` allow explicit alternatives; the report
records them. Without `--gru`, the original MobileNet-only benchmark is used.

The GRU is **untrained**. This experiment measures compute overhead, not tracking,
navigation or learned memory. There is no policy, telemetry input or training loop.
Frames skipped by the one-frame queue are not seen by the GRU; no timestamps or
elapsed-time features are supplied. Results remain under `runs/mobilenet`, with
`model: mobilenet_v3_small+gru` and recurrent configuration in each JSON report.

Offline tests cover stream state continuity, reset between runs, timing accounting,
empty frames, connection failure and partial-report cleanup:

```powershell
.\.venv\Scripts\python.exe .\compvis\test_benchmark_mobilenet.py
```

References:
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- https://djitellopy.readthedocs.io/en/latest/tello/
- https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
