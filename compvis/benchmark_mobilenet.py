"""Benchmark MobileNetV3-Small with optional untrained GRU on grounded Tello video."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter, sleep

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true',
                        help='Download weights and check inference without connecting to Tello.')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads.')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--gru', action='store_true', help='Include an untrained streaming GRU.')
    parser.add_argument('--hidden-size', type=int, default=128, help='GRU hidden units (default: 128).')
    parser.add_argument('--seed', type=int, default=0, help='Random GRU initialization seed.')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--frame-timeout', type=float, default=10)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'mobilenet')
    args = parser.parse_args()
    if (args.seconds <= 0 or args.warmup < 1 or args.threads < 1
            or args.frame_timeout <= 0 or args.hidden_size < 1):
        parser.error('Seconds, warmup, threads, hidden size and frame timeout must be positive.')
    return args


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_encoder(args):
    torch.set_num_threads(args.threads)
    # Keep downloaded weights in this project, including when run by another user.
    torch.hub.set_dir(str(ROOT / 'models' / 'torch'))
    selected = args.device
    if selected == 'auto':
        selected = 'cuda' if torch.cuda.is_available() else 'cpu'
    if selected == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Use --device cpu.')
    device = torch.device(selected)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    network = mobilenet_v3_small(weights=weights)
    # Preserve the convolutional backbone and pooling; remove classification layers.
    encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
    encoder = encoder.to(device).eval()
    transform = weights.transforms()
    sample = transform(Image.fromarray(np.zeros((720, 960, 3), dtype=np.uint8)))
    sample = sample.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            features = encoder(sample)
        synchronize(device)
    if features.shape != (1, 576) or not torch.isfinite(features).all().item():
        raise RuntimeError('Encoder check failed: expected 576 finite features per frame.')
    print(f'Encoder ready: {device}, FP32, 224x224, batch 1, 576 features.', flush=True)
    return encoder, transform, device


def load_gru(args, device):
    if not args.gru:
        return None
    # Reproducible CPU initialization without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        gru = nn.GRU(576, args.hidden_size, num_layers=1, batch_first=True)
    gru = gru.to(device).eval()
    with torch.inference_mode():
        sample = torch.zeros(1, 1, 576, device=device)
        hidden = None
        for _ in range(args.warmup):
            output, hidden = gru(sample, hidden)
        synchronize(device)
    if output.shape != (1, 1, args.hidden_size) or not torch.isfinite(output).all().item():
        raise RuntimeError('GRU inference check failed.')
    # Warmup state is discarded. run_live starts a new state for every run.
    print(f'GRU ready: 576 -> {args.hidden_size}, one layer, untrained (compute test only).', flush=True)
    return gru


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {name: float(value) for name, value in {
        'mean': array.mean(), 'p50': np.percentile(array, 50),
        'p95': np.percentile(array, 95), 'max': array.max(),
    }.items()}


def save_results(args, rows, elapsed, device, reason, battery):
    args.output.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    csv_path = args.output / f'{stem}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    timing_keys = ['preprocess_ms', 'inference_ms']
    if args.gru:
        timing_keys.append('gru_ms')
    timing_keys.extend(['processing_ms', 'display_ms'])
    metrics = {key: stats([row[key] for row in rows]) for key in timing_keys}
    summary = {
        'source': 'live_tello', 'model': 'mobilenet_v3_small+gru' if args.gru else 'mobilenet_v3_small',
        'weights': 'IMAGENET1K_V1', 'encoder_output_features': 576,
        'output_features': args.hidden_size if args.gru else 576,
        'recurrent': {
            'type': 'GRU', 'hidden_size': args.hidden_size, 'layers': 1,
            'batch_first': True, 'sequence_length_per_call': 1,
            'bidirectional': False, 'trained': False, 'initialization_seed': args.seed,
            'state_handling': 'Zero initial state per run, carried between processed frames; warmup state discarded.',
        } if args.gru else None,
        'input_shape': [1, 3, 224, 224], 'dtype': 'float32',
        'preprocessing': 'RGB; bilinear resize shorter edge 256; center crop 224; ImageNet normalization',
        'device': str(device), 'cpu_threads': torch.get_num_threads(),
        'processor': platform.processor(), 'platform': platform.platform(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'opencv': cv2.__version__,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'warmup_iterations_excluded': args.warmup,
        'preview_enabled': not args.no_display, 'battery_start_percent': battery,
        'frames_processed': len(rows), 'elapsed_seconds': elapsed,
        'requested_seconds': args.seconds, 'stop_reason': reason,
        'processed_fps': len(rows) / elapsed,
        'processing_capacity_fps': 1000 / metrics['processing_ms']['mean'],
        'timings_ms': metrics,
        'limitations': [
            'No sensor capture timestamp: camera, network, decoder and queue age are unmeasured.',
            'Processed FPS includes frame waiting and display; it is not camera FPS.',
            'A one-frame queue discards older decoded frames; dropped frames are not counted.',
            'Processing starts after dequeue; timings do not measure camera-to-action latency.',
            'Speed benchmark only: no detection, navigation accuracy or policy evaluation.',
            'Optional GRU has random weights; timing it does not demonstrate learned temporal understanding.',
        ],
    }
    json_path = args.output / f'{stem}.json'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n{len(rows)} frames / {elapsed:.2f}s: {summary["processed_fps"]:.2f} processed FPS')
    for name, values in metrics.items():
        print(f'{name}: mean {values["mean"]:.2f}, p50 {values["p50"]:.2f}, p95 {values["p95"]:.2f} ms')
    print(f'Stop: {reason}\nSummary: {json_path}\nFrame timings: {csv_path}')
    print('These are host processing measurements, not camera-to-action latency.')


def run_live(args, encoder, transform, device, gru=None):
    from djitellopy import Tello

    if args.gru != (gru is not None):
        raise ValueError('GRU configuration does not match the supplied model.')
    hidden = None
    drone = Tello()
    rows = []
    started = None
    finished = None
    battery = None
    reason = 'duration_complete'
    failure = None
    try:
        print('Connecting to grounded Tello. Camera benchmark only. Q exits preview.', flush=True)
        drone.connect()
        battery = drone.get_battery()
        print(f'Battery: {battery}%', flush=True)
        drone.streamon()
        # Pop each decoded frame at most once, keeping only the newest queued frame.
        # Re-reading the default .frame could falsely count one camera frame repeatedly.
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        first_deadline = perf_counter() + args.frame_timeout
        last_frame = perf_counter()
        last_report = last_frame
        with torch.inference_mode():
            while True:
                now = perf_counter()
                if started is not None and now - started >= args.seconds:
                    break
                frame = reader.frame  # DJITelloPy's decoded frames are RGB.
                if frame is None:
                    if (started is None and now >= first_deadline) or (
                            started is not None and now - last_frame >= args.frame_timeout):
                        raise TimeoutError('No fresh camera frames; check Tello Wi-Fi and stream.')
                    if not args.no_display and started is not None:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            reason = 'user_quit'
                            break
                    sleep(0.001)
                    continue
                begin = perf_counter()
                if started is None:
                    started = begin
                    last_report = begin
                last_frame = begin
                # Use the pretrained weights' official preprocessing.
                tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
                synchronize(device)
                prepared = perf_counter()
                features = encoder(tensor)
                synchronize(device)
                inferred = perf_counter()
                processed = inferred
                if gru is not None:
                    # (batch=1, sequence=1, features=576); preserve memory across frames.
                    _, hidden = gru(features.unsqueeze(1), hidden)
                    synchronize(device)
                    processed = perf_counter()
                processing_ms = (processed - begin) * 1000
                quit_requested = False
                if not args.no_display:
                    preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = f'MobileNet + GRU({args.hidden_size})' if args.gru else 'MobileNet'
                    text = f'{label} | processing {processing_ms:.1f} ms | Q: quit'
                    cv2.putText(preview, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.imshow('Tello MobileNet benchmark', preview)
                    quit_requested = cv2.waitKey(1) & 0xFF == ord('q')
                    quit_requested |= cv2.getWindowProperty(
                        'Tello MobileNet benchmark', cv2.WND_PROP_VISIBLE) < 1
                end = perf_counter()
                rows.append({
                    'frame_index': len(rows), 'elapsed_s': end - started,
                    'source_width': frame.shape[1], 'source_height': frame.shape[0],
                    'preprocess_ms': (prepared - begin) * 1000,
                    'inference_ms': (inferred - prepared) * 1000,
                    'processing_ms': processing_ms,
                    'display_ms': (end - processed) * 1000,
                })
                if gru is not None:
                    rows[-1]['gru_ms'] = (processed - inferred) * 1000
                if end - last_report >= 5:
                    print(f'{end - started:.0f}s: {len(rows) / (end - started):.1f} processed FPS', flush=True)
                    last_report = end
                if quit_requested:
                    reason = 'user_quit'
                    break
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except Exception as error:
        reason = f'error: {type(error).__name__}: {error}'
        failure = error
    finally:
        finished = perf_counter()
        # Connection/model setup and stream shutdown are outside the timed run.
        try:
            drone.end()
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()
            if rows:
                save_results(args, rows, finished - started, device, reason, battery)
    if failure:
        raise failure
    if not rows:
        print('No frames measured; no benchmark report was created.')


def main():
    args = parse_args()
    encoder, transform, device = load_encoder(args)
    gru = load_gru(args, device)
    if args.prepare:
        print('Weights cached and synthetic inference check passed. No Tello measurements taken.')
        return
    run_live(args, encoder, transform, device, gru)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Benchmark failed: {error}', file=sys.stderr)
        sys.exit(1)
