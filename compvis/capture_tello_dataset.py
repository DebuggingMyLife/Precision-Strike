"""Capture clean Tello images, optionally with keyboard flight via --fly."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter, sleep
from uuid import uuid4

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOW = 'Tello dataset capture'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets' / 'tello_raw')
    parser.add_argument('--session', default='capture', help='Scene/session name (a unique suffix is added).')
    parser.add_argument('--notes', default='', help='Lighting, location, objects, etc., stored locally.')
    parser.add_argument('--auto', action='store_true', help='Start timed capture immediately.')
    parser.add_argument('--fps', type=float, default=2.0, help='Timed save rate, default 2 images/second.')
    parser.add_argument('--seconds', type=float, default=0, help='Stop after this duration; 0 means unlimited.')
    parser.add_argument('--max-images', type=int, default=0, help='Stop at this image count; 0 means unlimited.')
    parser.add_argument('--frame-timeout', type=float, default=10.0)
    parser.add_argument('--no-display', action='store_true', help='Timed capture without a preview.')
    parser.add_argument('--fly', action='store_true', help='Enable keyboard flight in the live preview.')
    parser.add_argument('--speed', type=int, default=100, help='Flight RC input magnitude, 1-100 (default 100).')
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 30:
        parser.error('--fps must be finite and between 0 (exclusive) and 30.')
    if not math.isfinite(args.seconds) or args.seconds < 0 or args.max_images < 0:
        parser.error('--seconds and --max-images must be non-negative and finite.')
    if not math.isfinite(args.frame_timeout) or args.frame_timeout <= 0:
        parser.error('--frame-timeout must be positive and finite.')
    if args.no_display and not args.auto:
        parser.error('--no-display requires --auto because snapshot keys need the preview.')
    if args.fly and args.no_display:
        parser.error('--fly requires the live preview for keyboard control.')
    if not 1 <= args.speed <= 100:
        parser.error('--speed must be between 1 and 100.')
    return args


class CaptureSession:
    """Each session is independent and images are written before their manifest rows."""

    def __init__(self, args):
        label = re.sub(r'[^a-zA-Z0-9_-]+', '-', args.session).strip('-_')[:60] or 'capture'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.path = args.output.resolve() / f'{stamp}_{label}_{uuid4().hex[:8]}'
        (self.path / 'images').mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.frames_seen = 0
        self.last_saved_sequence = None
        self.metadata = {
            'schema_version': 1, 'status': 'connecting', 'created_utc': utc_now(),
            'source': 'tello_live_camera', 'session_name': args.session, 'notes': args.notes,
            'format': 'png', 'original_resolution': True, 'overlays_saved': False,
            'automatic_capture_initially': args.auto, 'requested_save_fps': args.fps,
            'requested_seconds': args.seconds, 'max_images': args.max_images,
            'preview_enabled': not args.no_display, 'battery_start_percent': None,
            'timestamp_definition': 'Host time when decoded frame was dequeued, not sensor exposure time.',
            'annotation_status': 'unlabeled',
            'flight_enabled': args.fly, 'rc_input_magnitude': args.speed if args.fly else None,
        }
        self._write_metadata()
        self.manifest = (self.path / 'frames.jsonl').open('x', encoding='utf-8')

    def _write_metadata(self):
        pending = self.path / 'session.json.tmp'
        pending.write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        pending.replace(self.path / 'session.json')

    def start(self, battery):
        self.metadata.update(status='capturing', capture_started_utc=utc_now(), battery_start_percent=battery)
        self._write_metadata()

    def save(self, rgb, sequence, received_utc, elapsed, trigger):
        if sequence == self.last_saved_sequence:
            return False
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected a uint8 RGB image with three channels.')
        # DJITelloPy supplies RGB; OpenCV's encoder expects BGR. No resize/overlay.
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError('PNG encoding failed; image was not saved.')
        filename = f'frame_{self.count + 1:06d}.png'
        destination = self.path / 'images' / filename
        pending = destination.with_suffix('.png.tmp')
        try:
            with pending.open('xb') as file:
                file.write(encoded.tobytes())
            # rename on Windows refuses to overwrite an existing destination.
            if destination.exists():
                raise FileExistsError(destination)
            pending.rename(destination)
        finally:
            if pending.exists():
                pending.unlink()
        record = {
            'file': f'images/{filename}', 'dequeued_frame_index': sequence,
            'received_utc': received_utc, 'saved_utc': utc_now(),
            'elapsed_seconds': elapsed, 'trigger': trigger,
            'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
        }
        self.manifest.write(json.dumps(record) + '\n')
        self.manifest.flush()
        self.count += 1
        self.last_saved_sequence = sequence
        print(f'Saved {filename} ({trigger})', flush=True)
        return True

    def close(self, reason):
        self.manifest.close()
        self.metadata.update(
            status='error' if reason.startswith('error:') else 'finished',
            stop_reason=reason, finished_utc=utc_now(), images_saved=self.count,
            frames_dequeued=self.frames_seen,
        )
        self._write_metadata()


def capture_stream(args, reader, session, flight=None, display=None):
    started = perf_counter()
    last_frame_time = started
    next_save = started
    automatic = args.auto
    pending_snapshot = False
    preview = None
    if not args.no_display and display is None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        now = perf_counter()
        if args.seconds and now - started >= args.seconds:
            if not session.frames_seen:
                raise TimeoutError('Capture duration elapsed without any camera frames.')
            return 'duration_complete'
        if args.max_images and session.count >= args.max_images:
            return 'max_images'
        frame = reader.frame  # Queue mode pops each fresh frame at most once.
        received_utc = utc_now() if frame is not None else None
        if frame is not None:
            last_frame_time = now
            session.frames_seen += 1
            if not args.no_display and display is None:
                preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif now - last_frame_time >= args.frame_timeout:
            raise TimeoutError('No fresh video frames; check Tello Wi-Fi and camera stream.')

        if display is not None:
            fresh = session.frames_seen > 0 and now - last_frame_time < 0.5
            requests, keys, focused = display.poll(
                frame, automatic, args.fps, session.count, args.speed,
                flight.possibly_airborne, fresh,
            )
            if 'quit' in requests:
                return 'user_quit'
            if 'land' in requests:
                flight.land()
            elif 'takeoff' in requests and focused and fresh:
                flight.takeoff()
            flight.update(keys, focused=focused, video_fresh=fresh)
            if 'snapshot' in requests:
                pending_snapshot = True
            if 'auto' in requests:
                automatic = not automatic
                next_save = now
                print(f'Automatic capture {"ON" if automatic else "OFF"}', flush=True)
        elif not args.no_display:
            canvas = preview.copy() if preview is not None else np.zeros((360, 640, 3), dtype=np.uint8)
            mode = f'AUTO {args.fps:g}/s' if automatic else 'MANUAL'
            status = f'{mode} | Saved {session.count} | S/Space: save | A: auto | Q/Esc: quit'
            cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if frame is None:
                cv2.putText(canvas, 'Waiting for fresh frame...', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 'user_quit'
            if key in (ord('s'), ord('S'), 32):
                pending_snapshot = True
            if key in (ord('a'), ord('A')):
                automatic = not automatic
                next_save = now
                print(f'Automatic capture {"ON" if automatic else "OFF"}', flush=True)

        due = automatic and now >= next_save
        if frame is not None and (pending_snapshot or due):
            trigger = 'manual' if pending_snapshot else 'automatic'
            session.save(frame, session.frames_seen, received_utc, now - started, trigger)
            pending_snapshot = False
            # Never catch up with bursts of near-identical frames after a delay.
            next_save = perf_counter() + 1 / args.fps
        if frame is None:
            sleep(0.001)


def run(args):
    from djitellopy import Tello

    session = CaptureSession(args)
    drone = None
    flight = None
    display = None
    reason = 'startup_interrupted'
    try:
        print(f'Session: {session.path}', flush=True)
        if args.fly:
            if __package__:
                from .tello_capture_flight import FlightControl, FlightDisplay
            else:
                from tello_capture_flight import FlightControl, FlightDisplay
            display = FlightDisplay()
            print(f'Flight enabled: RC input {args.speed}. Press T to take off; L to land.', flush=True)
        else:
            print('Camera capture only. Keep the drone grounded; close other Tello apps.', flush=True)
        drone = Tello()
        if args.fly:
            flight = FlightControl(drone, args.speed)
        drone.connect()
        battery = drone.get_battery()
        print(f'Battery: {battery}%', flush=True)
        drone.streamon()
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        session.start(battery)
        reason = capture_stream(args, reader, session, flight, display)
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except Exception as error:
        reason = f'error: {type(error).__name__}: {error}'
        raise
    finally:
        try:
            try:
                if flight is not None:
                    flight.land()
            finally:
                if drone is not None:
                    drone.end()
        except Exception as error:
            reason = f'error: shutdown {type(error).__name__}: {error}'
            raise
        finally:
            try:
                if display is not None:
                    display.close()
                elif not args.no_display:
                    cv2.destroyAllWindows()
            finally:
                session.close(reason)
                print(f'Finished: {reason}\nSaved {session.count} images in {session.path}', flush=True)


if __name__ == '__main__':
    try:
        run(parse_args())
    except Exception as error:
        print(f'Capture failed: {error}', file=sys.stderr)
        sys.exit(1)
