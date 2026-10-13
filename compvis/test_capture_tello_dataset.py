"""Offline capture tests: no real camera, GUI or network required."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from compvis import capture_tello_dataset as capture


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 0.1
        return self.value


class Reader:
    def __init__(self, frames):
        self.frames = iter(frames)

    @property
    def frame(self):
        return next(self.frames, None)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = capture.parse_args(['--output', self.temp.name, '--auto', '--no-display'])
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.frame[:, :, 0] = 240  # RGB red: catches incorrect RGB/BGR conversion.
        self.frame[10:20, 10:20, 1] = 80

    def read_manifest(self, session):
        return [json.loads(line) for line in (session.path / 'frames.jsonl').read_text().splitlines()]

    def test_lossless_original_pixels_unique_sessions_and_metadata(self):
        self.args.session = '../same scene'
        session = capture.CaptureSession(self.args)
        another = capture.CaptureSession(self.args)
        self.assertNotEqual(session.path, another.path)
        self.assertEqual(session.path.parent, Path(self.temp.name).resolve())
        try:
            with patch('builtins.print'):
                self.assertTrue(session.save(self.frame, 1, '2026-10-06T00:00:00+00:00', 0.5, 'manual'))
                self.assertFalse(session.save(self.frame, 1, '', 1, 'automatic'))
            row = self.read_manifest(session)[0]
            decoded = cv2.imdecode(np.frombuffer((session.path / row['file']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), self.frame)
            self.assertEqual((row['width'], row['height']), (64, 48))
            self.assertEqual(row['trigger'], 'manual')
        finally:
            session.close('test_complete')
            another.close('test_complete')
        metadata = json.loads((session.path / 'session.json').read_text())
        self.assertEqual(metadata['images_saved'], 1)
        self.assertEqual(metadata['annotation_status'], 'unlabeled')

    def test_auto_rate_and_max_images(self):
        self.args.max_images = 3
        session = capture.CaptureSession(self.args)
        try:
            with patch.object(capture, 'perf_counter', Clock()), patch('builtins.print'):
                reason = capture.capture_stream(self.args, Reader([self.frame] * 50), session)
            self.assertEqual(reason, 'max_images')
            records = self.read_manifest(session)
            self.assertEqual(len(records), 3)
            self.assertGreaterEqual(records[1]['elapsed_seconds'] - records[0]['elapsed_seconds'], 0.5)
            self.assertGreaterEqual(records[2]['elapsed_seconds'] - records[1]['elapsed_seconds'], 0.5)
            self.assertEqual(len({row['dequeued_frame_index'] for row in records}), 3)
        finally:
            session.close('test_complete')

    def test_lost_stream_does_not_save_stale_frame(self):
        self.args.frame_timeout = 0.5
        session = capture.CaptureSession(self.args)
        try:
            with patch.object(capture, 'perf_counter', Clock()), patch.object(capture, 'sleep'), patch('builtins.print'):
                with self.assertRaises(TimeoutError):
                    capture.capture_stream(self.args, Reader([self.frame]), session)
            self.assertEqual(len(self.read_manifest(session)), 1)
        finally:
            session.close('error: TimeoutError: test')
        self.assertEqual(json.loads((session.path / 'session.json').read_text())['status'], 'error')

    def test_snapshot_waits_for_fresh_frame_and_preview_does_not_modify_save(self):
        self.args.auto = False
        self.args.no_display = False
        session = capture.CaptureSession(self.args)
        try:
            with patch.object(capture, 'perf_counter', Clock()), patch.object(capture, 'sleep'), \
                    patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'getWindowProperty', return_value=1), \
                    patch.object(cv2, 'waitKey', side_effect=[ord('s'), -1, ord('a'), ord('q')]), \
                    patch('builtins.print'):
                reason = capture.capture_stream(self.args, Reader([None, self.frame, self.frame.copy(), None]), session)
            self.assertEqual(reason, 'user_quit')
            records = self.read_manifest(session)
            self.assertEqual([row['trigger'] for row in records], ['manual', 'automatic'])
            for row in records:
                decoded = cv2.imdecode(np.frombuffer((session.path / row['file']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), self.frame)
        finally:
            session.close('test_complete')

    def test_encoding_failure_does_not_create_image_or_manifest_record(self):
        session = capture.CaptureSession(self.args)
        try:
            with patch.object(cv2, 'imencode', return_value=(False, None)):
                with self.assertRaises(OSError):
                    session.save(self.frame, 1, '', 0, 'manual')
            self.assertEqual(self.read_manifest(session), [])
            self.assertEqual(list((session.path / 'images').iterdir()), [])
        finally:
            session.close('error: encoding')

    def test_connection_failure_still_closes_drone_and_records_session_error(self):
        drone = Mock()
        drone.connect.side_effect = RuntimeError('offline test')
        with patch('djitellopy.Tello', return_value=drone), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'offline test'):
                capture.run(self.args)
        drone.end.assert_called_once()
        drone.takeoff.assert_not_called()
        drone.send_rc_control.assert_not_called()
        metadata = json.loads(next(Path(self.temp.name).glob('*/session.json')).read_text())
        self.assertEqual(metadata['status'], 'error')
        self.assertEqual(metadata['images_saved'], 0)

    def test_invalid_cli_options(self):
        for options in [['--fps', 'nan'], ['--fps', '0'], ['--no-display'], ['--seconds', '-1']]:
            with self.subTest(options=options), patch('sys.stderr'), self.assertRaises(SystemExit):
                capture.parse_args(options)


if __name__ == '__main__':
    unittest.main()
