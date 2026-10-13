"""Exercise benchmark accounting and cleanup without connecting to hardware."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import benchmark_mobilenet as benchmark


class Clock:
    def __init__(self):
        self.time = 0

    def __call__(self):
        self.time += 0.001
        return self.time


class Reader:
    def __init__(self, frames):
        self.frames = iter(frames)

    @property
    def frame(self):
        return next(self.frames, None)


class BenchmarkTests(unittest.TestCase):
    def args(self, directory):
        return SimpleNamespace(seconds=1, frame_timeout=0.01, no_display=True,
                               warmup=20, output=Path(directory), gru=False,
                               hidden_size=128, seed=0)

    def test_gru_stream_matches_sequence_and_resets_between_runs(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.gru = True
            with patch('builtins.print'):
                gru = benchmark.load_gru(args, device)
            sequence = torch.linspace(-1, 1, 2 * 576).reshape(1, 2, 576)
            with torch.inference_mode():
                expected_output, expected_hidden = gru(sequence)
            for run in range(2):
                with self.subTest(run=run):
                    drone = Mock()
                    drone.get_frame_read.return_value = Reader([frame, None, frame.copy()])
                    drone.get_battery.return_value = 80
                    outputs, incoming_states = [], []
                    def observe(features, hidden):
                        self.assertFalse(torch.is_grad_enabled())
                        incoming_states.append(hidden)
                        output, updated = gru(features, hidden)
                        outputs.append(output)
                        return output, updated
                    spy = Mock(side_effect=observe)
                    encoder = Mock(side_effect=[sequence[:, 0, :], sequence[:, 1, :]])
                    transform = Mock(return_value=torch.zeros(3, 224, 224))
                    with patch('djitellopy.Tello', return_value=drone), \
                            patch.object(benchmark, 'perf_counter', Clock()), \
                            patch.object(benchmark, 'sleep'), patch('builtins.print'):
                        with self.assertRaises(TimeoutError):
                            benchmark.run_live(args, encoder, transform, device, spy)
                    self.assertEqual(spy.call_count, 2)
                    self.assertIsNone(incoming_states[0])
                    torch.testing.assert_close(incoming_states[1], outputs[0].transpose(0, 1))
                    torch.testing.assert_close(torch.cat(outputs, dim=1), expected_output)
                    torch.testing.assert_close(outputs[-1].transpose(0, 1), expected_hidden)
                    drone.end.assert_called_once()
            reports = [json.loads(p.read_text()) for p in Path(directory).glob('*.json')]
            self.assertEqual(len(reports), 2)
            for report in reports:
                self.assertEqual(report['model'], 'mobilenet_v3_small+gru')
                self.assertFalse(report['recurrent']['trained'])
                self.assertEqual(report['output_features'], 128)
                timings = report['timings_ms']
                self.assertGreater(timings['gru_ms']['mean'], 0)
                self.assertAlmostEqual(timings['processing_ms']['mean'],
                                       timings['preprocess_ms']['mean'] +
                                       timings['inference_ms']['mean'] +
                                       timings['gru_ms']['mean'])

    def test_fresh_frames_timeout_partial_report_and_cleanup(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        # One empty read between two distinct frames must not count as inference.
        reader = Reader([frame, None, frame.copy()])
        drone = Mock()
        drone.get_frame_read.return_value = reader
        drone.get_battery.return_value = 80
        clock = Clock()
        # A long cleanup delay must not contaminate throughput.
        def cleanup():
            clock.time += 100
        drone.end.side_effect = cleanup
        encoder = Mock(return_value=torch.zeros(1, 576))
        transform = Mock(return_value=torch.zeros(3, 224, 224))
        with tempfile.TemporaryDirectory() as directory:
            with patch('djitellopy.Tello', return_value=drone), \
                    patch.object(benchmark, 'perf_counter', clock), \
                    patch.object(benchmark, 'sleep'), patch('builtins.print'):
                with self.assertRaises(TimeoutError):
                    benchmark.run_live(self.args(directory), encoder, transform, torch.device('cpu'))
            summary = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(summary['frames_processed'], 2)
            self.assertEqual(encoder.call_count, 2)
            self.assertLess(summary['elapsed_seconds'], 1)
            self.assertIn('TimeoutError', summary['stop_reason'])
            self.assertGreater(summary['timings_ms']['processing_ms']['mean'], 0)
            drone.get_frame_read.assert_called_once_with(with_queue=True, max_queue_len=1)
            drone.end.assert_called_once()
            drone.takeoff.assert_not_called()
            drone.send_rc_control.assert_not_called()

    def test_no_frames_produces_no_report(self):
        drone = Mock()
        drone.get_frame_read.return_value = Reader([])
        with tempfile.TemporaryDirectory() as directory:
            with patch('djitellopy.Tello', return_value=drone), \
                    patch.object(benchmark, 'perf_counter', Clock()), \
                    patch.object(benchmark, 'sleep'), patch('builtins.print'):
                with self.assertRaises(TimeoutError):
                    benchmark.run_live(self.args(directory), Mock(), Mock(), torch.device('cpu'))
            self.assertEqual(list(Path(directory).iterdir()), [])
            drone.end.assert_called_once()

    def test_connection_failure_cleans_up(self):
        drone = Mock()
        drone.connect.side_effect = RuntimeError('not connected')
        with tempfile.TemporaryDirectory() as directory:
            with patch('djitellopy.Tello', return_value=drone), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'not connected'):
                    benchmark.run_live(self.args(directory), Mock(), Mock(), torch.device('cpu'))
            drone.end.assert_called_once()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
