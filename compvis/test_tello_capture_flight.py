"""Offline flight-control tests. Never contacts a real drone."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from compvis.capture_tello_dataset import parse_args, run
from compvis.tello_capture_flight import FlightControl
import numpy as np


class FlightTests(unittest.TestCase):
    def test_headless_live_preview_and_capture_key_events(self):
        from compvis.tello_capture_flight import FlightDisplay
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy'}):
            display = FlightDisplay()
            try:
                frame = np.full((48, 64, 3), [240, 80, 10], dtype=np.uint8)
                original = frame.copy()
                pg = display.pg
                pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_c))
                pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_r))
                requests, _, _ = display.poll(frame, False, 2, 0, 100, False, True)
                self.assertEqual(requests, {'snapshot', 'auto'})
                np.testing.assert_array_equal(frame, original)
                pg.event.post(pg.event.Event(pg.QUIT))
                requests, _, _ = display.poll(None, False, 2, 0, 100, False, False)
                self.assertIn('quit', requests)
            finally:
                display.close()

    def test_maximum_rc_mapping_and_opposing_keys(self):
        drone = Mock()
        control = FlightControl(drone)
        control.update({'w'})
        drone.send_rc_control.assert_not_called()
        control.takeoff()
        control.takeoff()
        drone.takeoff.assert_called_once()
        control.update({'d', 'w', 'up', 'right'})
        drone.send_rc_control.assert_called_with(100, 100, 100, 100)
        control.update({'a', 's', 'down', 'left'})
        drone.send_rc_control.assert_called_with(-100, -100, -100, -100)
        control.update({'w', 's', 'a', 'd', 'up', 'down', 'left', 'right'})
        drone.send_rc_control.assert_called_with(0, 0, 0, 0)

    def test_hover_on_key_release_space_focus_loss_and_stale_video(self):
        for keys, focused, fresh in [(set(), True, True), ({'w', 'space'}, True, True),
                                      ({'w'}, False, True), ({'w'}, True, False)]:
            with self.subTest(keys=keys, focused=focused, fresh=fresh):
                drone = Mock()
                control = FlightControl(drone)
                control.takeoff()
                control.update({'w'})
                control.update(keys, focused, fresh)
                drone.send_rc_control.assert_called_with(0, 0, 0, 0)

    def test_lost_takeoff_acknowledgement_still_attempts_landing(self):
        drone = Mock()
        drone.takeoff.side_effect = RuntimeError('ack lost')
        control = FlightControl(drone)
        with self.assertRaises(RuntimeError):
            control.takeoff()
        self.assertTrue(control.possibly_airborne)
        control.land()
        drone.land.assert_called_once()
        self.assertFalse(control.possibly_airborne)

    def test_land_attempted_even_if_zero_rc_fails(self):
        drone = Mock()
        control = FlightControl(drone)
        control.takeoff()
        drone.send_rc_control.side_effect = RuntimeError('send failure')
        with self.assertRaises(RuntimeError):
            control.land()
        drone.land.assert_called_once()

    def test_exit_and_stream_error_land_before_disconnecting(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                args = parse_args(['--fly', '--output', directory])
                drone = Mock()
                drone.get_battery.return_value = 80
                display = Mock()
                def stream(args, reader, session, flight, display):
                    flight.takeoff()
                    flight.update({'w'})
                    if fail:
                        raise TimeoutError('offline test')
                    return 'user_quit'
                with patch('djitellopy.Tello', return_value=drone), \
                        patch('compvis.tello_capture_flight.FlightDisplay', return_value=display), \
                        patch('compvis.capture_tello_dataset.capture_stream', side_effect=stream), \
                        patch('builtins.print'):
                    if fail:
                        with self.assertRaises(TimeoutError):
                            run(args)
                    else:
                        run(args)
                calls = [entry[0] for entry in drone.mock_calls]
                self.assertLess(calls.index('land'), calls.index('end'))
                drone.land.assert_called_once()
                display.close.assert_called_once()
                metadata = json.loads(next(Path(directory).glob('*/session.json')).read_text())
                self.assertEqual(metadata['rc_input_magnitude'], 100)
                self.assertEqual(metadata['status'], 'error' if fail else 'finished')

    def test_flight_requires_preview_and_valid_speed(self):
        for options in [['--fly', '--auto', '--no-display'], ['--speed', '101'], ['--speed', '0']]:
            with self.subTest(options=options), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args(options)


if __name__ == '__main__':
    unittest.main()
