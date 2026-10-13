"""Keyboard flight controls and live preview for the dataset capture script."""

from time import perf_counter


class FlightControl:
    def __init__(self, drone, speed=100):
        self.drone = drone
        self.speed = speed
        self.possibly_airborne = False
        self.last_command = None
        self.last_sent = float('-inf')

    def takeoff(self):
        if not self.possibly_airborne:
            # A missing acknowledgement does not prove that takeoff failed.
            self.possibly_airborne = True
            self.drone.takeoff()

    def land(self):
        if self.possibly_airborne:
            try:
                self.drone.send_rc_control(0, 0, 0, 0)
            finally:
                self.drone.land()
            self.possibly_airborne = False
            self.last_command = None

    def update(self, keys, focused=True, video_fresh=True):
        if not self.possibly_airborne:
            return
        command = (0, 0, 0, 0)
        if focused and video_fresh and 'space' not in keys:
            command = tuple(self.speed * (int(positive in keys) - int(negative in keys))
                            for positive, negative in [('d', 'a'), ('w', 's'),
                                                       ('up', 'down'), ('right', 'left')])
        now = perf_counter()
        # Send changed commands immediately (especially key release/focus loss).
        if command != self.last_command or now - self.last_sent >= 0.05:
            self.drone.send_rc_control(*command)
            self.last_command = command
            self.last_sent = now


class FlightDisplay:
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((960, 840))
        pygame.display.set_caption('Tello flight + dataset capture')
        self.font = pygame.font.Font(None, 26)
        self.clock = pygame.time.Clock()
        self.camera_surface = None

    def poll(self, rgb, automatic, fps, saved, speed, airborne, video_fresh):
        pg = self.pg
        requests = set()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                requests.add('quit')
            elif event.type == pg.KEYDOWN:
                action = {pg.K_q: 'quit', pg.K_ESCAPE: 'quit', pg.K_t: 'takeoff',
                          pg.K_l: 'land', pg.K_c: 'snapshot', pg.K_r: 'auto'}.get(event.key)
                if action:
                    requests.add(action)
        focused = bool(pg.key.get_focused())
        pressed = pg.key.get_pressed()
        keys = {name for name, key in [('w', pg.K_w), ('s', pg.K_s), ('a', pg.K_a),
                                      ('d', pg.K_d), ('up', pg.K_UP), ('down', pg.K_DOWN),
                                      ('left', pg.K_LEFT), ('right', pg.K_RIGHT),
                                      ('space', pg.K_SPACE)] if pressed[key]}
        if rgb is not None:
            # pygame expects RGB with (width, height, channels), not OpenCV's BGR.
            surface = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
            ratio = min(960 / rgb.shape[1], 720 / rgb.shape[0])
            self.camera_surface = pg.transform.scale(surface, (int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio)))
        self.window.fill((20, 20, 25))
        if self.camera_surface is not None:
            self.window.blit(self.camera_surface, self.camera_surface.get_rect(center=(480, 360)))
        mode = f'AUTO {fps:g}/s' if automatic else 'MANUAL'
        status = 'AIRBORNE' if airborne else 'GROUNDED'
        if not focused:
            status += ' | Unfocused: hover'
        if not video_fresh:
            status += ' | Waiting for video: hover'
        lines = [f'{status} | RC input {speed} | {mode} | Saved {saved}',
                 'T: take off | L: land | Q/Esc: land and quit | Space: hover',
                 'W/S: forward/back | A/D: left/right | Up/Down: altitude | Left/Right: yaw',
                 'C: snapshot | R: automatic capture on/off']
        for index, line in enumerate(lines):
            self.window.blit(self.font.render(line, True, (240, 240, 240)), (12, 728 + index * 26))
        pg.display.flip()
        self.clock.tick(20)
        return requests, keys, focused

    def close(self):
        self.pg.quit()
