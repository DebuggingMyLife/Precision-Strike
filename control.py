import pygame
from djitellopy import Tello

CONTROL = 20  # Gentle initial RC input; each channel accepts -100 to 100.

pygame.init()
window = pygame.display.set_mode((640, 180))
pygame.display.set_caption("Tello Keyboard Control")
font = pygame.font.Font(None, 26)
clock = pygame.time.Clock()

drone = Tello()
possibly_airborne = False

try:
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                elif event.key == pygame.K_t and not possibly_airborne:
                    # Set before takeoff in case its acknowledgement is lost.
                    possibly_airborne = True
                    drone.takeoff()

                elif event.key == pygame.K_l and possibly_airborne:
                    drone.send_rc_control(0, 0, 0, 0)
                    drone.land()
                    possibly_airborne = False

        if not running:
            break

        left_right = forward_back = up_down = yaw = 0

        if pygame.key.get_focused():
            keys = pygame.key.get_pressed()

            if not keys[pygame.K_SPACE]:
                left_right = CONTROL * (
                    int(keys[pygame.K_d]) - int(keys[pygame.K_a])
                )
                forward_back = CONTROL * (
                    int(keys[pygame.K_w]) - int(keys[pygame.K_s])
                )
                up_down = CONTROL * (
                    int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
                )
                yaw = CONTROL * (
                    int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
                )

        if possibly_airborne:
            drone.send_rc_control(
                left_right, forward_back, up_down, yaw
            )

        window.fill((25, 25, 30))
        lines = [
            "T: take off | L: land | Esc: land and exit",
            "W/S: forward/back | A/D: left/right",
            "Up/Down: altitude | Left/Right: rotate",
            "Hold keys to move | Space: hover",
        ]
        for index, line in enumerate(lines):
            text = font.render(line, True, (240, 240, 240))
            window.blit(text, (15, 15 + index * 35))

        pygame.display.flip()
        clock.tick(20)  # Approximately 20 control updates per second.

finally:
    try:
        if possibly_airborne:
            drone.send_rc_control(0, 0, 0, 0)
            drone.land()
    finally:
        drone.end()
        pygame.quit()
