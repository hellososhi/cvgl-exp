import pygame
from lib_game import GameScene, ResultScene, SequenceManager, StartScene

"""App entrypoint that demonstrates the lib_game sequence interface.

This module intentionally only demonstrates how the sequence API is
called. Scene implementations are minimal placeholders. After your
review I'll implement full behaviour (drawing, pose generation, camera,
etc.).
"""


def main():
    manager = SequenceManager()
    manager.initialize()

    # register minimal placeholder scenes
    manager.register_scene("start", StartScene())
    manager.register_scene("game", GameScene())
    manager.register_scene("result", ResultScene())

    # start with the start scene
    manager.start("start")

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True
    while running and manager.running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            manager.handle_event(event)

        manager.update(dt)
        manager.render(screen)
        pygame.display.flip()

    pygame.quit()

    manager.shutdown()


if __name__ == "__main__":
    main()
