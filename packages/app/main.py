import os
from typing import Optional

import pygame
from lib_game import SceneInterface
from lib_game.sequence import SequenceManager

"""App entrypoint that demonstrates the lib_game sequence interface.

This module intentionally only demonstrates how the sequence API is
called. Scene implementations are minimal placeholders. After your
review I'll implement full behaviour (drawing, pose generation, camera,
etc.).
"""


class StartScene(SceneInterface):
    def __init__(self, manager=None):
        super().__init__(manager)
        self.image = None

    def enter(self):
        """Load the start image if pygame is available.

        The image file is expected at `packages/app/images/start.png`.
        """

        # compute image path relative to this file; defer actual load until
        # we have a valid surface/display (to avoid convert() errors)

        base = os.path.dirname(__file__)
        self._image_path = os.path.join(base, "images", "start.png")
        self.image = None

        print("StartScene: enter")

    def exit(self):
        # release image reference
        self.image = None
        print("StartScene: exit")

    def update(self, dt: float) -> None:
        # No logic for start screen yet
        return None

    def render(self, surface: Optional[pygame.Surface]) -> None:
        """Render the start image centered on the provided surface.

        If `surface` is None (fallback mode) or the image failed to load,
        this method will print a short message instead of raising.
        """
        # If we don't have a surface or pygame image, run in fallback mode
        if surface is None:
            print("StartScene: render (no surface)")
            return

        # If pygame is available but we haven't loaded/converted the image yet,
        # try to load now (display should be initialized by the caller).
        if self.image is None and getattr(self, "_image_path", None):
            self.image = pygame.image.load(self._image_path)
            try:
                self.image = self.image.convert_alpha()
            except Exception:
                self.image = self.image.convert()

        if self.image is None:
            # nothing to blit
            print("StartScene: render (no image)")
            return

        try:
            rect = self.image.get_rect()
            surf_w, surf_h = surface.get_size()
            rect.center = (surf_w // 2, surf_h // 2)
            surface.blit(self.image, rect)
        except Exception as e:
            print("StartScene: render failed:", e)

    def handle_event(self, event) -> None:
        # placeholder: no event handling yet
        return None


class GameScene(SceneInterface):
    def enter(self):
        print("GameScene: enter")

    def exit(self):
        print("GameScene: exit")

    def update(self, dt: float) -> None:
        pass

    def render(self, surface):
        print("GameScene: render")

    def handle_event(self, event) -> None:
        pass


class ResultScene(SceneInterface):
    def enter(self):
        print("ResultScene: enter")

    def exit(self):
        print("ResultScene: exit")

    def update(self, dt: float) -> None:
        pass

    def render(self, surface):
        print("ResultScene: render")

    def handle_event(self, event) -> None:
        pass


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
