import os
from typing import Optional

import pygame

from ..sequence import SceneInterface


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
        """Handle events on the start screen.

        Pressing the 's' key transitions to the game scene via the
        registered sequence manager.
        """
        if event is None:
            return None

        if event.type == pygame.KEYDOWN:
            # start the game when user presses 's'
            if event.key == pygame.K_s:
                if self.manager is not None:
                    self.manager.start("random_pose")
