"""Sequence manager and Scene interface for lib_game.

This file defines minimal, interface-like classes used by the app to
manage game sequences. Methods are intentionally minimal so the review
can focus on the API; implementations will be provided after review.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import pygame


class SceneInterface(abc.ABC):
    """Abstract interface for a scene.

    Implementations should override lifecycle methods. For the first
    iteration we provide only signatures and minimal no-op behaviour.
    """

    def __init__(self, manager: Optional["SequenceManager"] = None) -> None:
        self.manager = manager

    def enter(self) -> None:
        """Called when the scene becomes active."""
        # Interface only: override in concrete scenes
        return None

    def exit(self) -> None:
        """Called when the scene is no longer active."""
        return None

    def update(self, dt: float) -> None:
        """Update scene logic. dt is seconds since last update."""
        return None

    def render(self, surface: Optional[pygame.Surface]) -> None:
        """Render the scene to the given drawing surface (pygame.Surface).

        surface may be None in non-graphical tests.
        """
        return None

    def handle_event(self, event: Optional[pygame.event.Event]) -> None:
        """Handle an input/event object (pygame.Event or similar)."""
        return None


class GlobalState:
    """Class to hold global state for the sequence manager and scenes."""

    def __init__(self):
        pass


class SequenceManager:
    """Simple manager for scenes/sequences.

    Responsibilities:
    - register scenes
    - switch active scene
    - forward update/render/event calls
    This is intentionally lightweight — behaviour will be extended after review.
    """

    def __init__(self) -> None:
        self._scenes: Dict[str, SceneInterface] = {}
        self._current: Optional[SceneInterface] = None
        self.running: bool = False
        self.global_state = GlobalState()

    def initialize(self) -> None:
        """Initialize manager resources. Call before starting the loop."""
        self.running = True

    def register_scene(self, name: str, scene: SceneInterface) -> None:
        """Register a scene instance under a name."""
        scene.manager = self
        self._scenes[name] = scene

    def start(self, name: str) -> None:
        """Switch to the named scene, calling lifecycle hooks."""
        if self._current is not None:
            self._current.exit()

        self._current = self._scenes.get(name)
        if self._current is not None:
            self._current.enter()

    def update(self, dt: float) -> None:
        """Forward update to current scene."""
        if self._current is not None:
            self._current.update(dt)

    def render(self, surface: Any) -> None:
        """Forward render to current scene."""
        if self._current is not None:
            self._current.render(surface)

    def handle_event(self, event: Any) -> None:
        """Forward event to current scene."""
        if self._current is not None:
            self._current.handle_event(event)

    def shutdown(self) -> None:
        """Shutdown manager and active scene."""
        if self._current is not None:
            self._current.exit()
        self.running = False
