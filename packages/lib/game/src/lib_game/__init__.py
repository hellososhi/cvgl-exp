"""lib_game package exports for simple sequence/scene management.

This module exposes the minimal interfaces used by the app to manage
game sequences (start screen, game screen, result screen).

Implementations are intentionally minimal / interface-like.
"""

from .game import GameScene
from .result import ResultScene
from .sequence import SceneInterface, SequenceManager
from .start import StartScene

__all__ = [
    "SequenceManager",
    "SceneInterface",
    "GameScene",
    "StartScene",
    "ResultScene",
]
