"""lib_game package exports for simple sequence/scene management.

This module exposes the minimal interfaces used by the app to manage
game sequences (start screen, game screen, result screen).

Implementations are intentionally minimal / interface-like.
"""

from .scenes.game import GameScene
from .scenes.ramdom_pose import RandomPoseScene
from .scenes.result import ResultScene
from .scenes.start import StartScene
from .sequence import SceneInterface, SequenceManager

__all__ = [
    "SequenceManager",
    "SceneInterface",
    "StartScene",
    "RandomPoseScene",
    "GameScene",
    "ResultScene",
]
