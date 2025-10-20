from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PoseData:
    """姿勢データを表すシンプルなコンテナ。

    attributes:
            keypoints: numpy.ndarray (N, 4) 形式で各キーポイントの (x, y, z, v)
                    x,y は画像座標（ピクセル）を想定。z は深度情報（x でスケーリング）、v は可視性/存在フラグ。
            image_size: Tuple[int, int] 元の画像サイズ (width, height)
    """

    keypoints: np.ndarray
    image_size: Tuple[int, int]


# 姿勢ランドマークの接続 (Mediapipe Pose の定義に基づく)
POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]
)
