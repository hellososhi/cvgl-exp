"""姿勢に関するデータの定義"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PoseData:
    """姿勢データを表すシンプルなコンテナ。

    attributes:
            keypoints: numpy.ndarray (N, 3) 形式で各キーポイントの (x, y, z)
                    x,y は画像座標（ピクセル）を想定。z は深度情報（x でスケーリング）  。
            keypoints_world: numpy.ndarray (N, 3) 形式で各キーポイントのワールド座標 (x, y, z)
            visibility: numpy.ndarray (N,) 形式で各キーポイントの可視性スコア
            image_size: Tuple[int, int] 元の画像サイズ (width, height)
    """

    keypoints: np.ndarray
    keypoints_world: np.ndarray
    visibility: np.ndarray
    image_size: Tuple[int, int]

    def __str__(self) -> str:
        np.set_printoptions(precision=3, suppress=True)
        return (
            f"PoseData(keypoints={self.keypoints}, "
            f"keypoints_world={self.keypoints_world}, "
            f"visibility={self.visibility}, "
            f"image_size={self.image_size})"
        )


# 姿勢ランドマークのラベル (Mediapipe Pose の定義に基づく)
POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
    # added
    "neck_top": 33,
    "neck_bottom": 34,
}

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
        # added
        (11, 34),
        (12, 34),
        (33, 34),
    ]
)

# サンプル画像から得られた骨（ランドマーク間）の距離（L2 ノルム）
POSE_SAMPLE_BONE_LENGTHS = {
    (0, 1): 0.1265,
    (1, 2): 0.0028,
    (2, 3): 0.0012,
    (3, 7): 0.3613,
    (0, 4): 0.1338,
    (4, 5): 0.0038,
    (5, 6): 0.0048,
    (6, 8): 0.3526,
    (9, 10): 0.1303,
    (11, 12): 1.0000,
    (11, 13): 0.7491,
    (13, 15): 0.7449,
    (15, 17): 0.2821,
    (15, 19): 0.3411,
    (15, 21): 0.0792,
    (17, 19): 0.1283,
    (12, 14): 0.7943,
    (14, 16): 0.7694,
    (16, 18): 0.2738,
    (16, 20): 0.3082,
    (16, 22): 0.0862,
    (18, 20): 0.1197,
    (11, 23): 1.5017,
    (12, 24): 1.4749,
    (23, 24): 0.6750,
    (23, 25): 1.2454,
    (24, 26): 1.2209,
    (25, 27): 1.1575,
    (26, 28): 1.1402,
    (27, 29): 0.1341,
    (28, 30): 0.1346,
    (29, 31): 0.4057,
    (30, 32): 0.4020,
    (27, 31): 0.4719,
    (28, 32): 0.4640,
    (33, 34): 0.5851,  # added
}
