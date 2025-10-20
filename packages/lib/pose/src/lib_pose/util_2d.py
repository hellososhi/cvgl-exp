from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .data import POSE_CONNECTIONS, PoseData


def draw_keypoints_on_frame(frame: np.ndarray, pose: PoseData) -> None:
    """フレーム上にキーポイントを描画する補助関数。

    引数:
            frame: BGR 画像（描画はこの配列に行われる）
            pose: PoseData インスタンス

    返り値: なし（frame がインプレースで変更される）
    """
    if pose is None or pose.keypoints is None:
        return

    for k1, k2 in POSE_CONNECTIONS:
        if k1 < pose.keypoints.shape[0] and k2 < pose.keypoints.shape[0]:
            x1, y1, _z1, v1 = pose.keypoints[k1]
            x2, y2, _z2, v2 = pose.keypoints[k2]
            if v1 > 0 and v2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    for i, (x, y, _z, v) in enumerate(pose.keypoints):
        if v == 0 or v <= 0.0:
            color = (150, 150, 150)
        else:
            g = int(max(0.0, min(1.0, v)) * 255)
            r = int((1.0 - max(0.0, min(1.0, v))) * 255)
            b = 0
            color = (b, g, r)
        # デバッグ出力が不要ならコメントアウト
        # print(f"Keypoint {i}: x={x:.3f}, y={y:.3f}, v={v:.3f}, color={color}")
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)


def draw_similarity_on_frame(
    frame: np.ndarray, similarity: float, location: Tuple[int, int] = (10, 40)
) -> None:
    """フレーム上に類似度を描画する補助関数。

    引数:
            frame: BGR 画像（描画はこの配列に行われる）
            similarity: 0.0-1.0 の類似度スコア
            location: 描画開始座標 (x, y)

    返り値: なし（frame がインプレースで変更される）
    """
    x, y = location
    text = f"Similarity: {similarity:.3f}"
    if similarity >= 0.75:
        color = (0, 200, 0)
    elif similarity >= 0.5:
        color = (0, 200, 200)
    else:
        color = (0, 0, 200)

    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5, cv2.LINE_AA
    )
