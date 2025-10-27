"""姿勢同士の類似度計算"""

from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes

from .data import PoseData


def compute_similarity(
    pose_ref: PoseData, pose_query: PoseData, method: str = "procrustes"
) -> tuple[float, PoseData | None, PoseData | None]:
    """参照姿勢とクエリ姿勢の類似度を計算する。

    現状は Procrustes のみ実装。
    戻り値は 0.0-1.0 の範囲（1.0 が完全一致）
    """
    if pose_ref is None or pose_query is None:
        return 0.0, None, None

    pts1 = pose_ref.keypoints
    pts2 = pose_query.keypoints

    if pts1 is None or pts2 is None:
        return 0.0, None, None

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return 0.0, None, None

    k = min(pts1.shape[0], pts2.shape[0])
    points1 = pts1[:k, :3]
    points2 = pts2[:k, :3]

    if points1.shape[0] < 2:
        return 0.0, None, None

    try:
        transformed_ref, transformed_query, disparity = procrustes(points1, points2)
    except Exception:
        return 0.0, None, None

    sim = 1.0 - float(disparity)
    sim = max(0.0, min(1.0, sim))

    transformed_ref_keypoints = np.hstack(
        (transformed_ref, np.zeros((transformed_ref.shape[0], 1)))
    )
    transformed_ref_pose = PoseData(
        keypoints=transformed_ref_keypoints,
        keypoints_world=transformed_ref_keypoints,
        image_size=pose_ref.image_size,
        visibility=np.ones((transformed_ref_keypoints.shape[0],)),
    )

    transformed_query_keypoints = np.hstack(
        (transformed_query, np.zeros((transformed_query.shape[0], 1)))
    )
    transformed_query_pose = PoseData(
        keypoints=transformed_query_keypoints,
        keypoints_world=transformed_query_keypoints,
        image_size=pose_query.image_size,
        visibility=np.ones((transformed_query_keypoints.shape[0],)),
    )

    return sim, transformed_ref_pose, transformed_query_pose
