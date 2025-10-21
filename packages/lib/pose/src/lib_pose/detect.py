"""姿勢検出用の関数群"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as RunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)

from .data import PoseData


def _landmarks_to_keypoints(
    landmarks: "PoseLandmarkerResult", image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MediaPipe の landmarks を (N,3) の配列と visibility に変換する内部ヘルパー。
    画像座標, ワールド座標, 可視性スコアを返す。

    image_size: (width, height)
    """
    width, height = image_size
    lm_list = []
    lm_world_list = []
    v_list = []
    for lm in landmarks.pose_landmarks[0]:
        x = float(lm.x) * width
        y = float(lm.y) * height
        z = float(lm.z) * width  # MediaPipe は z を正規化 x スケールで返す
        v = lm.visibility
        lm_list.append((x, y, z))
        v_list.append(float(v))
    for lm in landmarks.pose_world_landmarks[0]:
        x = float(lm.x)
        y = float(lm.y)
        z = float(lm.z)
        lm_world_list.append((x, y, z))

    return (
        np.array(lm_list, dtype=float),
        np.array(lm_world_list, dtype=float),
        np.array(v_list, dtype=float),
    )


def load_reference_pose(image_path: str) -> PoseData:
    """参照画像ファイルから MediaPipe を用いて姿勢を推定する。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"参照画像を開けませんでした: {image_path}")

    # PoseEstimator を使って推定
    with PoseEstimator() as estimator:
        pose_data = estimator.process_frame(img)
    return pose_data


def extract_pose_from_frame(frame: np.ndarray) -> Optional[PoseData]:
    """単一のフレームから姿勢を推定して PoseData を返す。

    引数:
            frame: BGR フォーマットの画像（numpy.ndarray）

    戻り値:
            検出成功時は PoseData、検出できなければ None を返す。
    """
    if frame is None:
        return None

    with PoseEstimator() as estimator:
        pose_data = estimator.process_frame(frame)
    # 検出できなければ keypoints.shape[0] == 0 になる
    if pose_data.keypoints.shape[0] > 0:
        return pose_data
    return None


class PoseEstimator:
    """長寿命の MediaPipe Pose ラッパー。

    with 文で使用でき、`process_frame(frame)` を呼んで各フレームごとに
    `PoseData` を返します。デモはこれを使えば mediapipe を直接 import する必要がなくなります。
    """

    def __init__(
        self,
        model_asset_path: str = "pose_landmarker_full.task",
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        base_options = BaseOptions(model_asset_path=model_asset_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=enable_segmentation,
        )
        self._detector = PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> PoseData:
        """BGR フレームを入力に取り、PoseData を返す（検出無ければ空配列を返す）。"""
        if frame is None:
            return PoseData(keypoints=np.zeros((0, 3)), image_size=(0, 0))

        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
        results = self._detector.detect(mp_image)
        image_rgb.flags.writeable = True
        keypoints, keypoints_world, visibility = _landmarks_to_keypoints(
            results, (width, height)
        )

        return PoseData(
            keypoints=keypoints,
            keypoints_world=keypoints_world,
            visibility=visibility,
            image_size=(width, height),
        )

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
