"""姿勢検出用の関数群"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from .data import PoseData


def _landmarks_to_keypoints(landmarks, image_size: Tuple[int, int]) -> np.ndarray:
    """MediaPipe の landmarks を (N,4) の配列に変換する内部ヘルパー。

    image_size: (width, height)
    """
    width, height = image_size
    lm_list = []
    for lm in landmarks.landmark:
        x = float(lm.x) * width
        y = float(lm.y) * height
        z = float(lm.z) * width  # MediaPipe は z を正規化 x スケールで返す
        v = getattr(lm, "visibility", 1.0)
        lm_list.append((x, y, z, float(v)))

    return np.array(lm_list, dtype=float)


def load_reference_pose(image_path: str) -> Tuple[PoseData, np.ndarray]:
    """参照画像ファイルから MediaPipe を用いて姿勢を推定し、PoseData と注釈画像を返す。

    注: 注釈画像は元画像にランドマークを描画した BGR 画像を返す。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"参照画像を開けませんでした: {image_path}")

    height, width = img.shape[:2]
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    annotated = img.copy()
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)

    landmarks = getattr(results, "pose_landmarks", None)
    if results and landmarks:
        keypoints = _landmarks_to_keypoints(landmarks, (width, height))
        mp_drawing.draw_landmarks(
            annotated,
            landmarks,
            list(mp_pose.POSE_CONNECTIONS),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )
    else:
        keypoints = np.zeros((0, 3), dtype=float)

    return PoseData(keypoints=keypoints, image_size=(width, height)), annotated


def extract_pose_from_frame(frame: np.ndarray) -> Optional[PoseData]:
    """単一のフレームから姿勢を推定して PoseData を返す。

    引数:
            frame: BGR フォーマットの画像（numpy.ndarray）

    戻り値:
            検出成功時は PoseData、検出できなければ None を返す。
    """
    if frame is None:
        return None

    height, width = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)

    landmarks = getattr(results, "pose_landmarks", None)
    if results and landmarks:
        keypoints = _landmarks_to_keypoints(landmarks, (width, height))
        return PoseData(keypoints=keypoints, image_size=(width, height))

    return None


class PoseEstimator:
    """長寿命の MediaPipe Pose ラッパー。

    with 文で使用でき、`process_frame(frame)` を呼んで各フレームごとに
    `PoseData` を返します。デモはこれを使えば mediapipe を直接 import する必要がなくなります。
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # 遅延 import は不要だが、明示的に依存をここで参照する
        from mediapipe.python.solutions import pose as mp_pose

        self._mp_pose = mp_pose
        self._pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> PoseData:
        """BGR フレームを入力に取り、PoseData を返す（検出無ければ空配列を返す）。"""
        if frame is None:
            return PoseData(keypoints=np.zeros((0, 3)), image_size=(0, 0))

        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._pose.process(image_rgb)
        image_rgb.flags.writeable = True

        landmarks = getattr(results, "pose_landmarks", None)
        if results and landmarks:
            keypoints = _landmarks_to_keypoints(landmarks, (width, height))
        else:
            keypoints = np.zeros((0, 3), dtype=float)

        return PoseData(keypoints=keypoints, image_size=(width, height))

    def close(self) -> None:
        try:
            # MediaPipe の Pose は close() を提供
            self._pose.close()
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
