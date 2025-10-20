"""
参照画像とカメラ映像から MediaPipe Pose を使って姿勢を推定し、
類似度を計算してカメラ映像上に表示するデモ。

このファイルではまず関数のインターフェースのみを定義します。
実装はユーザのレビュー後に追加します。
"""

from __future__ import annotations

import sys

import cv2
from lib_pose import (
    PoseEstimator,
    compute_similarity,
    draw_keypoints_on_frame,
    draw_similarity_on_frame,
    load_reference_pose,
)


def run(
    camera_index: int = 0,
    reference_image_path: str = "image.png",
    method: str = "procrustes",
) -> int:
    """メインのランナー関数。ライブラリ API を利用する形に簡潔化。

    この関数は以前の実装と同等の機能を保ちつつ、姿勢検出や描画は
    `lib.pose` の関数を利用する。
    """
    try:
        ref_pose, ref_annotated = load_reference_pose(reference_image_path)
    except Exception as e:
        print(f"参照画像の読み込みに失敗しました: {e}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"カメラを開けませんでした (index={camera_index})", file=sys.stderr)
        return 3

    with PoseEstimator(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as estimator:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームを取得できませんでした", file=sys.stderr)
                    break

                query_pose = estimator.process_frame(frame)

                draw_keypoints_on_frame(frame, query_pose)
                # 参照画像注釈は既に注釈済み画像を表示
                draw_keypoints_on_frame(ref_annotated, ref_pose)

                sim = compute_similarity(ref_pose, query_pose, method=method)
                print(f"Similarity: {sim:.3f}")
                flipped = cv2.flip(frame, 1)
                draw_similarity_on_frame(flipped, sim)
                cv2.imshow("MediaPipe Pose Similarity (mirrored)", flipped)
                ref_flipped = cv2.flip(ref_annotated, 1)
                cv2.imshow("Reference Pose", ref_flipped)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    idx = 0
    ref = "image.png"
    method = "procrustes"

    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            print("カメラ index は整数で指定してください。", file=sys.stderr)
            sys.exit(1)

    if len(sys.argv) > 2:
        ref = sys.argv[2]

    if len(sys.argv) > 3:
        method = sys.argv[3]

    sys.exit(run(idx, ref, method))
