"""
Webカメラで MediaPipe Pose を使って姿勢推定を行い、OpenCV で左右反転して表示するデモ。

実行:
        uv run python pose.py

ウィンドウで 'q' キーを押すと終了します。
"""

import sys
import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


def main(camera_index: int = 0) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"カメラを開けませんでした (index={camera_index})", file=sys.stderr)
        return 2

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームを取得できませんでした", file=sys.stderr)
                    break

                # MediaPipe は RGB を期待
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)

                # 描画のために書き込み可能に戻す
                image_rgb.flags.writeable = True

                # ランドマークを元の BGR フレームに描画
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    )

                # 表示は左右反転
                flipped = cv2.flip(frame, 1)
                cv2.imshow("MediaPipe Pose (mirrored)", flipped)

                # 'q' キーで終了
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    # 引数にカメラ index を渡せる
    idx = 0
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            print("カメラ index は整数で指定してください。", file=sys.stderr)
            sys.exit(1)

    sys.exit(main(idx))
