"""
画像ファイルから姿勢を推定するデモ。
"""

from __future__ import annotations

import sys

import cv2
from lib_pose import PoseEstimator, draw_keypoints_on_frame


def run(
    reference_image_path: str = "image.png",
) -> int:
    try:
        ref = cv2.imread(reference_image_path)
        if ref is None:
            raise FileNotFoundError(
                f"参照画像を開けませんでした: {reference_image_path}"
            )
        ref_pose = PoseEstimator().process_frame(ref)
        print(ref_pose)
        # ref_pose.keypoints 外部ファイルに書き出す
        with open("ref_pose_keypoints.txt", "w") as f:
            for keypoint in ref_pose.keypoints_world:
                f.write(f"{keypoint[0]} {keypoint[1]} {keypoint[2]}\n")
    except Exception as e:
        print(f"参照画像の読み込みに失敗しました: {e}", file=sys.stderr)
        return 2

    try:
        while True:
            draw_keypoints_on_frame(ref, ref_pose)
            cv2.imshow("Reference Pose", ref)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    ref = "image.png"

    if len(sys.argv) > 1:
        ref = sys.argv[1]

    sys.exit(run(ref))
