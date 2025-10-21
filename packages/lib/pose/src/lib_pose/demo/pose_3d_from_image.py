"""画像ファイルから姿勢推定を行い、matplotlib で 3D 可視化するデモ。"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from lib_pose.data import POSE_CONNECTIONS, PoseData
from lib_pose.detect import PoseEstimator
from lib_pose.util_2d import draw_keypoints_on_frame
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class PoseVisualsMatplotlib:
    points: Optional[Artist]
    segments: Optional[List[Artist]]


def _prepare_coordinates(
    pose: PoseData,
    *,
    normalize: bool,
    translate: Tuple[float, float, float],
    scale: float,
) -> np.ndarray:
    coords = np.asarray(pose.keypoints_world[:, :3], dtype=np.float32)

    if normalize:
        width, height = pose.image_size
        span = max(float(width), float(height)) or 1.0
        coords[:, 0] = (coords[:, 0] - width * 0.5) / span
        coords[:, 1] = (coords[:, 1] - height * 0.5) / span
        coords[:, 2] = coords[:, 2] / span

    coords *= scale
    coords[:, 0] += translate[0]
    coords[:, 1] += translate[1]
    coords[:, 2] += translate[2]
    return coords


def create_pose_3d_matplotlib(
    pose: PoseData,
    *,
    ax,
    visibility_threshold: float = 0.0,
    normalize: bool = True,
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> PoseVisualsMatplotlib:
    if pose.keypoints.size == 0:
        return PoseVisualsMatplotlib(points=None, segments=None)

    coords = _prepare_coordinates(
        pose, normalize=normalize, translate=translate, scale=scale
    )
    visibility = pose.visibility
    visible_mask = visibility >= visibility_threshold

    # Map MediaPipe coords -> plotting coords: (X=x, Y=z, Z=y)
    plot_coords = np.empty_like(coords)
    plot_coords[:, 0] = coords[:, 0]
    plot_coords[:, 1] = coords[:, 2]
    plot_coords[:, 2] = coords[:, 1]

    xs = plot_coords[:, 0]
    ys = plot_coords[:, 1]
    zs = plot_coords[:, 2]

    # Points
    pts_x = xs[visible_mask]
    pts_y = ys[visible_mask]
    pts_z = zs[visible_mask]
    if pts_x.size:
        points = ax.scatter(pts_x, pts_y, pts_z, c="r", s=30)
    else:
        points = None

    # Segments
    segment_lines: List[Artist] = []
    for start, end in POSE_CONNECTIONS:
        if start >= len(coords) or end >= len(coords):
            continue
        if not (visible_mask[start] and visible_mask[end]):
            continue
        xseg = [plot_coords[start, 0], plot_coords[end, 0]]
        yseg = [plot_coords[start, 1], plot_coords[end, 1]]
        zseg = [plot_coords[start, 2], plot_coords[end, 2]]
        line = ax.plot(xseg, yseg, zseg, c="b", linewidth=2)[0]
        segment_lines.append(line)

    return PoseVisualsMatplotlib(points=points, segments=segment_lines)


def dispose_pose_visuals_matplotlib(visuals: PoseVisualsMatplotlib) -> None:
    if visuals is None:
        return
    if visuals.points is not None:
        try:
            visuals.points.remove()
        except Exception:
            pass
    if visuals.segments is not None:
        for line in list(visuals.segments):
            try:
                line.remove()
            except Exception:
                pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_path",
        type=str,
        help="入力画像ファイルのパス",
    )
    parser.add_argument(
        "--visibility",
        type=float,
        default=0.5,
        help="ランドマークを可視とみなす閾値 (default: 0.5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"画像ファイルが読み込めません: {args.image_path}")
        return 1

    estimator = PoseEstimator(
        model_asset_path="pose_landmarker_lite.task",
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = estimator.process_frame(image)

    # 2D キーポイントを画像上に描画して表示 (任意)
    display_frame = image.copy()
    draw_keypoints_on_frame(display_frame, pose)
    cv2.imshow("Pose 2D View", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3D 可視化
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("x (horizontal)")
    ax.set_ylabel("z (depth)")
    ax.set_zlabel("y (vertical)")

    # y座標を反転 (matplotlib 3D座標系に合わせる)
    if pose.keypoints.size:
        keypoints = np.array(pose.keypoints, copy=True)
        if keypoints.shape[1] >= 2:
            keypoints[:, 1] = pose.image_size[1] - keypoints[:, 1]
        keypoints_world = np.array(pose.keypoints_world, copy=True)
        if keypoints_world.shape[1] >= 2:
            keypoints_world[:, 1] = -keypoints_world[:, 1]
        pose = PoseData(
            keypoints=keypoints,
            keypoints_world=keypoints_world,
            visibility=pose.visibility,
            image_size=pose.image_size,
        )
        print(pose)

    visuals = create_pose_3d_matplotlib(
        pose,
        ax=ax,
        visibility_threshold=args.visibility,
        normalize=False,
        # translate=(0.0, -0.2, 0.0),
        # scale=2.0,
    )

    plt.show()
    dispose_pose_visuals_matplotlib(visuals)
    estimator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
