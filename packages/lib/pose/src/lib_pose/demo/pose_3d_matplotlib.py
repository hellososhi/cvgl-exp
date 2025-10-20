"""カメラの画像から姿勢推定を行い、matplotlib で 3D 可視化するデモ。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from lib_pose.data import POSE_CONNECTIONS, PoseData
from lib_pose.detect import PoseEstimator
from lib_pose.util_2d import draw_keypoints_on_frame
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)


@dataclass
class PoseVisualsMatplotlib:
    """Container for matplotlib artists representing a pose."""

    points: Optional[Artist]
    segments: Optional[List[Artist]]


def _prepare_coordinates(
    pose: PoseData,
    *,
    normalize: bool,
    translate: Tuple[float, float, float],
    scale: float,
) -> np.ndarray:
    coords = np.asarray(pose.keypoints[:, :3], dtype=np.float32)

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
    """Create matplotlib artists for 3D pose rendering.

    Returns a PoseVisualsMatplotlib containing scatter and line collections (as lists).
    """
    if pose.keypoints.size == 0:
        return PoseVisualsMatplotlib(points=None, segments=None)

    coords = _prepare_coordinates(
        pose, normalize=normalize, translate=translate, scale=scale
    )
    visibility = pose.keypoints[:, 3]
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


class Pose3DMatplotlibDemo:
    """Real-time pose estimator that renders a 3D skeleton with matplotlib."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        update_rate: float = 30.0,
        visibility_threshold: float = 0.5,
        debug: bool = False,
    ) -> None:
        if update_rate <= 0:
            raise ValueError("update_rate must be positive")

        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index}")

        self._cv_window_name = "Pose 2D View"
        self._cv_window_created = False
        try:
            cv2.namedWindow(self._cv_window_name, cv2.WINDOW_NORMAL)
            self._cv_window_created = True
        except cv2.error:
            self._cv_window_created = False

        self._estimator = PoseEstimator(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # matplotlib figure and 3D axes
        # We map MediaPipe coordinates to matplotlib as:
        #   MediaPipe: x=horizontal, y=vertical, z=depth
        #   matplotlib axes: X=x (horizontal), Y=z (depth), Z=y (vertical)
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect((1, 1, 1))
        # set limits for (X, Y, Z) where Y is depth
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)  # depth (z)
        self.ax.set_zlim(-1.5, 1.5)  # vertical (y)
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_xlabel("x (horizontal)")
        self.ax.set_ylabel("z (depth)")
        self.ax.set_zlabel("y (vertical)")

        self.pose_visuals: Optional[PoseVisualsMatplotlib] = None
        self.visibility_threshold = visibility_threshold
        self._update_rate = update_rate
        self._debug = debug

        # For animation
        self.ani: Optional[FuncAnimation] = None

    def _prepare_pose_for_render(self, pose: PoseData) -> PoseData:
        if pose.keypoints.size == 0:
            return pose

        keypoints = np.array(pose.keypoints, copy=True)
        # invert y to match display coordinate (origin at bottom-left for matplotlib 3D)
        if keypoints.shape[1] >= 2:
            keypoints[:, 1] = pose.image_size[1] - keypoints[:, 1]

        return PoseData(keypoints=keypoints, image_size=pose.image_size)

    def _update(self, frame_index: int) -> list[Artist]:
        if not self._capture.isOpened():
            plt.close(self.fig)
            return []

        ok, frame = self._capture.read()
        if not ok:
            return []

        pose = self._estimator.process_frame(frame)
        if self._cv_window_created:
            display_frame = frame.copy()
            draw_keypoints_on_frame(display_frame, pose)
            try:
                cv2.imshow(self._cv_window_name, display_frame)
            except cv2.error:
                self._cv_window_created = False

        render_pose = self._prepare_pose_for_render(pose)
        if self._debug and render_pose.keypoints.size:
            print(f"Max visibility: {np.max(render_pose.keypoints[:, 3])}")

        # Clear prior visuals
        if self.pose_visuals:
            dispose_pose_visuals_matplotlib(self.pose_visuals)
            self.pose_visuals = None

        if render_pose.keypoints.size == 0:
            # nothing to draw
            return []

        self.pose_visuals = create_pose_3d_matplotlib(
            render_pose,
            ax=self.ax,
            visibility_threshold=self.visibility_threshold,
            normalize=True,
            translate=(0.0, -0.2, 0.0),
            scale=2.0,
        )

        # redraw canvas
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

        # FuncAnimation expects the update function to return an iterable of
        # artists that have been modified. Return current artists for blitting
        artists: list[Artist] = []
        if self.pose_visuals:
            if self.pose_visuals.points is not None:
                artists.append(self.pose_visuals.points)
            if self.pose_visuals.segments is not None:
                artists.extend(self.pose_visuals.segments)
        return artists

    def run(self) -> None:
        # Use matplotlib animation to periodically call _update
        interval_ms = int(1000.0 / self._update_rate)
        self.ani = FuncAnimation(self.fig, self._update, interval=interval_ms)
        try:
            plt.show()
        finally:
            self.close()

    def close(self) -> None:
        try:
            if self.pose_visuals:
                dispose_pose_visuals_matplotlib(self.pose_visuals)
                self.pose_visuals = None
        except Exception:
            pass

        if self._capture.isOpened():
            self._capture.release()

        if self._cv_window_created:
            try:
                cv2.destroyWindow(self._cv_window_name)
            except cv2.error:
                pass
            self._cv_window_created = False

        try:
            self._estimator.close()
        except Exception:
            pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index passed to OpenCV's VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--update-rate",
        type=float,
        default=30.0,
        help="Scheduler frequency in Hz for capturing frames (default: 30).",
    )
    parser.add_argument(
        "--visibility",
        type=float,
        default=0.5,
        help="Visibility threshold to treat landmarks as visible (default: 0.5).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    demo = Pose3DMatplotlibDemo(
        camera_index=args.camera,
        update_rate=args.update_rate,
        visibility_threshold=args.visibility,
        debug=args.debug,
    )
    demo.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
