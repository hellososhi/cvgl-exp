"""カメラの画像から姿勢推定を行い、3Dで可視化するデモンストレーション。"""

from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np
import pyglet
from lib_pose import (
    PoseData,
    PoseEstimator,
    PoseVisuals,
    create_pose_3d_batch,
    dispose_pose_visuals,
    draw_keypoints_on_frame,
)
from pyglet import clock, graphics, window
from pyglet.math import Mat4, Vec3
from pyglet.window import key


class Pose3DDemo:
    """Real-time pose estimator that renders a 3D skeleton with pyglet."""

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
            # 2D 表示用のウィンドウを作れない環境では無効化
            self._cv_window_created = False

        self._estimator = PoseEstimator(
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.window = window.Window(
            width=960,
            height=720,
            caption="Pose 3D Demo",
            resizable=True,
        )
        self.window.projection = Mat4.perspective_projection(
            fov=60.0,
            aspect=self.window.width / self.window.height,
            z_near=0.1,
            z_far=100.0,
        )
        self.window.view = Mat4.look_at(
            Vec3(0.0, -0.4, 3.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )

        self.batch = graphics.Batch()
        self.pose_visuals: Optional[PoseVisuals] = None
        self.visibility_threshold = visibility_threshold
        self._update_rate = update_rate
        self._closed = False
        self._debug = debug

        self._register_handlers()
        clock.schedule_interval(self.update, 1.0 / self._update_rate)

    def _register_handlers(self) -> None:
        @self.window.event
        def on_draw() -> None:
            self.window.clear()
            if self.pose_visuals:
                self.pose_visuals.batch.draw()

        @self.window.event
        def on_key_press(symbol: int, modifiers: int) -> None:
            if symbol in (key.ESCAPE, key.Q):
                pyglet.app.exit()

    def update(self, dt: float) -> None:
        if not self._capture.isOpened():
            pyglet.app.exit()
            return

        ok, frame = self._capture.read()
        if not ok:
            return

        pose = self._estimator.process_frame(frame)
        if self._cv_window_created:
            self._render_pose_on_frame(frame, pose)
        render_pose = self._prepare_pose_for_render(pose)
        if self._debug:
            print(f"Max visibility: {np.max(render_pose.keypoints[:, 3])}")

        if render_pose.keypoints.size == 0:
            if self.pose_visuals:
                if self._debug:
                    print("Disposing pose visuals due to no detected pose.")
                dispose_pose_visuals(self.pose_visuals)
                self.pose_visuals = None
            return

        if self.pose_visuals:
            dispose_pose_visuals(self.pose_visuals)

        self.pose_visuals = create_pose_3d_batch(
            render_pose,
            batch=self.batch,
            visibility_threshold=self.visibility_threshold,
            normalize=True,
            translate=(0.0, -0.2, 0.0),
            scale=2.0,
        )

    def _prepare_pose_for_render(self, pose: PoseData) -> PoseData:
        if pose.keypoints.size == 0:
            return pose

        keypoints = np.array(pose.keypoints, copy=True)
        if keypoints.shape[1] >= 2:
            keypoints[:, 1] = pose.image_size[1] - keypoints[:, 1]
        keypoints_world = np.array(pose.keypoints_world, copy=True)
        if keypoints_world.shape[1] >= 2:
            keypoints_world[:, 1] = -keypoints_world[:, 1]

        return PoseData(
            keypoints=keypoints,
            keypoints_world=keypoints_world,
            visibility=pose.visibility,
            image_size=pose.image_size,
        )

    def _render_pose_on_frame(self, frame: np.ndarray, pose: PoseData) -> None:
        display_frame = frame.copy()
        draw_keypoints_on_frame(display_frame, pose)
        try:
            cv2.imshow(self._cv_window_name, display_frame)
        except cv2.error:
            self._cv_window_created = False
            return
        key_code = cv2.waitKey(1) & 0xFF
        if key_code in (27, ord("q"), ord("Q")):
            pyglet.app.exit()

    def close(self) -> None:
        if self._closed:
            return

        clock.unschedule(self.update)
        if self.pose_visuals:
            dispose_pose_visuals(self.pose_visuals)
            self.pose_visuals = None

        if self._capture.isOpened():
            self._capture.release()

        if self._cv_window_created:
            try:
                cv2.destroyWindow(self._cv_window_name)
            except cv2.error:
                pass
            self._cv_window_created = False

        self._estimator.close()
        self._closed = True

    def run(self) -> None:
        try:
            pyglet.app.run()
        finally:
            self.close()


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

    demo = Pose3DDemo(
        camera_index=args.camera,
        update_rate=args.update_rate,
        visibility_threshold=args.visibility,
        debug=args.debug,
    )
    demo.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
