"""Display randomly generated poses with a 3D matplotlib viewer."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
from lib_pose.data import PoseData
from lib_pose.demo.pose_3d_matplotlib import (
    PoseVisualsMatplotlib,
    create_pose_3d_matplotlib,
    dispose_pose_visuals_matplotlib,
)
from lib_pose.generate_kinematics import generate_random_pose
from matplotlib import pyplot as plt
from matplotlib.artist import Artist

NORMALIZE = False
TRANSLATE = (0.0, 0.0, 0.0)
SCALE = 1.0


def _prepare_pose_for_display(pose: PoseData) -> PoseData:
    if pose.keypoints.size == 0:
        return pose

    keypoints = np.array(pose.keypoints, copy=True)
    if keypoints.shape[1] >= 2:
        keypoints[:, 1] = pose.image_size[1] - keypoints[:, 1]

    keypoints_world = np.array(pose.keypoints_world, copy=True)
    if keypoints_world.shape[1] >= 2:
        keypoints_world[:, 1] = -keypoints_world[:, 1]

    visibility = np.array(pose.visibility, copy=True)
    return PoseData(
        keypoints=keypoints,
        keypoints_world=keypoints_world,
        visibility=visibility,
        image_size=pose.image_size,
    )


class RandomPoseViewer:
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        seed: Optional[int],
        visibility_threshold: float,
    ) -> None:
        self._image_size = image_size
        self._rng = np.random.default_rng(seed)
        self._visibility_threshold = visibility_threshold

        self.fig = plt.figure(figsize=(9, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect((1.0, 1.0, 1.0))
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(-2.0, 2.0)
        self.ax.set_zlim(-2.5, 1.5)
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_xlabel("x (horizontal)")
        self.ax.set_ylabel("z (depth)")
        self.ax.set_zlabel("y (vertical)")
        self.fig.suptitle("Random pose viewer — press 'n' for next, 'q' to quit")

        self.pose_visuals: Optional[PoseVisualsMatplotlib] = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._render_next_pose()

    def _render_next_pose(self) -> None:
        try:
            pose = generate_random_pose(
                rng=self._rng,
            )
        except RuntimeError as exc:
            print(f"Pose generation failed: {exc}")
            return

        display_pose = _prepare_pose_for_display(pose)
        if self.pose_visuals is not None:
            dispose_pose_visuals_matplotlib(self.pose_visuals)
            self.pose_visuals = None

        self.pose_visuals = create_pose_3d_matplotlib(
            display_pose,
            ax=self.ax,
            visibility_threshold=self._visibility_threshold,
            normalize=NORMALIZE,
            translate=TRANSLATE,
            scale=SCALE,
        )

        artists: list[Artist] = []
        if self.pose_visuals.points is not None:
            artists.append(self.pose_visuals.points)
        if self.pose_visuals.segments is not None:
            artists.extend(self.pose_visuals.segments)
        if self.pose_visuals.labels is not None:
            artists.extend(self.pose_visuals.labels)
        for artist in artists:
            artist.set_animated(False)

        self.fig.canvas.draw_idle()

    def _on_key_press(self, event) -> None:
        if not event.key:
            return
        key = event.key.lower()
        if key == "n":
            self._render_next_pose()
        elif key in {"q", "escape"}:
            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        self.close()

    def close(self) -> None:
        if self.pose_visuals is not None:
            dispose_pose_visuals_matplotlib(self.pose_visuals)
            self.pose_visuals = None

    def run(self) -> None:
        try:
            plt.show()
        finally:
            self.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width passed to the pose generator (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height passed to the pose generator (default: 480).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible random poses (default: random).",
    )
    parser.add_argument(
        "--visibility",
        type=float,
        default=0.0,
        help="Visibility threshold for rendering landmarks (default: 0.0).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    viewer = RandomPoseViewer(
        image_size=(args.width, args.height),
        seed=args.seed,
        visibility_threshold=args.visibility,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
