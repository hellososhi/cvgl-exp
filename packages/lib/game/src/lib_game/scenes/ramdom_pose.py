from __future__ import annotations

import math
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
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ..sequence import SceneInterface, SequenceManager


def _prepare_pose_for_display(pose: PoseData) -> PoseData:
    """Flip axes for matplotlib display without mutating source data."""

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


class RandomPoseScene(SceneInterface):
    """Scene that renders a random 3D pose in a matplotlib window."""

    def __init__(
        self,
        manager: Optional[SequenceManager] = None,
        *,
        rotation_speed_deg: float = 45.0,
    ) -> None:
        super().__init__(manager)
        self._rotation_speed = rotation_speed_deg
        self._target_rotation = 720.0  # two full rotations
        self._current_rotation = 0.0
        self._figure: Optional[Figure] = None
        self._axes: Optional[Axes3D] = None
        self._pose_visuals: Optional[PoseVisualsMatplotlib] = None
        self._transitioned = False
        self._base_elev = 20.0
        self._base_azim = -60.0

    def enter(self) -> None:
        print("RandomPoseScene: enter")
        self._transitioned = False
        self._current_rotation = 0.0
        self._open_viewer()

    def exit(self) -> None:
        print("RandomPoseScene: exit")
        self._close_viewer()

    def update(self, dt: float) -> None:
        if self._figure is None or self._axes is None:
            return

        if not plt.fignum_exists(self._figure.number):
            # User closed the window manually.
            self._transition_to_next()
            return

        self._current_rotation += self._rotation_speed * dt
        # keep azimuth bounded for numerical stability
        azim_wrapped = self._base_azim + math.fmod(self._current_rotation, 360.0)
        self._axes.view_init(elev=self._base_elev, azim=azim_wrapped)
        try:
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            # Ignore rendering issues; fallback to transitioning away.
            self._transition_to_next()
            return

        if self._current_rotation >= self._target_rotation:
            self._transition_to_next()

    def render(self, surface) -> None:
        # Rendering handled by matplotlib window; pygame surface remains untouched.
        return None

    def handle_event(self, event) -> None:
        # No additional event handling required for this scene.
        return None

    def _open_viewer(self) -> None:
        self._close_viewer()

        plt.ion()
        self._figure = plt.figure(figsize=(8, 6))
        figure = self._figure
        if figure is None:
            return

        axes = figure.add_subplot(111, projection="3d")
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_xlim(-1.5, 1.5)
        axes.set_ylim(-1.5, 1.5)
        axes.set_zlim(-1.5, 1.5)
        axes.view_init(elev=self._base_elev, azim=self._base_azim)
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        axes.zaxis.set_ticklabels([])
        self._axes = axes

        pose = generate_random_pose()
        if self.manager is not None:
            self.manager.global_state.pose = pose
        pose_for_display = _prepare_pose_for_display(pose)
        self._pose_visuals = create_pose_3d_matplotlib(
            pose_for_display,
            ax=axes,
            show_labels=False,
            visibility_threshold=0.1,
            normalize=False,
        )

        # Draw once to ensure the window is ready before the game loop continues.
        canvas = figure.canvas
        canvas.draw_idle()
        canvas.flush_events()
        plt.show(block=False)
        canvas.mpl_connect("close_event", self._on_close_event)

    def _on_close_event(self, _event) -> None:
        self._transition_to_next()

    def _close_viewer(self) -> None:
        if self._pose_visuals is not None:
            dispose_pose_visuals_matplotlib(self._pose_visuals)
            self._pose_visuals = None
        if self._figure is not None:
            try:
                plt.close(self._figure)
            except Exception:
                pass
        self._figure = None
        self._axes = None

    def _transition_to_next(self) -> None:
        if self._transitioned:
            return
        self._transitioned = True
        self._close_viewer()
        if self.manager is not None:
            self.manager.start("game")
