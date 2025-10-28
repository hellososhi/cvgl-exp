from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame
from lib_pose import compute_similarity
from lib_pose.data import PoseData
from lib_pose.demo.pose_3d_matplotlib import (
    PoseVisualsMatplotlib,
    create_pose_3d_matplotlib,
    dispose_pose_visuals_matplotlib,
)
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


@dataclass
class _ResultVisuals:
    reference: Optional[PoseVisualsMatplotlib]
    query: Optional[PoseVisualsMatplotlib]


class ResultScene(SceneInterface):
    def __init__(self, manager: Optional[SequenceManager] = None) -> None:
        super().__init__(manager)
        self._figure: Optional[Figure] = None
        self._axes: Optional[Axes3D] = None
        self._visuals = _ResultVisuals(reference=None, query=None)
        self._similarity: float = 0.0
        self._info_text = None

    def enter(self) -> None:
        print("ResultScene: enter")
        self._open_viewer()

    def exit(self) -> None:
        print("ResultScene: exit")
        self._close_viewer()

    def update(self, dt: float) -> None:
        if self._figure is None:
            return

        if not plt.fignum_exists(self._figure.number):
            self._close_viewer()
            return

        try:
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self._close_viewer()

    def render(self, surface) -> None:
        # Rendering handled by matplotlib window.
        return None

    def handle_event(self, event) -> None:
        if event is None:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._restart_sequence()
            elif event.key == pygame.K_q:
                self._request_quit()

    def _restart_sequence(self) -> None:
        self._close_viewer()
        if self.manager is not None:
            state = self.manager.global_state
            state.query_pose = None
            self.manager.start("random_pose")

    def _request_quit(self) -> None:
        self._close_viewer()
        if self.manager is not None:
            self.manager.running = False

    def _open_viewer(self) -> None:
        self._close_viewer()

        plt.ion()
        self._figure = plt.figure(figsize=(8, 6))
        figure = self._figure
        if figure is None:
            return

        axes = figure.add_subplot(111, projection="3d")
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_xlim(-0.2, 0.2)
        axes.set_ylim(-0.2, 0.2)
        axes.set_zlim(-0.2, 0.2)
        axes.view_init(elev=20.0, azim=-60.0)
        axes.set_title("Result")
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        axes.zaxis.set_ticklabels([])
        self._axes = axes

        state = self.manager.global_state if self.manager is not None else None
        pose_ref = state.pose if state is not None else None
        pose_query = state.query_pose if state is not None else None

        transformed_ref: Optional[PoseData] = None
        self._similarity = 0.0

        if pose_ref is not None and pose_query is not None:
            ref_has_points = pose_ref.keypoints.size > 0
            query_has_points = pose_query.keypoints.size > 0
            if ref_has_points and query_has_points:
                self._similarity, transformed_ref, transformed_query = (
                    compute_similarity(pose_ref, pose_query)
                )

        self._draw_poses(transformed_ref, transformed_query)

        missing: list[str] = []
        if pose_ref is None or pose_ref.keypoints.size == 0:
            missing.append("Reference pose not available.")
        if pose_query is None or pose_query.keypoints.size == 0:
            missing.append("No captured pose available.")

        self._info_text = axes.text2D(
            0.02,
            0.95,
            f"Similarity: {self._similarity:.3f}",
            transform=axes.transAxes,
            fontsize=14,
            color="black",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
        )
        axes.text2D(
            0.02,
            0.88,
            "Press R to retry / Q to quit",
            transform=axes.transAxes,
            fontsize=10,
            color="dimgray",
        )
        if missing:
            axes.text2D(
                0.02,
                0.8,
                "\n".join(missing),
                transform=axes.transAxes,
                fontsize=10,
                color="firebrick",
            )

        canvas = figure.canvas
        canvas.draw_idle()
        canvas.flush_events()
        plt.show(block=False)
        canvas.mpl_connect("close_event", self._on_close_event)

    def _draw_poses(
        self, pose_ref: Optional[PoseData], pose_query: Optional[PoseData]
    ) -> None:
        axes = self._axes
        if axes is None:
            return

        if self._visuals.reference is not None:
            dispose_pose_visuals_matplotlib(self._visuals.reference)
            self._visuals.reference = None
        if self._visuals.query is not None:
            dispose_pose_visuals_matplotlib(self._visuals.query)
            self._visuals.query = None

        if pose_ref is not None:
            display_pose = _prepare_pose_for_display(pose_ref)
            self._visuals.reference = create_pose_3d_matplotlib(
                display_pose,
                ax=axes,
                show_labels=False,
                visibility_threshold=0.1,
                normalize=False,
            )
            if self._visuals.reference.points is not None:
                scatter = self._visuals.reference.points
                set_facecolor = getattr(scatter, "set_facecolor", None)
                if callable(set_facecolor):
                    set_facecolor("tab:blue")
                set_edgecolor = getattr(scatter, "set_edgecolor", None)
                if callable(set_edgecolor):
                    set_edgecolor("tab:blue")
            if self._visuals.reference.segments is not None:
                for line in self._visuals.reference.segments:
                    set_color = getattr(line, "set_color", None)
                    if callable(set_color):
                        set_color("tab:blue")

        if pose_query is not None:
            display_transformed_query = _prepare_pose_for_display(pose_query)
            self._visuals.query = create_pose_3d_matplotlib(
                display_transformed_query,
                ax=axes,
                show_labels=False,
                visibility_threshold=0.1,
                normalize=False,
            )
            if self._visuals.query.points is not None:
                scatter = self._visuals.query.points
                set_facecolor = getattr(scatter, "set_facecolor", None)
                if callable(set_facecolor):
                    set_facecolor("tab:orange")
                set_edgecolor = getattr(scatter, "set_edgecolor", None)
                if callable(set_edgecolor):
                    set_edgecolor("tab:orange")
            if self._visuals.query.segments is not None:
                for line in self._visuals.query.segments:
                    set_color = getattr(line, "set_color", None)
                    if callable(set_color):
                        set_color("tab:orange")

    def _on_close_event(self, _event) -> None:
        self._close_viewer()

    def _close_viewer(self) -> None:
        if self._visuals.reference is not None:
            dispose_pose_visuals_matplotlib(self._visuals.reference)
            self._visuals.reference = None
        if self._visuals.query is not None:
            dispose_pose_visuals_matplotlib(self._visuals.query)
            self._visuals.query = None
        if self._figure is not None:
            try:
                plt.close(self._figure)
            except Exception:
                pass
        self._figure = None
        self._axes = None
        if self._info_text is not None:
            try:
                self._info_text.remove()
            except Exception:
                pass
        self._info_text = None
        self._visuals = _ResultVisuals(reference=None, query=None)
