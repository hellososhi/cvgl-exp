"""Utilities for rendering 3D poses with pyglet."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pyglet

from .data import POSE_CONNECTIONS, PoseData


@dataclass
class PoseVisuals:
    """Container returned by :func:`create_pose_3d_batch`.

    batch: The pyglet batch that owns the vertex lists.
    entries: Mapping of semantic names (``points``/``segments``) to the
        vertex lists created for rendering. Entries with no geometry are
        omitted.
    """

    batch: "pyglet.graphics.Batch"
    entries: Dict[str, Any]


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


def create_pose_3d_batch(
    pose: PoseData,
    *,
    batch: Optional["pyglet.graphics.Batch"] = None,
    group: Optional["pyglet.graphics.Group"] = None,
    visibility_threshold: float = 0.0,
    point_color: Tuple[int, int, int, int] = (255, 80, 80, 255),
    segment_color: Tuple[int, int, int, int] = (80, 160, 255, 255),
    normalize: bool = True,
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> PoseVisuals:
    """Create pyglet batches that render a 3D pose skeleton.

    Returns a :class:`PoseVisuals` structure storing the batch and the vertex
    lists used for points and segments. Callers can add the batch to a pyglet
    window's draw routine via ``batch.draw()``.
    """

    coords = _prepare_coordinates(
        pose,
        normalize=normalize,
        translate=translate,
        scale=scale,
    )

    if pose.keypoints.shape[1] >= 4:
        visibility = pose.keypoints[:, 3]
    else:
        visibility = np.ones(len(pose.keypoints), dtype=np.float32)

    visible_mask = visibility >= visibility_threshold

    working_batch = batch or pyglet.graphics.Batch()
    shader = pyglet.graphics.get_default_shader()
    entries: Dict[str, Any] = {}

    visible_points = coords[visible_mask]
    if len(visible_points):
        point_vertices = visible_points.flatten().tolist()
        color_vec = [component / 255.0 for component in point_color]
        point_colors = color_vec * len(visible_points)
        entries["points"] = shader.vertex_list(
            len(visible_points),
            pyglet.gl.GL_POINTS,
            batch=working_batch,
            group=cast(Any, group),
            position=("f", point_vertices),
            colors=("f", point_colors),
        )

    segment_vertices: List[float] = []
    vertex_count = 0

    for start, end in POSE_CONNECTIONS:
        if start >= len(coords) or end >= len(coords):
            continue
        if not (visible_mask[start] and visible_mask[end]):
            continue
        segment_vertices.extend(coords[start])
        segment_vertices.extend(coords[end])
        vertex_count += 2

    if vertex_count:
        color_vec = [component / 255.0 for component in segment_color]
        segment_color_floats = color_vec * vertex_count
        entries["segments"] = shader.vertex_list(
            vertex_count,
            pyglet.gl.GL_LINES,
            batch=working_batch,
            group=cast(Any, group),
            position=("f", list(segment_vertices)),
            colors=("f", segment_color_floats),
        )

    return PoseVisuals(batch=working_batch, entries=entries)


__all__ = [
    "PoseVisuals",
    "create_pose_3d_batch",
]
