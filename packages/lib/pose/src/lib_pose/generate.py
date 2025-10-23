"""Pose generation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .data import POSE_LANDMARKS, POSE_SAMPLE_BONE_LENGTHS, PoseData

__all__ = ["generate_random_pose"]


LandmarkIndex = int
ArrayF64: TypeAlias = NDArray[np.float64]
ArrayF32: TypeAlias = NDArray[np.float32]


def generate_random_pose(
    image_size: Sequence[int] = (640, 480),
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 5,
    max_iterations: int = 1500,
    tolerance: float = 1e-3,
) -> PoseData:
    """Generate a random pose that matches the sample bone-length constraints."""

    generator = _ensure_generator(rng)
    edges = _build_edges()
    anchors = _default_anchor_positions()
    fixed_indices = set(anchors.keys())
    num_landmarks = len(POSE_LANDMARKS)

    for _ in range(max_attempts):
        positions: ArrayF64 = generator.normal(scale=0.25, size=(num_landmarks, 3))
        for idx, pos in anchors.items():
            positions[idx] = pos.copy()

        success = _project_bone_lengths(
            positions,
            edges,
            fixed_indices,
            generator,
            max_iterations,
            tolerance,
        )
        if not success:
            continue

        max_error = _max_length_error(positions, edges)
        if max_error > tolerance * 5:
            # The solver may have stalled in a shallow minimum; retry.
            continue

        transformed = _apply_random_transform(positions, generator)
        keypoints: ArrayF32 = transformed.astype(np.float32)
        keypoints_world: ArrayF32 = keypoints.copy()
        visibility: ArrayF32 = np.ones(num_landmarks, dtype=np.float32)
        width, height = int(image_size[0]), int(image_size[1])
        return PoseData(
            keypoints=keypoints,
            keypoints_world=keypoints_world,
            visibility=visibility,
            image_size=(width, height),
        )

    raise RuntimeError("Failed to generate a pose that satisfies bone constraints.")


def _ensure_generator(source: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(source, np.random.Generator):
        return source
    return np.random.default_rng(source)


def _build_edges() -> list[tuple[LandmarkIndex, LandmarkIndex, float]]:
    edges: list[tuple[LandmarkIndex, LandmarkIndex, float]] = []
    for (i, j), length in POSE_SAMPLE_BONE_LENGTHS.items():
        edges.append((i, j, float(length)))
    return edges


def _default_anchor_positions() -> Dict[LandmarkIndex, ArrayF64]:
    left_shoulder = POSE_LANDMARKS["left_shoulder"]
    right_shoulder = POSE_LANDMARKS["right_shoulder"]
    left_hip = POSE_LANDMARKS["left_hip"]

    shoulder_span = _bone_length(left_shoulder, right_shoulder)
    torso_length = _bone_length(left_shoulder, left_hip)

    anchors: Dict[LandmarkIndex, ArrayF64] = {
        left_shoulder: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        right_shoulder: np.array([shoulder_span, 0.0, 0.0], dtype=np.float64),
        left_hip: np.array([0.0, -torso_length, 0.0], dtype=np.float64),
    }
    return anchors


def _project_bone_lengths(
    positions: ArrayF64,
    edges: Sequence[tuple[LandmarkIndex, LandmarkIndex, float]],
    fixed_indices: set[LandmarkIndex],
    generator: np.random.Generator,
    max_iterations: int,
    tolerance: float,
) -> bool:
    edge_indices = list(range(len(edges)))
    for _ in range(max_iterations):
        generator.shuffle(edge_indices)
        max_error = 0.0
        for edge_idx in edge_indices:
            i, j, length = edges[edge_idx]
            pi = positions[i]
            pj = positions[j]
            delta = pj - pi
            dist = float(np.linalg.norm(delta))
            if dist < 1e-8:
                direction = _random_unit_vector(generator)
            else:
                direction = delta / dist
            diff = dist - length
            max_error = max(max_error, float(abs(diff)))
            if abs(diff) < tolerance:
                continue

            inv_mass_i = 0.0 if i in fixed_indices else 1.0
            inv_mass_j = 0.0 if j in fixed_indices else 1.0
            inv_total = inv_mass_i + inv_mass_j
            if inv_total == 0.0:
                continue

            correction = direction * diff
            if inv_mass_i > 0.0:
                positions[i] += correction * (inv_mass_i / inv_total)
            if inv_mass_j > 0.0:
                positions[j] -= correction * (inv_mass_j / inv_total)

        if max_error < tolerance:
            return True
    return False


def _apply_random_transform(
    positions: ArrayF64, generator: np.random.Generator
) -> ArrayF64:
    rotated = positions @ _random_rotation_matrix(generator).T
    translation = generator.uniform(-0.5, 0.5, size=3).astype(np.float64)
    return rotated + translation


def _random_rotation_matrix(generator: np.random.Generator) -> ArrayF64:
    random_matrix = generator.normal(size=(3, 3))
    q, _ = np.linalg.qr(random_matrix)
    # Ensure a proper rotation (determinant == 1)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _random_unit_vector(generator: np.random.Generator) -> ArrayF64:
    vec = generator.normal(size=3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def _max_length_error(
    positions: ArrayF64,
    edges: Iterable[tuple[LandmarkIndex, LandmarkIndex, float]],
) -> float:
    max_error = 0.0
    for i, j, length in edges:
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        max_error = max(max_error, float(abs(dist - length)))
    return float(max_error)


def _bone_length(i: LandmarkIndex, j: LandmarkIndex) -> float:
    try:
        return float(POSE_SAMPLE_BONE_LENGTHS[(i, j)])
    except KeyError:
        return float(POSE_SAMPLE_BONE_LENGTHS[(j, i)])
