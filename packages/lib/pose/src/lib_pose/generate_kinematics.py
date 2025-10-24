"""Procedural pose generation with forward kinematics constraints.

This module is currently a design skeleton that documents the staged flow for
generating a random pose while respecting bone-length and joint-angle limits.
Implementation details will be filled in once the structure has been reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Tuple, cast

import numpy as np

from .data import JOINT_ANGLE_LIMITS, POSE_LANDMARKS, POSE_SAMPLE_BONE_LENGTHS, PoseData

# These constants are consumed by downstream helpers; keep references to appease linters
_CONSTRAINT_CONSTANTS = (JOINT_ANGLE_LIMITS, POSE_SAMPLE_BONE_LENGTHS)


JointName = str


@dataclass
class JointFrame:
    """Local coordinate frame anchored at a joint.

    forward: unit vector pointing along the bone away from the parent joint.
    up: unit vector representing the local "up" direction for the joint frame.
    right: unit vector completing the orthonormal basis via right-handed rule.
    position: world-space origin of this joint frame.
    parent: parent joint name for hierarchical traversal.
    """

    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray
    parent: Optional[JointName]


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized copy of `vector`."""

    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector.")
    return vector / norm


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Construct a rotation matrix from an axis-angle representation."""

    axis_n = _normalize(axis)
    x, y, z = axis_n
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _rotate_frame(frame: JointFrame, rotation: np.ndarray) -> JointFrame:
    """Rotate all basis vectors of `frame` with the provided matrix."""

    return JointFrame(
        position=frame.position,
        forward=_normalize(rotation @ frame.forward),
        up=_normalize(rotation @ frame.up),
        right=_normalize(rotation @ frame.right),
        parent=frame.parent,
    )


def _clone_frame(
    frame: JointFrame,
    *,
    position: Optional[np.ndarray] = None,
    parent: Optional[JointName] = None,
) -> JointFrame:
    """Return a shallow copy of `frame` with optional overrides."""

    return JointFrame(
        position=frame.position if position is None else position,
        forward=frame.forward.copy(),
        up=frame.up.copy(),
        right=frame.right.copy(),
        parent=frame.parent if parent is None else parent,
    )


def _apply_yaw_pitch_roll(
    frame: JointFrame, yaw: float, pitch: float, roll: float
) -> JointFrame:
    """Apply intrinsic yaw/pitch/roll rotations to `frame`."""

    rotated = _rotate_frame(frame, _axis_angle_rotation(frame.up, yaw))
    rotated = _rotate_frame(rotated, _axis_angle_rotation(rotated.right, pitch))
    rotated = _rotate_frame(rotated, _axis_angle_rotation(rotated.forward, roll))
    return rotated


def _apply_hinge_rotation(frame: JointFrame, angle: float) -> JointFrame:
    """Rotate `frame` around its local right axis by `angle`."""

    return _rotate_frame(frame, _axis_angle_rotation(frame.right, angle))


def _bone_length(joint_a: JointName, joint_b: JointName) -> float:
    """Fetch the reference bone length between two landmarks."""

    index_a = POSE_LANDMARKS[joint_a]
    index_b = POSE_LANDMARKS[joint_b]
    key: Tuple[int, int] = (min(index_a, index_b), max(index_a, index_b))
    if key not in POSE_SAMPLE_BONE_LENGTHS:
        raise KeyError(f"Missing bone length for {joint_a}-{joint_b}.")
    return float(POSE_SAMPLE_BONE_LENGTHS[key])


def generate_random_pose(
    rng: Optional[np.random.Generator] = None,
) -> PoseData:
    """Return a random `PoseData` sample that satisfies kinematic constraints.

    Algorithm overview (to be implemented):

    1. Seed deterministic RNG if requested and prepare pose buffers.
    2. Lay down the torso scaffold with `neck_bottom` at the origin and both
       shoulders on the x-axis at the prescribed bone length.
    3. Solve for hip placement in the xy-plane so that the midpoint lies on the
       positive y-axis while satisfying the pelvis-to-shoulder chain lengths.
    4. Traverse the kinematic tree breadth-first, sampling joint rotations
       within `JOINT_ANGLE_LIMITS`, updating local frames, and emitting endpoint
       positions that preserve the target bone lengths from
       `POSE_SAMPLE_BONE_LENGTHS`.
    5. Skip face landmarks (0-10) by forcing their positions to the origin and
       visibility to zero, per specification.
    6. Assemble `PoseData` with both camera-space and world-space keypoints and
       default visibility for articulated joints.

    Raises:
            NotImplementedError: The numeric solver is pending implementation.
    """

    generator = rng or np.random.default_rng()

    joints, visibility = _initialize_pose_buffers()
    frames = _initialize_joint_frames()

    _place_torso_scaffold(joints, frames)
    _solve_hip_positions(joints, frames)
    _propagate_spine_and_neck(joints, frames, generator)
    _propagate_upper_limbs(joints, frames, generator)
    _propagate_lower_limbs(joints, frames, generator)

    for name, index in POSE_LANDMARKS.items():
        if index > 10:
            visibility[index] = 1.0

    _suppress_face_landmarks(joints, visibility)
    return _finalize_pose(joints, visibility)


def _initialize_pose_buffers() -> Tuple[
    MutableMapping[JointName, np.ndarray], np.ndarray
]:
    """Allocate storage for joint positions and visibility mask.

    Returns a dict keyed by joint names in `POSE_LANDMARKS` and a visibility
    array with one slot per landmark. The buffers remain zero-initialized here
    and will be populated by subsequent stages.
    """

    joints: MutableMapping[JointName, np.ndarray] = {
        name: np.zeros(3, dtype=np.float64) for name in POSE_LANDMARKS.keys()
    }
    visibility: np.ndarray = np.zeros(len(POSE_LANDMARKS), dtype=np.float64)
    return joints, visibility


def _initialize_joint_frames() -> MutableMapping[JointName, JointFrame]:
    """Create an empty map for joint coordinate frames.

    Frames are populated during torso anchoring and propagated along the limb
    chains. Each frame stores the joint origin and its local orthonormal basis.
    """

    origin: np.ndarray = np.zeros(3, dtype=np.float64)
    right: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    frames: MutableMapping[JointName, JointFrame] = {
        name: JointFrame(
            position=origin.copy(),
            forward=forward.copy(),
            up=up.copy(),
            right=right.copy(),
            parent=None,
        )
        for name in POSE_LANDMARKS.keys()
    }
    return frames


def _place_torso_scaffold(
    joints: MutableMapping[JointName, np.ndarray],
    frames: MutableMapping[JointName, JointFrame],
) -> None:
    """Position `neck_bottom`, `neck_top`, and both shoulders in canonical pose.

    Responsibilities:
            * Set `neck_bottom` to the global origin.
            * Place `left_shoulder` and `right_shoulder` on the x-axis with spacing
              derived from `POSE_SAMPLE_BONE_LENGTHS[(11, 12)]`.
            * Elevate `neck_top` along the positive y-axis using the neck segment
              length `(33, 34)`.
            * Record the root frame for `neck_bottom` and propagate shoulder frames
              aligned with the torso axes.
    """

    origin: np.ndarray = np.zeros(3, dtype=np.float64)
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    forward: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    joints["neck_bottom"][:] = origin
    frames["neck_bottom"] = JointFrame(
        position=origin.copy(),
        forward=forward.copy(),
        up=up.copy(),
        right=right.copy(),
        parent=None,
    )

    shoulder_span: float = POSE_SAMPLE_BONE_LENGTHS[(11, 12)]
    half_span = shoulder_span / 2.0
    left_shoulder_pos = np.array([-half_span, 0.0, 0.0], dtype=np.float64)
    right_shoulder_pos = np.array([half_span, 0.0, 0.0], dtype=np.float64)

    joints["left_shoulder"][:] = left_shoulder_pos
    joints["right_shoulder"][:] = right_shoulder_pos

    frames["left_shoulder"] = JointFrame(
        position=left_shoulder_pos,
        forward=forward.copy(),
        up=up.copy(),
        right=right.copy(),
        parent="neck_bottom",
    )
    frames["right_shoulder"] = JointFrame(
        position=right_shoulder_pos,
        forward=forward.copy(),
        up=up.copy(),
        right=right.copy(),
        parent="neck_bottom",
    )

    neck_length: float = POSE_SAMPLE_BONE_LENGTHS[(33, 34)]
    neck_top_pos = np.array([0.0, neck_length, 0.0], dtype=np.float64)
    joints["neck_top"][:] = neck_top_pos
    frames["neck_top"] = JointFrame(
        position=neck_top_pos,
        forward=forward.copy(),
        up=up.copy(),
        right=right.copy(),
        parent="neck_bottom",
    )


def _solve_hip_positions(
    joints: MutableMapping[JointName, np.ndarray],
    frames: MutableMapping[JointName, JointFrame],
) -> None:
    """Place hips in the xy-plane while honoring pelvis geometry constraints.

    Steps to implement:
            * Use bone lengths `(11, 23)` and `(12, 24)` to triangulate hip centers
              relative to `neck_bottom`.
            * Align hip midpoint on the positive y-axis and ensure the pelvis width
              matches `(23, 24)`.
            * Emit joint frames for the hips aligned with the torso frame.
    """

    left_shoulder = joints["left_shoulder"]
    right_shoulder = joints["right_shoulder"]
    shoulder_span = float(np.linalg.norm(right_shoulder - left_shoulder))

    left_length: float = POSE_SAMPLE_BONE_LENGTHS[(11, 23)]
    right_length: float = POSE_SAMPLE_BONE_LENGTHS[(12, 24)]
    pelvis_width: float = POSE_SAMPLE_BONE_LENGTHS[(23, 24)]
    delta_limit: float = min(left_length, right_length) - 1e-8

    def width_error(delta: float) -> float | None:
        if abs(delta) >= delta_limit:
            return None
        left_term = left_length * left_length - delta * delta
        right_term = right_length * right_length - delta * delta
        if left_term < 0.0 or right_term < 0.0:
            return None
        y_left = float(np.sqrt(left_term))
        y_right = float(np.sqrt(right_term))
        a_val = shoulder_span / 2.0 - delta
        return (2.0 * a_val) ** 2 + (y_left - y_right) ** 2 - pelvis_width**2

    bracket: Optional[Tuple[float, float]] = None
    previous_delta: Optional[float] = None
    previous_value: Optional[float] = None
    sample_grid = np.linspace(-delta_limit, delta_limit, 1025)

    for delta in sample_grid:
        value = width_error(float(delta))
        if value is None or np.isnan(value):
            previous_delta = None
            previous_value = None
            continue
        if value == 0.0:
            bracket = (float(delta), float(delta))
            break
        if previous_delta is not None and previous_value is not None:
            if previous_value * value < 0.0:
                bracket = (previous_delta, float(delta))
                break
        previous_delta = float(delta)
        previous_value = value

    if bracket is None:
        candidate_values: list[Tuple[float, float]] = []
        for delta in sample_grid:
            value = width_error(float(delta))
            if value is None or np.isnan(value):
                continue
            candidate_values.append((float(delta), float(value)))
        if not candidate_values:
            raise RuntimeError("Unable to solve hip placements: no valid samples.")
        delta_solution = min(candidate_values, key=lambda item: abs(item[1]))[0]
    else:
        lo, hi = bracket
        if lo == hi:
            delta_solution = lo
        else:
            f_lo = width_error(lo)
            f_hi = width_error(hi)
            if f_lo is None or f_hi is None:
                raise RuntimeError("Invalid bracket when solving hip placements.")
            for _ in range(60):
                delta_mid = 0.5 * (lo + hi)
                f_mid = width_error(delta_mid)
                if f_mid is None:
                    break
                if abs(f_mid) < 1e-9:
                    lo = hi = delta_mid
                    break
                if f_lo * f_mid < 0.0:
                    hi = delta_mid
                    f_hi = f_mid
                else:
                    lo = delta_mid
                    f_lo = f_mid
            delta_solution = 0.5 * (lo + hi)

    left_y = float(np.sqrt(max(left_length * left_length - delta_solution**2, 0.0)))
    right_y = float(np.sqrt(max(right_length * right_length - delta_solution**2, 0.0)))
    half_width = shoulder_span / 2.0 - delta_solution

    left_hip = np.array([-half_width, left_y, 0.0], dtype=np.float64)
    right_hip = np.array([half_width, right_y, 0.0], dtype=np.float64)

    joints["left_hip"][:] = left_hip
    joints["right_hip"][:] = right_hip

    root_frame = frames["neck_bottom"]
    frames["left_hip"] = JointFrame(
        position=left_hip,
        forward=root_frame.forward.copy(),
        up=root_frame.up.copy(),
        right=root_frame.right.copy(),
        parent="neck_bottom",
    )
    frames["right_hip"] = JointFrame(
        position=right_hip,
        forward=root_frame.forward.copy(),
        up=root_frame.up.copy(),
        right=root_frame.right.copy(),
        parent="neck_bottom",
    )


def _propagate_upper_limbs(
    joints: MutableMapping[JointName, np.ndarray],
    frames: MutableMapping[JointName, JointFrame],
    rng: np.random.Generator,
) -> None:
    """Sample shoulder and arm joints subject to ball-and-hinge constraints."""

    limb_configs = (
        (
            "left",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            (
                ("left_pinky", -0.35, -0.05),
                ("left_index", 0.35, -0.02),
                ("left_thumb", 0.10, 0.65),
            ),
        ),
        (
            "right",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            (
                ("right_pinky", 0.35, -0.05),
                ("right_index", -0.35, -0.02),
                ("right_thumb", -0.10, 0.65),
            ),
        ),
    )

    for side, shoulder_name, elbow_name, wrist_name, digits in limb_configs:
        right_sign = -1.0 if side == "left" else 1.0
        base_forward = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        base_right = np.array([right_sign, 0.0, 0.0], dtype=np.float64)
        base_up = _normalize(np.cross(base_right, base_forward))
        base_forward = _normalize(np.cross(base_up, base_right))

        shoulder_frame = JointFrame(
            position=frames[shoulder_name].position,
            forward=base_forward,
            up=base_up,
            right=base_right,
            parent=frames[shoulder_name].parent,
        )

        shoulder_limits_raw = JOINT_ANGLE_LIMITS[f"{side}_shoulder"]
        if not isinstance(shoulder_limits_raw, dict):  # pragma: no cover - defensive
            raise TypeError("Shoulder limits must provide yaw/pitch/roll ranges.")
        shoulder_limits = cast(Dict[str, Tuple[float, float]], shoulder_limits_raw)
        yaw = rng.uniform(*shoulder_limits["yaw"])
        pitch = rng.uniform(*shoulder_limits["pitch"])
        roll = rng.uniform(*shoulder_limits["roll"])
        shoulder_frame = _apply_yaw_pitch_roll(shoulder_frame, yaw, pitch, roll)
        shoulder_frame = _clone_frame(
            shoulder_frame,
            position=frames[shoulder_name].position,
            parent=frames[shoulder_name].parent,
        )
        frames[shoulder_name] = shoulder_frame

        upper_length = _bone_length(shoulder_name, elbow_name)
        elbow_position = shoulder_frame.position + shoulder_frame.forward * upper_length
        joints[elbow_name][:] = elbow_position
        elbow_frame = _clone_frame(
            shoulder_frame, position=elbow_position, parent=shoulder_name
        )
        elbow_limits = JOINT_ANGLE_LIMITS.get(elbow_name)
        if isinstance(elbow_limits, tuple):
            elbow_angle = rng.uniform(*elbow_limits)
            elbow_frame = _apply_hinge_rotation(elbow_frame, elbow_angle)
        frames[elbow_name] = elbow_frame

        lower_length = _bone_length(elbow_name, wrist_name)
        wrist_position = elbow_frame.position + elbow_frame.forward * lower_length
        joints[wrist_name][:] = wrist_position
        wrist_frame = _clone_frame(
            elbow_frame, position=wrist_position, parent=elbow_name
        )
        wrist_limits = JOINT_ANGLE_LIMITS.get(wrist_name)
        if isinstance(wrist_limits, tuple):
            wrist_angle = rng.uniform(*wrist_limits)
            wrist_frame = _apply_hinge_rotation(wrist_frame, wrist_angle)
        frames[wrist_name] = wrist_frame

    for digit_name, yaw_offset, roll_offset in digits:
        length = _bone_length(wrist_name, digit_name)
        direction = wrist_frame.forward.copy()
        if yaw_offset != 0.0:
            direction = _axis_angle_rotation(wrist_frame.up, yaw_offset) @ direction
        if roll_offset != 0.0:
            direction = _axis_angle_rotation(wrist_frame.right, roll_offset) @ direction
        direction = _normalize(direction)
        digit_position = wrist_frame.position + direction * length
        joints[digit_name][:] = digit_position

        tentative_up = _normalize(wrist_frame.up.copy())
        tentative_right = np.cross(direction, tentative_up)
        if float(np.linalg.norm(tentative_right)) < 1e-6:
            tentative_right = _normalize(wrist_frame.right.copy())
            tentative_up = _normalize(np.cross(tentative_right, direction))
        else:
            tentative_right = _normalize(tentative_right)
            tentative_up = _normalize(np.cross(tentative_right, direction))

        frames[digit_name] = JointFrame(
            position=digit_position,
            forward=direction,
            up=tentative_up,
            right=tentative_right,
            parent=wrist_name,
        )


def _propagate_lower_limbs(
    joints: MutableMapping[JointName, np.ndarray],
    frames: MutableMapping[JointName, JointFrame],
    rng: np.random.Generator,
) -> None:
    """Sample hip, knee, and ankle rotations and update world positions."""

    limb_configs = (
        ("left", "left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index"),
        (
            "right",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_heel",
            "right_foot_index",
        ),
    )

    for side, hip_name, knee_name, ankle_name, heel_name, toe_name in limb_configs:
        right_sign = -1.0 if side == "left" else 1.0
        base_forward = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        base_right = np.array([right_sign, 0.0, 0.0], dtype=np.float64)
        base_up = _normalize(np.cross(base_right, base_forward))
        base_forward = _normalize(np.cross(base_up, base_right))

        hip_frame = JointFrame(
            position=frames[hip_name].position,
            forward=base_forward,
            up=base_up,
            right=base_right,
            parent=frames[hip_name].parent,
        )

        hip_limits_raw = JOINT_ANGLE_LIMITS[f"{side}_hip"]
        if not isinstance(hip_limits_raw, dict):  # pragma: no cover - defensive
            raise TypeError("Hip limits must provide yaw/pitch/roll ranges.")
        hip_limits = cast(Dict[str, Tuple[float, float]], hip_limits_raw)
        hip_yaw = rng.uniform(*hip_limits["yaw"])
        hip_pitch = rng.uniform(*hip_limits["pitch"])
        hip_roll = rng.uniform(*hip_limits["roll"])
        hip_frame = _apply_yaw_pitch_roll(hip_frame, hip_yaw, hip_pitch, hip_roll)
        hip_frame = _clone_frame(
            hip_frame,
            position=frames[hip_name].position,
            parent=frames[hip_name].parent,
        )
        frames[hip_name] = hip_frame

        thigh_length = _bone_length(hip_name, knee_name)
        knee_position = hip_frame.position + hip_frame.forward * thigh_length
        joints[knee_name][:] = knee_position
        knee_frame = _clone_frame(hip_frame, position=knee_position, parent=hip_name)
        knee_limits = JOINT_ANGLE_LIMITS.get(knee_name)
        if isinstance(knee_limits, tuple):
            knee_angle = rng.uniform(*knee_limits)
            knee_frame = _apply_hinge_rotation(knee_frame, knee_angle)
        frames[knee_name] = knee_frame

        shank_length = _bone_length(knee_name, ankle_name)
        ankle_position = knee_frame.position + knee_frame.forward * shank_length
        joints[ankle_name][:] = ankle_position
        ankle_frame = _clone_frame(
            knee_frame, position=ankle_position, parent=knee_name
        )
        ankle_limits = JOINT_ANGLE_LIMITS.get(ankle_name)
        if isinstance(ankle_limits, tuple):
            ankle_angle = rng.uniform(*ankle_limits)
            ankle_frame = _apply_hinge_rotation(ankle_frame, ankle_angle)
        frames[ankle_name] = ankle_frame

        ankle_to_toe = _bone_length(ankle_name, toe_name)
        ankle_to_heel = _bone_length(ankle_name, heel_name)
        heel_to_toe = _bone_length(heel_name, toe_name)

        toe_direction = ankle_frame.forward.copy()
        toe_pitch = -0.1
        toe_yaw = -0.2 if side == "left" else 0.2
        toe_direction = (
            _axis_angle_rotation(ankle_frame.right, toe_pitch) @ toe_direction
        )
        toe_direction = _axis_angle_rotation(ankle_frame.up, toe_yaw) @ toe_direction
        toe_direction = _normalize(toe_direction)

        toe_position = ankle_frame.position + toe_direction * ankle_to_toe
        joints[toe_name][:] = toe_position

        segment_vector = toe_position - ankle_frame.position
        distance_af = float(np.linalg.norm(segment_vector))
        if distance_af < 1e-8:
            ex_unit = ankle_frame.forward.copy()
        else:
            ex_unit = segment_vector / distance_af

        up_axis = ankle_frame.up.copy()
        ey_axis = np.cross(up_axis, ex_unit)
        if float(np.linalg.norm(ey_axis)) < 1e-6:
            ey_axis = np.cross(ankle_frame.right, ex_unit)
        ey_axis = _normalize(ey_axis)

        a_offset = (ankle_to_heel**2 - heel_to_toe**2 + distance_af**2) / (
            2.0 * distance_af
        )
        heel_radius_sq = ankle_to_heel**2 - a_offset**2
        heel_radius_sq = max(heel_radius_sq, 0.0)
        heel_radius = float(np.sqrt(heel_radius_sq))
        circle_center = ankle_frame.position + ex_unit * a_offset
        heel_candidate_a = circle_center + ey_axis * heel_radius
        heel_candidate_b = circle_center - ey_axis * heel_radius

        score_a = float(
            np.dot(heel_candidate_a - ankle_frame.position, ankle_frame.forward)
        )
        score_b = float(
            np.dot(heel_candidate_b - ankle_frame.position, ankle_frame.forward)
        )
        heel_position = heel_candidate_a if score_a < score_b else heel_candidate_b
        joints[heel_name][:] = heel_position

        heel_forward = toe_position - heel_position
        if float(np.linalg.norm(heel_forward)) < 1e-8:
            heel_forward = ankle_frame.forward.copy()
        heel_forward = _normalize(heel_forward)
        heel_right = np.cross(heel_forward, ankle_frame.up)
        if float(np.linalg.norm(heel_right)) < 1e-6:
            heel_right = ankle_frame.right.copy()
        heel_right = _normalize(heel_right)
        heel_up = _normalize(np.cross(heel_right, heel_forward))
        frames[heel_name] = JointFrame(
            position=heel_position,
            forward=heel_forward,
            up=heel_up,
            right=heel_right,
            parent=ankle_name,
        )

        toe_right = np.cross(toe_direction, ankle_frame.up)
        if float(np.linalg.norm(toe_right)) < 1e-6:
            toe_right = ankle_frame.right.copy()
        toe_right = _normalize(toe_right)
        toe_up = _normalize(np.cross(toe_right, toe_direction))
        frames[toe_name] = JointFrame(
            position=toe_position,
            forward=toe_direction,
            up=toe_up,
            right=toe_right,
            parent=ankle_name,
        )


def _propagate_spine_and_neck(
    joints: MutableMapping[JointName, np.ndarray],
    frames: MutableMapping[JointName, JointFrame],
    rng: np.random.Generator,
) -> None:
    """Populate spine chain above the shoulders if additional joints exist."""

    base_frame = frames["neck_bottom"]
    yaw = rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0))
    pitch = rng.uniform(-np.deg2rad(10.0), np.deg2rad(10.0))
    roll = rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0))

    neck_frame = _apply_yaw_pitch_roll(base_frame, yaw, pitch, roll)
    neck_frame = _clone_frame(
        neck_frame,
        position=joints["neck_top"],
        parent="neck_bottom",
    )
    frames["neck_top"] = neck_frame


def _finalize_pose(
    joints: Mapping[JointName, np.ndarray],
    visibility: np.ndarray,
) -> PoseData:
    """Convert accumulated joint data into a `PoseData` payload."""

    landmark_count = len(POSE_LANDMARKS)
    keypoints: np.ndarray = np.zeros((landmark_count, 3), dtype=np.float64)

    for name, index in POSE_LANDMARKS.items():
        keypoints[index] = joints[name]

    keypoints_world: np.ndarray = keypoints.copy()
    pose = PoseData(
        keypoints=keypoints,
        keypoints_world=keypoints_world,
        visibility=visibility.copy(),
        image_size=(0, 0),
    )
    return pose


def _suppress_face_landmarks(
    joints: MutableMapping[JointName, np.ndarray],
    visibility: np.ndarray,
) -> None:
    """Zero-out face landmarks (0-10) and mark them invisible."""

    for name, index in POSE_LANDMARKS.items():
        if index > 10:
            continue
        joints[name][:] = 0.0
        visibility[index] = 0.0
