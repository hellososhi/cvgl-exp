import numpy as np
from lib_pose.data import POSE_CONNECTIONS

# キーポイントの読み込み
keypoints = []
with open("packages/lib/pose/src/lib_pose/demo/ref_pose_keypoints.txt") as f:
    for line in f:
        x, y, z = map(float, line.strip().split())
        keypoints.append([x, y, z])
keypoints_np = np.array(keypoints, dtype=float)  # shape: (33, 3)

# 肩幅（(11, 12)）の計算
shoulder_width = np.linalg.norm(keypoints_np[11] - keypoints_np[12])
if shoulder_width == 0:
    raise ValueError("Shoulder width is zero; check ref_pose_keypoints.txt")

# 骨ごとの長さ（肩幅で正規化）
bone_lengths = {}
for i, j in POSE_CONNECTIONS:
    length = np.linalg.norm(keypoints_np[i] - keypoints_np[j]) / shoulder_width
    bone_lengths[(i, j)] = length

# 結果表示
for (i, j), length in bone_lengths.items():
    print(f"Bone ({i}, {j}): {length:.4f}")
