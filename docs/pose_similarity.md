時系列を考慮しない静的な二次元姿勢データ（キーポイントの集合）同士の類似度を計算する3つの手法について、Pythonでの簡単な実装例を示します。

ここでは、`numpy` と `scipy` ライブラリを使用します。

-----

### 準備：サンプルデータ

比較のために、2つの姿勢データを`numpy`配列として定義します。
各姿勢は **N個のキーポイント** からなり、それぞれのキーポイントは **(x, y, v)** の3つの値を持つとします。`v`はキーポイントの可視性（visibility）を示し、`v > 0` の場合のみ有効なキーポイントとして扱います（OKSの実装で利用します）。

```python
import numpy as np

# サンプルデータ (N, 3) 形式: N個のキーポイント, (x, y, visibility)
# 非常に似ている2つの姿勢を想定
pose1 = np.array([
    [250, 200, 1],  # Nose
    [255, 250, 1],  # Left Shoulder
    [245, 250, 1],  # Right Shoulder
    [260, 300, 1],  # Left Elbow
    [240, 300, 1],  # Right Elbow
    [265, 350, 1],  # Left Wrist
    [235, 350, 1]   # Right Wrist
])

# pose1と少しだけ異なる姿勢
pose2 = np.array([
    [252, 202, 1],  # Nose
    [257, 251, 1],  # Left Shoulder
    [247, 251, 1],  # Right Shoulder
    [265, 305, 1],  # Left Elbow
    [238, 303, 1],  # Right Elbow
    [270, 355, 1],  # Left Wrist
    [230, 352, 1]   # Right Wrist
])

# pose1とは大きく異なる姿勢（片手を上げている）
pose3 = np.array([
    [300, 200, 1],  # Nose
    [305, 250, 1],  # Left Shoulder
    [295, 250, 1],  # Right Shoulder
    [305, 300, 1],  # Left Elbow
    [250, 220, 1],  # Right Elbow (Raised)
    [305, 350, 1],  # Left Wrist
    [220, 180, 1]   # Right Wrist (Raised)
])
```

-----

## 1\. コサイン類似度 (Cosine Similarity)

姿勢を構成する全キーポイントの座標を1次元のベクトルに変換し、2つのベクトルがなす角度のコサインを計算します。これにより、人物の全体的なスケールに影響されずに姿勢の「形」の類似度を評価できます。

  - **特徴**: スケール不変。計算が高速。
  - **実装**: `scipy`の`distance.cosine`は「コサイン距離（1 - 類似度）」を返すため、`1`から引くことで類似度を求めます。

<!-- end list -->

```python
from scipy.spatial import distance

def calculate_cosine_similarity(p1, p2):
    """
    2つの姿勢のコサイン類似度を計算する。
    Args:
        p1 (np.array): (N, 3) の姿勢データ1
        p2 (np.array): (N, 3) の姿勢データ2
    Returns:
        float: コサイン類似度 (0から1の範囲)
    """
    # x, y座標のみを抽出し、1次元ベクトルに変換
    vec1 = p1[:, :2].flatten()
    vec2 = p2[:, :2].flatten()
    
    # コサイン距離を計算し、1から引いて類似度を求める
    cosine_similarity = 1 - distance.cosine(vec1, vec2)
    
    return cosine_similarity

# --- 計算例 ---
similarity_1_2 = calculate_cosine_similarity(pose1, pose2)
similarity_1_3 = calculate_cosine_similarity(pose1, pose3)

print(f"コサイン類似度 (pose1 vs pose2): {similarity_1_2:.4f}") # 非常に高い値になるはず
print(f"コサイン類似度 (pose1 vs pose3): {similarity_1_3:.4f}") # 低めの値になるはず
```

-----

## 2\. Procrustes Analysis

位置・回転・スケールの影響を取り除き、純粋な「形状」としてどれだけ似ているかを評価します。`scipy.spatial.procrustes`は、2つの点群を最適に重ね合わせた後の差（非類似度）を返します。

  - **特徴**: 位置、回転、スケールに対して不変。形状の比較に特化。
  - **実装**: `scipy.spatial.procrustes`は3つの値を返します（`mtx1`, `mtx2`, `disparity`）。`disparity`が正規化された非類似度です。これを`1`から引くことで、簡易的な類似度スコアとします。

<!-- end list -->

```python
from scipy.spatial import procrustes

def calculate_procrustes_similarity(p1, p2):
    """
    Procrustes解析を用いて2つの姿勢の類似度を計算する。
    Args:
        p1 (np.array): (N, 3) の姿勢データ1
        p2 (np.array): (N, 3) の姿勢データ2
    Returns:
        float: 類似度 (disparityを1から引いた値)
    """
    # x, y座標のみを抽出
    points1 = p1[:, :2]
    points2 = p2[:, :2]
    
    # Procrustes解析を実行
    # 戻り値: 整列後の点群1, 整列後の点群2, 非類似度(disparity)
    mtx1, mtx2, disparity = procrustes(points1, points2)
    
    # disparityは0に近いほど似ている。1から引いて類似度とする。
    return 1 - disparity

# --- 計算例 ---
similarity_1_2_proc = calculate_procrustes_similarity(pose1, pose2)
similarity_1_3_proc = calculate_procrustes_similarity(pose1, pose3)

print(f"Procrustes類似度 (pose1 vs pose2): {similarity_1_2_proc:.4f}")
print(f"Procrustes類似度 (pose1 vs pose3): {similarity_1_3_proc:.4f}")
```

-----

## 3\. Object Keypoint Similarity (OKS)

人物のスケール（バウンディングボックスの面積）と、キーポイントの種類ごとのばらつき（`k`値）を考慮して類似度を計算します。人間の知覚的な類似性に近い評価が可能です。

  - **特徴**: 人物のスケールとキーポイントの重要度を考慮。
  - **実装**: COCOデータセットで定義されている`k`値を参考に実装します。OKSはGT（正解データ）と予測データの比較に用いられるため、ここでは`pose1`をGT、`pose2`を予測と見なして計算します。

<!-- end list -->

```python
def calculate_oks(p1, p2, bbox_area, k_values):
    """
    2つの姿勢間のObject Keypoint Similarity (OKS)を計算する。
    Args:
        p1 (np.array): (N, 3) の姿勢データ1 (GTと見なす)
        p2 (np.array): (N, 3) の姿勢データ2 (予測と見なす)
        bbox_area (float): p1の人物のバウンディングボックスの面積
        k_values (np.array): (N,) の各キーポイントのk値
    Returns:
        float: OKSスコア
    """
    if bbox_area == 0:
        return 0.0

    # p1で可視(visible)なキーポイントのみを対象とする
    visible_indices = p1[:, 2] > 0
    if np.sum(visible_indices) == 0:
        return 0.0
        
    # キーポイント間のユークリッド距離の2乗を計算
    d_sq = np.sum((p1[visible_indices, :2] - p2[visible_indices, :2])**2, axis=1)
    
    # スケール(s)とk値を取得
    s = np.sqrt(bbox_area)
    k = k_values[visible_indices]
    
    # OKSの計算
    # exp(-d^2 / 2s^2k^2)
    oks = np.exp(-d_sq / (2 * (s**2) * (k**2)))
    
    # 可視キーポイントのOKSの平均値を返す
    return np.mean(oks)

# --- 計算例 ---
# COCOデータセットのk値 (最初の7点に対応)
# Nose, L/R-Shoulder, L/R-Elbow, L/R-Wrist
COCO_K_VALUES = np.array(
    [0.026, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062]
)

# pose1を囲むバウンディングボックスの面積を仮定
# 例: (265-235) * (350-200) = 30 * 150 = 4500
bbox_area_1 = 4500.0

# pose3を囲むバウンディングボックスの面積を仮定
# 例: (305-220) * (350-180) = 85 * 170 = 14450
bbox_area_3 = 14450.0


oks_1_2 = calculate_oks(pose1, pose2, bbox_area_1, COCO_K_VALUES)
oks_1_3 = calculate_oks(pose1, pose3, bbox_area_1, COCO_K_VALUES)

print(f"OKS (pose1 vs pose2): {oks_1_2:.4f}")
print(f"OKS (pose1 vs pose3): {oks_1_3:.4f}")
```
