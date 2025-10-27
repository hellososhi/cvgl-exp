# OpenCV/GL Project

## Setup

```bash
uv sync
uv run pre-commit install
```

## Directory Structure

```plaintext
.
├── docs
├── misc
├── packages
│   ├── app
│   └── lib
│       ├── game
│       └── pose
├── pyproject.toml
├── README.md
└── uv.lock
```

## Specifications

- スタート画面
    - 画像を表示
    - s キーでゲーム画面へ遷移
- ゲーム画面
    - 姿勢をランダムに生成し、表示（数秒間）
    - ユーザーが姿勢を真似る
    - 数秒間はカメラ画像を表示
    - 数秒経過後に画像を取得し、姿勢推定を実行
- リザルト画面
    - 推定結果と正解姿勢を比較し、スコアを表示
    - r キーで再挑戦、q キーで終了

