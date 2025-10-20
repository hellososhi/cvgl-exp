# 環境について

- このプロジェクトは、 uv による monorepo 環境である。
- パッケージを追加する際は、 `uv add <package> --project <project_path>` とする。
- Python ファイルを実行する際は、 `uv run <file>` とする。
- デバッグ等で Python コマンドにより直接コードを実行する際は、 `uv run python -c "<code>"` とする。

# 実装について

- インストールされているパッケージを確認するためには、 `uv tree -d 1` を実行する。

# 外部ライブラリについて

- Pyglet の実装を行う際は、 [公式ドキュメント](https://pyglet.readthedocs.io/en/latest/index.html) の該当するページを参照すること。
