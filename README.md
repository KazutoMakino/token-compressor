# template_repository_python

## 概要

Pythonプロジェクトの開始に必要な基本的な設定と構成を提供するテンプレートリポジトリです。  
`uv`を用いた依存関係の管理、`pre-commit`を用いたコードのリンティング (`mypy`) およびフォーマット (`ruff`) の設定ファイルが含み、すぐに環境構築することが可能にします。

## 主な機能

* **依存関係管理:** `uv` が構成されており、依存関係の管理が容易です。
* **pre-commit:** `pre-commit` が構成されており、コミット時に以下レンティングとフォーマットを実行し、コード品質を維持できます。
* **リンティングとフォーマット:** `mypy`, `ruff` がプリセットされており、コード品質を維持できます。

## 使い方

### テンプレートからリポジトリを作成する

1. このリポジトリをforkします。
2.  新しいリポジトリを作成します。
3.  "Repository template" でドロップダウンから上記リポジトリを選択します。

### ローカル環境のセットアップ

上記作成したリポジトリをローカルにクローンします。

```bash
git clone https://github.com/your-username/your-new-repository.git
cd your-new-repository
```

### 依存関係のインストール

プロジェクトの依存関係をインストールします。  
`uv`を使用する場合は、次のコマンドを実行します。

```bash
uv sync
```

※ パッケージ管理の`uv`がない場合は次のコマンドでインストールしてから上記実行してください。

```bash
pip install uv
```

依存関係の追加／削除／更新するには次のコマンドで実行します。  
(初期状態では主に`jupyterlab`,`pre-commit`,`ruff`,`uv`のパッケージが依存関係として入っており、`pyproject.toml`に記載されています。詳細な依存関係については`uv.lock`に記載されています)

```bash
uv add {package_name}
uv remove {package_name}
uv sync
```

### uvで構築した環境の入り方

`uv`で構築した`.venv/`の環境に入るには以下コマンドで実行します。  
(以下例はwindowsの場合。windows以外では `.venv/Scripts` を `.venv/bin` と置き換えてください)

```bash
# bash系の場合
source .venv\Scripts\activate

# コマンドプロンプトの場合
.venv\Scripts\activate.bat

# PowerShellの場合
.venv\Scripts\activate.ps1
```

### pre-commitの適用

`pre-commit`は次のコマンドで有効化します。  
(設定は `.pre-commit-config.yaml` に記載しています)

```bash
pre-commit install
pre-commit autoupdate
```

### リンティングとフォーマット

`pre-commit`を設定することによりコミット時にコードのリンティングとフォーマットが自動で行われるのですが、コマンドラインからマニュアル実行した場合は以下のコマンドを実行します。

```bash
pre-commit run --all-files
```
