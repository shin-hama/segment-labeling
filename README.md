# SAM インタラクティブアノテーションツール

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.3.0+-green.svg)

Meta の **Segment Anything Model (SAM)** を使用した、Webベースのインタラクティブ画像セグメンテーション・アノテーションツールです。ユーザーが画像をクリックするだけで、高精度なセグメンテーションマスクを生成し、ラベル付けを行うことができます。

## ✨ 機能

- 🖱️ **ワンクリックセグメンテーション**: 画像の任意の位置をクリックするだけで自動セグメンテーション
- 🎯 **複数候補表示**: SAMが生成する複数のマスク候補から最適なものを選択可能
- 🏷️ **ラベリング機能**: セグメンテーション結果にカスタムラベルを付与
- 👀 **リアルタイム可視化**: アノテーション進行状況をリアルタイムで確認
- 📊 **CSV エクスポート**: アノテーション結果をCSV形式で出力（座標、ポリゴン情報含む）
- 🌐 **Webベース UI**: Gradioによる直感的なWebインターフェース
- ⚡ **GPU/CPU対応**: CUDA利用可能環境では自動的にGPUアクセラレーション

## 🛠️ 技術スタック

- **SAM (Segment Anything Model)**: Meta開発のゼロショット画像セグメンテーションモデル
- **Gradio**: Web UIフレームワーク
- **OpenCV**: コンピュータビジョン処理
- **PyTorch**: 深層学習フレームワーク
- **PIL**: 画像処理ライブラリ
- **NumPy**: 数値計算ライブラリ

## 📋 必要要件

- 16GB以上のRAM（大きなSAMモデル使用時）

## 🚀 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/segment-labeling.git
cd segment-labeling
```

### 2. 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# またはpipを使用
pip install -e .
```

### 3. SAMモデルのダウンロード

```bash
# SAM ViT-H モデル（高精度）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

あるいは、[Segment Anything リポジトリ](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)からダウンロード可能。

**その他のモデルオプション:**
- ViT-L（軽量版）
- ViT-B（最軽量）

## 📖 使用方法

### アプリケーションの起動

uvを使用（推奨）する場合:

```bash
uv run main.py
```

あるいは、直接Pythonで実行する場合:

```bash
python main.py
```

ブラウザで http://localhost:7860 にアクセスしてください。

### アノテーション手順

1. **画像アップロード**
   - 左側の画像アップロード領域に画像をドラッグ&ドロップまたはクリックして選択

2. **セグメンテーション実行**
   - アップロードした画像上で、セグメンテーションしたい領域をクリック
   - SAMが自動的に複数のマスク候補を生成

3. **マスク選択**
   - 表示されたセグメンテーション候補から最適なマスクを選択

4. **ラベル付与**
   - 選択したセグメンテーションに対してラベル名を入力
   - 「ラベルを追加」ボタンで確定

5. **複数オブジェクト対応**
   - 手順2-4を繰り返して複数のオブジェクトにアノテーション

6. **エクスポート**
   - 「Export CSV」ボタンでアノテーション結果をCSV形式でダウンロード

### エクスポート形式

CSVファイルには以下の情報が含まれます：

| カラム名 | 説明 |
|---------|------|
| no | アノテーション番号 |
| type | ラベル名 |
| center_x | セグメンテーションの中心座標（X） |
| center_y | セグメンテーションの中心座標（Y） |
| polygon | ポリゴン座標（JSON形式） |

## 🔧 設定とカスタマイズ

### SAMモデルの変更

[main.py](main.py#L15-L17)の以下の部分を編集してモデルを変更：

```python
sam_checkpoint = "sam_vit_h_4b8939.pth"  # モデルファイル名
model_type = "vit_h"                     # "vit_h", "vit_l", "vit_b"
```

### デバイス設定

CUDAの利用可否は自動検出されますが、手動で設定する場合：

```python
device = "cuda"  # または "cpu"
```

## 🐛 トラブルシューティング

### よくある問題

**CUDA out of memory エラー**
- より小さなSAMモデル（vit_l または vit_b）を使用
- バッチサイズや画像サイズを調整
- `torch.cuda.empty_cache()`でメモリクリア

**セグメンテーション精度が低い**
- より大きなSAMモデル（vit_h）を使用
- クリック位置を調整
- 複数のクリックポイントを使用（将来の機能）

**パフォーマンスが遅い**
- GPU環境での実行を推奨
- 画像サイズを縮小
- より軽量なモデル（vit_b）を使用

## 🏗️ 開発者向け情報

### プロジェクト構造

```
segment-labeling/
├── main.py                 # メインアプリケーション
├── pyproject.toml          # プロジェクト設定
├── README.md              # このファイル
└── sam_vit_h_4b8939.pth   # SAMモデルファイル
```

### コードスタイル

- [Ruff](https://docs.astral.sh/ruff/) による linting
- 型ヒント必須
- PEP 8 準拠
- 関数には docstring を記述

### 開発環境のセットアップ

```bash
# 開発依存関係のインストール
uv sync

# linting実行
uv run ruff check .

# フォーマット実行
uv run ruff format .
```

### アーキテクチャ

- **状態管理**: `AnnotationState` クラスでグローバル状態管理
- **画像処理パイプライン**: PIL → NumPy → OpenCV → SAM → Gradio
- **非同期処理**: Gradioのイベントハンドリング
- **メモリ管理**: 大きなマスクデータの効率的な処理

## 📞 サポート

問題や質問がありましたら、GitHubのIssuesページでお知らせください。
