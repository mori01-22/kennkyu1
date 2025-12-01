使い方（概要）

**セットアップ:**

- Python 3.8+ を推奨します（TensorFlow の対応バージョンに合わせてください）。
- 依存パッケージをインストール（Windows PowerShell の例）:

```powershell
pip install tensorflow opencv-python matplotlib
```

**データ配置:**

- 下記のように配置してください:
  - `dataset/train/fall/` : 倒れるクラスの画像
  - `dataset/train/not_fall/` : 倒れないクラスの画像
  - `dataset/test/` : 推論したい画像（クラスラベル不要）

**実行例:**

- 学習（デフォルト5エポック）→評価→テスト画像推論:

```powershell
python train_and_infer.py --data-dir dataset --epochs 5
```

**主なファイル:**

- `train_and_infer.py` : 学習・評価・推論を1つにまとめたスクリプト

**出力:**

- 学習済みモデルは `saved_model/` に保存されます。
- 学習後に検証データの `accuracy` が表示され、`dataset/test/` 内の各ファイルに対する予測結果も表示されます。

**注意点:**

- 画像枚数が少ないと過学習しやすいです。必要に応じて `base_model.trainable = True` にして微調整（fine-tuning）を行ってください。
- GPU を使う場合は TensorFlow が GPU を認識していることを確認してください。

---

何か追加機能（学習の途中でのモデル保存、コマンドラインで推論のみ実行するモード、結果の可視化など）が必要なら教えてください。
