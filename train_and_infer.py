#!/usr/bin/env python
"""
倒壊判定（fall / not_fall）を学習・評価・推論する単一ファイルスクリプト

使い方:
  - まず画像を以下のように配置してください:
    dataset/
      train/
        fall/       (倒れるクラスの画像)
        not_fall/   (倒れないクラスの画像)
      test/         (分類したいテスト画像をここに入れる)

  - 依存ライブラリをインストール:
    `pip install tensorflow opencv-python matplotlib`

  - 実行（デフォルトで学習→評価→テスト画像分類を行います）:
    `python train_and_infer.py --data-dir dataset --epochs 5`

このファイルは初心者向けのコメントを多めに入れています。
"""

import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


def create_datasets(data_dir, img_size=224, batch_size=32, val_split=0.2):
    """
    トレーニング用のデータセットと検証用データセットを作成します。
    dataset/train 配下にクラスフォルダ (fall, not_fall) がある想定です。
    """
    train_dir = str(Path(data_dir) / "train")

    # image_dataset_from_directory を使って、簡単にデータセットを作成します
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    # class_names は image_dataset_from_directory がセットする属性なので
    # キャッシュや prefetch を適用する前に取り出しておきます。
    class_names = train_ds.class_names

    # パフォーマンス向上のためにキャッシュ・プリフェッチを設定
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(img_size=224, dropout_rate=0.2):
    """
    MobileNetV2 をバックボーンに使った転移学習モデルを構築します。
    出力は2クラスのバイナリ（sigmoid）です。
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(img_size, img_size, 3))
    base_model.trainable = False  # まずは凍結して特徴抽出器として使う

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)  # MobileNetV2 の前処理
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # バイナリ分類

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def load_and_preprocess_image(path, img_size=224):
    """ファイルパスから画像を読み込み、前処理をしてバッチ形状を返す"""
    img = image.load_img(path, target_size=(img_size, img_size))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_test_images(model, test_dir, img_size=224, class_names=None, threshold=0.5):
    """
    `test_dir` にある画像を1つずつモデルで分類して表示します。
    クラス名は `class_names`（例: ['fall','not_fall']）を渡すと良いです。
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"テスト用ディレクトリが見つかりません: {test_dir}")
        return

    img_files = [p for p in test_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not img_files:
        print(f"テスト画像が見つかりません: {test_dir}")
        return

    results = []
    for p in img_files:
        x = load_and_preprocess_image(str(p), img_size=img_size)
        pred = model.predict(x)
        prob = float(pred[0][0])
        label = 1 if prob >= threshold else 0
        label_name = class_names[label] if class_names is not None else str(label)
        results.append((p.name, label_name, prob))

    # 結果を表示
    print(f'\n=== Test Results (threshold={threshold}) ===')
    for name, lbl, prob in results:
        print(f"{name}: {lbl} (prob={prob:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train and run inference for fall/not_fall')
    parser.add_argument('--data-dir', type=str, default='dataset', help='dataset のルートフォルダ')
    parser.add_argument('--img-size', type=int, default=224, help='画像サイズ (px)')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=5, help='学習エポック数')
    parser.add_argument('--save-dir', type=str, default='saved_model', help='学習済みモデル保存先')
    parser.add_argument('--use-saved', action='store_true', help='既に保存済みのモデルを読み込んで推論のみ行う')
    parser.add_argument('--threshold', type=float, default=0.5, help='推論時の閾値（デフォルト0.5）')
    args = parser.parse_args()

    # データセットの作成
    train_ds, val_ds, class_names = create_datasets(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    print(f"クラス: {class_names}")

    model = None
    # 既存の保存モデルを使う場合は読み込む
    save_path = Path(args.save_dir)
    if args.use_saved:
        model_file = save_path / 'model.keras'
        if model_file.exists():
            print(f"既存モデルを読み込みます: {model_file}")
            model = keras.models.load_model(str(model_file))
        else:
            print(f"保存済みモデルが見つかりません: {model_file}。新規にモデルを構築します。")

    # モデルを構築して学習する（use_saved が True で読み込めていれば学習はスキップ）
    if model is None:
        model = build_model(img_size=args.img_size)
        model.summary()
        # 学習
        print('\n--- Training ---')
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # 評価（検証データでの精度表示）
    print('\n--- Evaluation on validation data ---')
    loss, acc = model.evaluate(val_ds)
    print(f"Validation loss: {loss:.4f}, accuracy: {acc:.4f}")

    # モデル保存
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    # Keras 3 の変更により拡張子なしで model.save を呼ぶとエラーになるため
    # 安全に保存するロジックを入れる（.keras 形式を優先、失敗したら export を試す）
    try:
        model.save(str(save_path / 'model.keras'))
        print(f"モデルを Keras 形式で保存しました: {save_path / 'model.keras'}")
    except Exception:
        try:
            # SavedModel 形式が必要な場合はこちらを使用
            model.export(str(save_path))
            print(f"モデルを SavedModel 形式で保存しました: {save_path}")
        except Exception as e:
            print(f"モデル保存に失敗しました: {e}")

    # テスト画像を分類（閾値を渡す）
    test_dir = Path(args.data_dir) / 'test'
    predict_test_images(model, test_dir, img_size=args.img_size, class_names=class_names, threshold=args.threshold)


if __name__ == '__main__':
    main()
