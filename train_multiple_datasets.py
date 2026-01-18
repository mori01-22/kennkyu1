#!/usr/bin/env python
"""
複数のデータセットを一度に学習して、それぞれのモデルを保存するスクリプト

使い方:
  - フォルダ構成:
    dataset1/
      train/
        fall/
        not_fall/
    
    dataset2/
      train/
        fall/
        not_fall/
    
    dataset3/
      train/
        fall/
        not_fall/

  - 実行:
    python train_multiple_datasets.py --datasets dataset1 dataset2 dataset3 --epochs 10

  - または、デフォルト（dataset1, dataset2, dataset3）で実行:
    python train_multiple_datasets.py --epochs 10
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
import matplotlib.pyplot as plt


def create_datasets(data_dir, img_size=224, batch_size=32, val_split=0.2):
    """トレーニング用と検証用のデータセットを作成"""
    train_dir = str(Path(data_dir) / "train")

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

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(img_size=224, dropout_rate=0.2):
    """モデルを構築"""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def train_and_save_model(data_dir, save_dir, img_size=224, epochs=100, batch_size=32):
    """1つのデータセットに対して学習・評価・保存を行う"""
    
    print(f"\n{'='*60}")
    print(f"📚 データセット: {data_dir}")
    print(f"{'='*60}")
    
    # データセット作成
    try:
        train_ds, val_ds, class_names = create_datasets(
            data_dir, 
            img_size=img_size, 
            batch_size=batch_size
        )
        print(f"✓ クラス: {class_names}")
    except Exception as e:
        print(f"✗ データセット読み込み失敗: {e}")
        return False

    # モデル構築と学習
    model = build_model(img_size=img_size)
    print("\n--- 学習開始 ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

    # 評価
    print("\n--- 評価 ---")
    loss, acc = model.evaluate(val_ds)
    print(f"Validation loss: {loss:.4f}, accuracy: {acc:.4f}")

    # 学習曲線の保存
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history.get('accuracy', []), label='train_acc')
        ax[0].plot(history.history.get('val_accuracy', []), label='val_acc')
        ax[0].set_title('Accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('accuracy')
        ax[0].legend()

        ax[1].plot(history.history.get('loss', []), label='train_loss')
        ax[1].plot(history.history.get('val_loss', []), label='val_loss')
        ax[1].set_title('Loss')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('loss')
        ax[1].legend()

        plot_path = Path(save_dir) / 'learning_curve.png'
        fig.tight_layout()
        fig.savefig(str(plot_path))
        plt.close(fig)
        print(f"✓ 学習曲線を保存: {plot_path}")
    except Exception as e:
        print(f"✗ 学習曲線の保存失敗: {e}")

    # モデル保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(save_path / 'model.keras'))
        print(f"✓ モデルを保存: {save_path / 'model.keras'}")
        
        # メタ情報も保存
        with open(save_path / 'metadata.txt', 'w') as f:
            f.write(f"Dataset: {data_dir}\n")
            f.write(f"Classes: {class_names}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"Validation Loss: {loss:.4f}\n")
        print(f"✓ メタ情報を保存: {save_path / 'metadata.txt'}")
        
    except Exception as e:
        print(f"✗ モデル保存失敗: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='複数のデータセットで学習')
    parser.add_argument('--datasets', nargs='+', default=['dataset1', 'dataset2', 'dataset3'],
                        help='学習するデータセットフォルダのリスト')
    parser.add_argument('--img-size', type=int, default=224, help='画像サイズ')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=100, help='エポック数')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 複数データセット学習スクリプト")
    print("="*60)
    print(f"対象データセット: {args.datasets}")
    print(f"エポック数: {args.epochs}")
    print(f"画像サイズ: {args.img_size}x{args.img_size}")
    print("="*60)

    results = {}
    for dataset_dir in args.datasets:
        # モデル保存先は saved_model_{dataset名} とする
        dataset_name = Path(dataset_dir).name
        save_dir = f'saved_model_{dataset_name}'
        
        success = train_and_save_model(
            dataset_dir,
            save_dir,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        results[dataset_dir] = success

    # 結果サマリー
    print("\n" + "="*60)
    print("📊 学習結果サマリー")
    print("="*60)
    for dataset_dir, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        dataset_name = Path(dataset_dir).name
        save_dir = f'saved_model_{dataset_name}'
        print(f"{dataset_dir:20} → {save_dir:25} {status}")
    print("="*60)


if __name__ == '__main__':
    main()
