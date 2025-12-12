#!/usr/bin/env python
"""
検証用スクリプト: 保存済みモデルで検証データを予測し、class_names・混同行列・確率分布を出力してCSVに保存します。

使い方 (PowerShell):
  & "C:\Program Files\Python311\python.exe" check_inference.py

"""
import os
import csv
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    print('TensorFlow がインポートできません:', e)
    raise

import train_and_infer as tni


def main():
    data_dir = Path('dataset')
    saved_model_dir = Path('saved_model')
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # create datasets
    try:
        train_ds, val_ds, class_names = tni.create_datasets(str(data_dir), img_size=224, batch_size=32, val_split=0.2)
    except Exception as e:
        print('データセットの作成に失敗しました:', e)
        raise

    print('class_names =', class_names)

    # count samples (approx)
    train_count = sum(1 for _ in train_ds.unbatch())
    val_count = sum(1 for _ in val_ds.unbatch())
    print(f'train samples (approx): {train_count}, val samples (approx): {val_count}')

    # load model
    model_path = saved_model_dir / 'model.keras'
    if model_path.exists():
        print('Loading model from', model_path)
        model = keras.models.load_model(str(model_path))
    else:
        print('保存済みモデルが存在しません。新規モデルを構築します（推奨: 事前に学習済みモデルを保存してください）。')
        model = tni.build_model(img_size=224)

    # predict on validation set
    true_labels = []
    probs = []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        preds = preds.reshape(-1)
        probs.extend(preds.tolist())
        true_labels.extend(batch_y.numpy().astype(int).reshape(-1).tolist())

    probs = np.array(probs)
    true_labels = np.array(true_labels)

    if len(true_labels) == 0:
        print('検証データが空です。dataset/train に画像を入れてください。')
        return

    threshold = 0.5
    pred_labels = (probs >= threshold).astype(int)

    TP = int(((pred_labels==1) & (true_labels==1)).sum())
    TN = int(((pred_labels==0) & (true_labels==0)).sum())
    FP = int(((pred_labels==1) & (true_labels==0)).sum())
    FN = int(((pred_labels==0) & (true_labels==1)).sum())

    print('\n=== Confusion Matrix (threshold=0.5) ===')
    print(f'TP (pred=1,true=1): {TP}')
    print(f'FP (pred=1,true=0): {FP}')
    print(f'FN (pred=0,true=1): {FN}')
    print(f'TN (pred=0,true=0): {TN}')

    probs_true0 = probs[true_labels==0]
    probs_true1 = probs[true_labels==1]
    print('\nSamples per true class:')
    print(f"{class_names[0]} (label 0): {int((true_labels==0).sum())}")
    print(f"{class_names[0]} probs: mean={probs_true0.mean() if len(probs_true0)>0 else float('nan'):.4f}, std={probs_true0.std() if len(probs_true0)>0 else float('nan'):.4f}")
    print(f"{class_names[1]} probs: mean={probs_true1.mean() if len(probs_true1)>0 else float('nan'):.4f}, std={probs_true1.std() if len(probs_true1)>0 else float('nan'):.4f}")

    wrong_idx = np.where(pred_labels != true_labels)[0]
    print(f'Number of misclassified samples: {len(wrong_idx)}')
    for i in wrong_idx[:20]:
        print(f'idx={i}, true={true_labels[i]}, pred={pred_labels[i]}, prob_not_fall={probs[i]:.4f}')

    # save CSV summary
    csv_path = saved_model_dir / 'val_prediction_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index','true_label','prob_not_fall','pred_label'])
        for i,(t,p,pr) in enumerate(zip(true_labels, pred_labels, probs)):
            writer.writerow([i,int(t),float(pr),int(p)])

    print('Saved validation summary to', csv_path)


if __name__ == '__main__':
    main()
