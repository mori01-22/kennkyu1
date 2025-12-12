#!/usr/bin/env python
"""
テスト画像を推論して詳細な分析を行うスクリプト。
- `prob_not_fall`（モデルの出力）と `prob_fall`（1-prob）をCSVに保存
- 閾値で分類し、不確かなサンプルも抽出して表示

使い方:
  & "C:\Program Files\Python311\python.exe" analyze_test_results.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path('saved_model') / 'model.keras'
TEST_DIR = Path('dataset') / 'test'
OUT_CSV = Path('saved_model') / 'test_analysis.csv'
IMG_SIZE = (224, 224)
THRESHOLD = 0.5
UNCERTAIN_MARGIN = 0.1  # |prob_not_fall - 0.5| < 0.1 を不確かとする


def classify(prob, threshold=THRESHOLD, margin=UNCERTAIN_MARGIN):
    pred = 'not_fall' if prob >= threshold else 'fall'
    if abs(prob - 0.5) < margin:
        category = 'uncertain'
    else:
        category = 'predicted_not_fall' if pred == 'not_fall' else 'predicted_fall'
    return pred, category


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"モデルが見つかりません: {MODEL_PATH}")
    if not TEST_DIR.exists():
        raise SystemExit(f"テストディレクトリが見つかりません: {TEST_DIR}")

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    files = [p for p in sorted(TEST_DIR.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not files:
        print('テスト画像がありません。')
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in files:
        try:
            with Image.open(p) as im:
                im = im.convert('RGB').resize(IMG_SIZE)
                arr = preprocess_input(np.array(im).astype('float32'))
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr, verbose=0)
                prob_not_fall = float(pred[0][0])
                prob_fall = 1.0 - prob_not_fall
                pred_label, category = classify(prob_not_fall)
                rows.append((p.name, prob_not_fall, prob_fall, pred_label, category))
        except Exception as e:
            print(f'処理失敗: {p} -> {e}')

    # save CSV
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prob_not_fall', 'prob_fall', 'pred_label', 'category'])
        for r in rows:
            writer.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", r[3], r[4]])

    # summary
    total = len(rows)
    counts = {}
    for r in rows:
        counts[r[4]] = counts.get(r[4], 0) + 1

    print(f"Processed {total} test images. Summary:")
    for k in ['predicted_not_fall', 'predicted_fall', 'uncertain']:
        print(f"  {k}: {counts.get(k,0)}")

    # show top confident mis-match candidates: high-confidence fall (prob_not_fall small) and high-confidence not_fall
    rows_sorted_fall = sorted(rows, key=lambda x: x[1])  # ascending prob_not_fall -> confident fall first
    rows_sorted_notfall = sorted(rows, key=lambda x: -x[1])

    print('\nTop 10 confident predicted FALL (lowest prob_not_fall):')
    for r in rows_sorted_fall[:10]:
        print(f"  {r[0]}: prob_not_fall={r[1]:.4f}, pred={r[3]}")

    print('\nTop 10 confident predicted NOT_FALL (highest prob_not_fall):')
    for r in rows_sorted_notfall[:10]:
        print(f"  {r[0]}: prob_not_fall={r[1]:.4f}, pred={r[3]}")

    print(f'CSV saved to {OUT_CSV}')


if __name__ == '__main__':
    main()
