#!/usr/bin/env python
"""
val_prediction_summary.csv から ROC を近似計算して
Youden の J (sensitivity - (1-specificity)) を最大化する閾値を返します。

使い方:
  & "C:\Program Files\Python311\python.exe" compute_threshold.py

出力: 標準出力に最適閾値を表示し、saved_model/threshold.txt に保存します。
"""
import csv
from pathlib import Path
import numpy as np

csv_path = Path('saved_model') / 'val_prediction_summary.csv'
out_txt = Path('saved_model') / 'threshold.txt'

if not csv_path.exists():
    raise SystemExit(f"検証結果 CSV が見つかりません: {csv_path}。先に `check_inference.py` を実行してください。")

y = []
probs = []
with csv_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row['true_label'])
        p = float(row['prob_not_fall'])
        y.append(t)
        probs.append(p)

y = np.array(y)
probs = np.array(probs)

thresholds = np.linspace(0.0, 1.0, 101)
best = None
best_j = -999
results = []
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    TP = ((preds==1) & (y==1)).sum()
    TN = ((preds==0) & (y==0)).sum()
    FP = ((preds==1) & (y==0)).sum()
    FN = ((preds==0) & (y==1)).sum()
    # sensitivity = TPR, specificity = TN / (TN+FP)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    j = TPR + TNR - 1.0  # Youden's J
    results.append((thr, TPR, TNR, j, TP,FP,FN,TN))
    if j > best_j:
        best_j = j
        best = (thr, TPR, TNR, j)

print('Threshold search (Youden J) completed.')
print(f'Best threshold: {best[0]:.3f}, TPR={best[1]:.3f}, TNR={best[2]:.3f}, J={best[3]:.3f}')

with out_txt.open('w', encoding='utf-8') as f:
    f.write(f"{best[0]:.3f}\n")

print('Saved threshold to', out_txt)
