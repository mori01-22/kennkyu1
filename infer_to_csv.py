import csv
from pathlib import Path
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Config
MODEL_PATH = Path('saved_model') / 'model.keras'
TEST_DIR = Path('dataset') / 'test'
OUT_CSV = Path('saved_model') / 'test_results.csv'
IMG_SIZE = (224, 224)
# 閾値は環境変数 TEST_THRESHOLD によって上書き可能（例: TEST_THRESHOLD=0.4）
THRESHOLD = float(os.environ.get('TEST_THRESHOLD', 0.5))

if not MODEL_PATH.exists():
    raise SystemExit(f"モデルが見つかりません: {MODEL_PATH} (まず学習して model.keras を保存してください)")
if not TEST_DIR.exists():
    raise SystemExit(f"テストディレクトリが見つかりません: {TEST_DIR}")

print(f"モデルを読み込みます: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

files = [p for p in sorted(TEST_DIR.iterdir()) if p.is_file()]
print(f"テスト画像数: {len(files)}")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open('w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['filename', 'prob_not_fall', 'label'])

    for p in files:
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                im = im.resize(IMG_SIZE)
                arr = np.array(im)
                arr = preprocess_input(arr.astype(np.float32))
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr)
                # pred shape for binary sigmoid: (1,1)
                prob = float(pred[0][0])
                label = 'not_fall' if prob >= THRESHOLD else 'fall'
                writer.writerow([str(p.name), f"{prob:.6f}", label])
                print(f"{p.name}: {label} (prob={prob:.6f})")
        except Exception as e:
            print(f"処理失敗: {p} -> {e}")

print(f"CSV に保存しました: {OUT_CSV}")
