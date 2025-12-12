import os
import cv2
from pathlib import Path
from PIL import Image

# Augment by horizontal flip for dataset/train/fall and dataset/train/not_fall
TARGET_DIRS = [Path("dataset/train/fall"), Path("dataset/train/not_fall")]

created = 0
skipped = 0
errors = 0

for d in TARGET_DIRS:
    if not d.exists():
        print(f"ディレクトリが見つかりません: {d}")
        continue
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        name = p.stem
        ext = p.suffix
        # skip already-augmented files
        if name.endswith("_flip"):
            skipped += 1
            continue
        try:
            # Use Pillow for robust Unicode path support on Windows
            with Image.open(p) as im:
                flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
                out = d / f"{name}_flip{ext}"
                # Save with the same format inferred from extension
                flipped.save(out)
                created += 1
        except Exception as e:
            print(f"読み込み/保存失敗: {p} -> {e}")
            errors += 1
            continue
        except Exception as e:
            print(f"エラー: {p} -> {e}")
            errors += 1

print("\n=== Augmentation summary ===")
print(f"作成した画像数: {created}")
print(f"スキップした（既に _flip あり）: {skipped}")
print(f"エラー数: {errors}")
print('Flip 画像は各フォルダに "_flip" サフィックス付きで保存されました。')
