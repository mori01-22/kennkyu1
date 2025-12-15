#!/usr/bin/env python
"""
指定画像について
- モデルの予測（prob_not_fall）を出力
- テスト時拡張（左右反転・回転など）での確率を表示
- Grad-CAM を作成して `saved_model/gradcam_<filename>.png` に保存

使い方 (PowerShell):
  & "C:\Program Files\Python311\python.exe" gradcam_tta.py "スクリーンショット 2025-11-20 180133_flip.png"
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_image(path, img_size=(224,224)):
    im = Image.open(path).convert('RGB').resize(img_size)
    arr = np.array(im).astype('float32')
    return im, preprocess_input(arr)


def predict_prob(model, arr):
    x = np.expand_dims(arr, axis=0)
    p = float(model.predict(x, verbose=0)[0,0])
    return p


def tta_predictions(model, pil_img, img_size=(224,224)):
    # produce a few augmentations: original, hflip, rotate +-15
    imgs = []
    imgs.append(pil_img)
    imgs.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    imgs.append(pil_img.rotate(15, resample=Image.BILINEAR))
    imgs.append(pil_img.rotate(-15, resample=Image.BILINEAR))
    probs = []
    for im in imgs:
        arr = preprocess_input(np.array(im.resize(img_size)).astype('float32'))
        probs.append(predict_prob(model, arr))
    return probs


def make_gradcam(model, img_array, last_conv_layer_name=None):
    # img_array: preprocessed numpy array (H,W,3)
    img_tensor = tf.expand_dims(img_array, axis=0)
    if last_conv_layer_name is None:
        # try to find a conv layer
        for layer in reversed(model.layers):
            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise RuntimeError('No conv layer found for Grad-CAM')

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap


def save_gradcam_on_image(pil_img, heatmap, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    heatmap = np.uint8(255 * heatmap)
    cmap = cm.get_cmap('jet')
    colored = cmap(heatmap)
    colored = np.delete(colored, 3, 2)
    colored = Image.fromarray((colored * 255).astype('uint8')).resize(pil_img.size)
    blended = Image.blend(pil_img.convert('RGBA'), colored.convert('RGBA'), alpha=0.4)
    blended.convert('RGB').save(out_path)


def main():
    if len(sys.argv) < 2:
        print('Usage: gradcam_tta.py <filename in dataset/test>')
        return
    fname = sys.argv[1]
    test_path = Path('dataset') / 'test' / fname
    if not test_path.exists():
        print('File not found:', test_path)
        return

    model = tf.keras.models.load_model('saved_model/model.keras', compile=False)

    pil_img, arr = load_image(test_path)
    prob = predict_prob(model, arr)
    print(f'Original prob_not_fall: {prob:.4f}')

    probs = tta_predictions(model, pil_img)
    print('TTA probs (original, hflip, +15, -15):', ['{:.4f}'.format(p) for p in probs])
    print('TTA mean prob_not_fall = {:.4f}'.format(float(np.mean(probs))))

    # Grad-CAM
    try:
        heatmap = make_gradcam(model, arr)
        out = Path('saved_model') / f'gradcam_{test_path.name}.png'
        save_gradcam_on_image(pil_img, heatmap, out)
        print('Saved Grad-CAM to', out)
    except Exception as e:
        print('Grad-CAM failed:', e)
        # fallback: occlusion sensitivity
        try:
            print('試行: オクルージョン感度マップを作成します（フォールバック）')
            heatmap = occlusion_sensitivity(model, pil_img)
            out = Path('saved_model') / f'occlusion_{test_path.name}.png'
            save_gradcam_on_image(pil_img, heatmap, out)
            print('Saved occlusion map to', out)
        except Exception as e2:
            print('Occlusion fallback も失敗しました:', e2)

def occlusion_sensitivity(model, pil_img, img_size=(224,224), patch_size=28, stride=14):
    # compute drop in prob_not_fall when occluding patches
    import numpy as np
    from PIL import Image

    base = np.array(pil_img.resize(img_size)).astype('float32')
    base_pre = preprocess_input(base)
    base_prob = float(model.predict(np.expand_dims(base_pre,0), verbose=0)[0,0])
    H, W = img_size
    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            y2 = min(H, y1 + patch_size)
            x2 = min(W, x1 + patch_size)
            img_occl = base.copy()
            img_occl[y1:y2, x1:x2, :] = img_occl.mean(axis=(0,1))
            img_occl_pre = preprocess_input(img_occl)
            prob = float(model.predict(np.expand_dims(img_occl_pre,0), verbose=0)[0,0])
            drop = base_prob - prob
            heat[y1:y2, x1:x2] += drop
            counts[y1:y2, x1:x2] += 1

    counts = np.maximum(counts, 1)
    heat = heat / counts
    # normalize
    heat = np.maximum(heat, 0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


if __name__ == '__main__':
    main()
