#!/usr/bin/env python
"""
転倒判定Webアプリ（Gradio版）

使い方:
  1. pip install gradio tensorflow をインストール
  2. python gradio_app.py を実行
  3. ブラウザが自動で開きます（または http://127.0.0.1:7860 にアクセス）
  4. 画像をドラッグ&ドロップまたは選択して判定

学習済みモデル (saved_model/model.keras) を使用します。
"""

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# モデル読み込み
MODEL_PATH = 'saved_model/model.keras'
IMG_SIZE = 224
THRESHOLD = 0.5

print("モデルを読み込んでいます...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ モデルの読み込み完了: {MODEL_PATH}")
except Exception as e:
    print(f"✗ モデルの読み込みに失敗しました: {e}")
    print("先に train_and_infer.py を実行してモデルを作成してください。")
    exit(1)

# クラス名（通常は ['fall', 'not_fall'] の順）
CLASS_NAMES = ['fall', 'not_fall']


def predict_image(image):
    """
    画像を受け取って転倒判定を行う関数
    
    Args:
        image: PIL Image または numpy array
    
    Returns:
        str: 判定結果のテキスト
    """
    if image is None:
        return "画像をアップロードしてください。"
    
    try:
        # PIL Image に変換（Gradioから渡される形式に対応）
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # リサイズと前処理
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 予測実行
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
        
        # 判定結果
        label_idx = 1 if probability >= THRESHOLD else 0
        label = CLASS_NAMES[label_idx]
        
        # 結果メッセージの組み立て
        if label == 'fall':
            emoji = "⚠️"
            message = "転倒の可能性があります。"
            confidence = f"{(1 - probability) * 100:.1f}%"
        else:
            emoji = "✅"
            message = "転倒の可能性は低いです。"
            confidence = f"{probability * 100:.1f}%"
        
        result = f"""
## {emoji} 判定結果: {label.upper()}

**確率:** {confidence}

**メッセージ:** {message}

---
*閾値: {THRESHOLD} | 生確率値: {probability:.4f}*
        """
        
        return result
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


# Gradio インターフェース構築
with gr.Blocks(title="転倒判定システム", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 転倒判定システム
        
        画像をアップロードして、転倒（fall）か非転倒（not_fall）かを判定します。
        
        **使い方:** 下のエリアに画像をドラッグ&ドロップ、またはクリックしてファイルを選択してください。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="画像をアップロード",
                type="pil",
                height=400
            )
            predict_btn = gr.Button("🔍 判定する", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### 💡 ヒント
                - JPG, PNG形式の画像に対応
                - 画像は自動的に224x224にリサイズされます
                - 複数の画像を試すこともできます
                """
            )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="判定結果")
    
    # ボタンクリック時の動作
    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output_text
    )
    
    # 画像変更時に自動判定（オプション）
    image_input.change(
        fn=predict_image,
        inputs=image_input,
        outputs=output_text
    )
    
    gr.Markdown(
        """
        ---
        **モデル情報:** MobileNetV2ベースの転移学習モデル  
        **閾値:** 0.5 (カスタマイズ可能)
        """
    )

# アプリ起動
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Webアプリを起動します...")
    print("="*50 + "\n")
    
    demo.launch(
        share=False,  # True にするとインターネット経由でアクセス可能なURLが生成されます
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
