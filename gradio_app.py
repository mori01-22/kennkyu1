#!/usr/bin/env python
"""
転倒判定Webアプリ（複数モデル対応版）

使い方:
  1. python gradio_app.py を実行
  2. ブラウザが開いて、モデルを選択できます
  3. 画像をアップロードして判定

モデル構成:
  - saved_model/model.keras（デフォルト）
  - saved_model_dataset1/model.keras
  - saved_model_dataset2/model.keras
  - saved_model_dataset3/model.keras
  など複数選択可能
"""

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# グローバル変数
current_model = None
current_model_path = None
IMG_SIZE = 224
THRESHOLD = 0.5
CLASS_NAMES = ['fall', 'not_fall']

# データセット名 → 表示ラベルのマッピング
# saved_model_dataset1 → 黒色のタンス 等として表示する
LABEL_MAP = {
    'dataset1': '黒色のタンス',
    'dataset2': '白色のタンス',
    'dataset3': '木目調のタンス',
}

# 利用可能なモデル一覧を検索
def find_available_models():
    """saved_model* フォルダからモデルを検索"""
    models = {}
    base_dir = Path('.')
    
    # デフォルトモデル（saved_model/model.keras）は今回は非表示にする
    
    # saved_model_* パターンをチェック
    for model_dir in sorted(base_dir.glob('saved_model_*')):
        if model_dir.is_dir():
            model_file = model_dir / 'model.keras'
            if model_file.exists():
                dataset_name = model_dir.name.replace('saved_model_', '')
                # ユーザーに分かりやすい表示名へ変換
                display_name = LABEL_MAP.get(dataset_name, dataset_name)
                models[display_name] = str(model_file)
    
    return models


# モデル一覧を取得
available_models = find_available_models()

if not available_models:
    print("✗ モデルが見つかりません。")
    print("以下のいずれかを実行してください:")
    print("  - python train_and_infer.py --epochs 5")
    print("  - python train_multiple_datasets.py --epochs 5")
    exit(1)

print(f"✓ 利用可能なモデル: {list(available_models.keys())}")


def load_model_by_path(model_path):
    """指定されたパスからモデルを読み込む"""
    global current_model, current_model_path
    try:
        current_model = keras.models.load_model(model_path)
        current_model_path = model_path
        print(f"✓ モデル読み込み完了: {model_path}")
        return True
    except Exception as e:
        print(f"✗ モデル読み込み失敗: {e}")
        return False


def on_model_change(model_name):
    """モデル選択時のコールバック"""
    model_path = available_models[model_name]
    success = load_model_by_path(model_path)
    
    if success:
        return f"✅ モデルを切り替えました: **{model_name}**"
    else:
        return f"❌ モデルの読み込みに失敗しました"


def predict_image(image, model_name):
    """
    画像を受け取って転倒判定を行う関数
    
    Args:
        image: PIL Image または numpy array
        model_name: 選択されたモデル名
    
    Returns:
        str: 判定結果のテキスト
    """
    if image is None:
        return "画像をアップロードしてください。"
    
    if current_model is None:
        return "モデルが読み込まれていません。モデルを選択してください。"
    
    try:
        # PIL Image に変換
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # リサイズと前処理
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 予測実行
        prediction = current_model.predict(img_array, verbose=0)
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

**使用モデル:** {model_name}

---
*閾値: {THRESHOLD} | 生確率値: {probability:.4f}*
        """
        
        return result
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


# 初期モデルを読み込む
initial_model_name = list(available_models.keys())[0]
load_model_by_path(available_models[initial_model_name])

# Gradio インターフェース構築
with gr.Blocks(title="転倒判定システム", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 転倒判定システム
        
        画像をアップロードして、転倒（fall）か非転倒（not_fall）かを判定します。
        
        複数のモデルから選択して比較することもできます。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # モデル選択
            model_selector = gr.Dropdown(
                choices=list(available_models.keys()),
                value=initial_model_name,
                label="📊 モデルを選択",
                interactive=True
            )
            model_status = gr.Markdown(f"✅ 現在のモデル: **{initial_model_name}**")
            
            # 画像入力
            image_input = gr.Image(
                label="画像をアップロード",
                type="pil",
                height=400
            )
            predict_btn = gr.Button("🔍 判定する", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### 説明
                - JPG, PNG形式の画像に対応
                - 画像は自動的に224x224にリサイズされます
                - モデルを切り替えて比較できます
                """
            )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="判定結果")
    
    # モデル選択時のイベント
    model_selector.change(
        fn=on_model_change,
        inputs=model_selector,
        outputs=model_status
    )
    
    # ボタンクリック時の動作
    predict_btn.click(
        fn=predict_image,
        inputs=[image_input, model_selector],
        outputs=output_text
    )
    
    # 画像変更時に自動判定
    image_input.change(
        fn=predict_image,
        inputs=[image_input, model_selector],
        outputs=output_text
    )
    
    gr.Markdown(
        f"""
        ---
        **モデル情報:** mobilenetV2ベースの転移学習モデル  
        **利用可能なモデル数:** {len(available_models)}個  
        **閾値:** {THRESHOLD}
        """
    )

# アプリ起動
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Webアプリを起動します...")
    print(f"利用可能なモデル: {len(available_models)}個")
    print("="*50 + "\n")
    
    demo.launch(
        share=True,  # 一時的な公開URLを生成（72時間有効）
        server_port=7860,
        show_error=True
    )
