import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import shutil
from pathlib import Path

def zoom_image(image, zoom_factor):
    """画像をズーム（拡大または縮小）"""
    width, height = image.size
    
    if zoom_factor > 1:
        # ズームイン
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        image = image.crop((left, top, left + new_width, top + new_height))
        image = image.resize((width, height), Image.LANCZOS)
    else:
        # ズームアウト
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        new_image = Image.new('RGB', (width, height), (0, 0, 0))
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        new_image.paste(resized, (left, top))
        image = new_image
    
    return image

def offset_image(image, offset_x, offset_y):
    """画像をオフセット（位置をずらす）"""
    width, height = image.size
    new_image = Image.new('RGB', (width, height), (0, 0, 0))
    new_image.paste(image, (offset_x, offset_y))
    return new_image

def adjust_brightness(image, factor):
    """明るさを調整"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_color(image, hue_shift, saturation_factor, brightness_factor):
    """色調を変換（色相、彩度、明度）"""
    # 彩度を調整
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)
    
    # 明度を調整
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    # 色相シフト（HSVで処理）
    if hue_shift != 0:
        import cv2
        img_array = np.array(image)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
        img_hsv = img_hsv.astype(np.uint8)
        img_array = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        image = Image.fromarray(img_array)
    
    return image

def augment_image(image):
    """ランダムにデータ拡張を適用"""
    # ランダムにどの変換を適用するか決定
    augmentations = []
    
    # ズーム（0.7〜1.3倍）
    if random.random() > 0.5:
        zoom_factor = random.uniform(0.7, 1.3)
        image = zoom_image(image, zoom_factor)
        augmentations.append(f"zoom_{zoom_factor:.2f}")
    
    # オフセット
    if random.random() > 0.5:
        width, height = image.size
        offset_x = random.randint(-int(width * 0.1), int(width * 0.1))
        offset_y = random.randint(-int(height * 0.1), int(height * 0.1))
        image = offset_image(image, offset_x, offset_y)
        augmentations.append(f"offset_{offset_x}_{offset_y}")
    
    # 明るさ変換（0.7〜1.3倍）
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        image = adjust_brightness(image, brightness_factor)
        augmentations.append(f"bright_{brightness_factor:.2f}")
    
    # 色調変換
    if random.random() > 0.5:
        hue_shift = random.randint(-20, 20)
        saturation_factor = random.uniform(0.7, 1.3)
        brightness_factor = random.uniform(0.9, 1.1)
        image = adjust_color(image, hue_shift, saturation_factor, brightness_factor)
        augmentations.append(f"color_{hue_shift}_{saturation_factor:.2f}_{brightness_factor:.2f}")
    
    return image, augmentations

def augment_dataset(source_dir, target_count=500):
    """データセットを拡張"""
    # 既存の画像ファイルを取得
    image_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    current_count = len(image_files)
    print(f"{source_dir}: 現在 {current_count} 枚の画像があります")
    
    if current_count >= target_count:
        print(f"すでに {target_count} 枚以上の画像があります")
        return current_count
    
    # 必要な拡張数
    needed = target_count - current_count
    print(f"{needed} 枚の画像を生成します")
    
    # 拡張画像を生成
    generated = 0
    while generated < needed:
        # ランダムに元画像を選択
        source_image_name = random.choice(image_files)
        source_path = os.path.join(source_dir, source_image_name)
        
        try:
            # 画像を読み込み
            image = Image.open(source_path).convert('RGB')
            
            # データ拡張を適用
            augmented_image, aug_names = augment_image(image)
            
            # 新しいファイル名を生成
            name_without_ext = os.path.splitext(source_image_name)[0]
            ext = os.path.splitext(source_image_name)[1]
            new_name = f"{name_without_ext}_aug_{generated}{ext}"
            new_path = os.path.join(source_dir, new_name)
            
            # 保存
            augmented_image.save(new_path)
            generated += 1
            
            if generated % 50 == 0:
                print(f"  {generated}/{needed} 枚生成完了")
        
        except Exception as e:
            print(f"エラー: {source_image_name} の処理中にエラーが発生しました: {e}")
            continue
    
    final_count = len([f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f"完了: {source_dir} には現在 {final_count} 枚の画像があります")
    return final_count

if __name__ == "__main__":
    # データセットのパス
    fall_dir = r"c:\kennkyu1\dataset\train\fall"
    not_fall_dir = r"c:\kennkyu1\dataset\train\not_fall"
    
    target_count = 500
    
    print("=" * 60)
    print("データ拡張を開始します")
    print("=" * 60)
    
    # fallフォルダを拡張
    print("\n[1/2] fallフォルダの拡張")
    fall_count = augment_dataset(fall_dir, target_count)
    
    # not_fallフォルダを拡張
    print("\n[2/2] not_fallフォルダの拡張")
    not_fall_count = augment_dataset(not_fall_dir, target_count)
    
    print("\n" + "=" * 60)
    print("データ拡張が完了しました")
    print(f"fall: {fall_count} 枚")
    print(f"not_fall: {not_fall_count} 枚")
    print("=" * 60)
