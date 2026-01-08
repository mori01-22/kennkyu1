import os
import random
from pathlib import Path

def reduce_dataset(source_dir, target_count=200):
    """データセットを指定枚数に削減"""
    # 既存の画像ファイルを取得
    image_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    current_count = len(image_files)
    print(f"{source_dir}: 現在 {current_count} 枚の画像があります")
    
    if current_count <= target_count:
        print(f"すでに {target_count} 枚以下の画像です。削除は不要です。")
        return current_count
    
    # 削除する必要がある枚数
    to_delete = current_count - target_count
    print(f"{to_delete} 枚の画像を削除します")
    
    # 拡張画像（_aug_を含む）と元画像を分ける
    augmented_images = [f for f in image_files if '_aug_' in f]
    original_images = [f for f in image_files if '_aug_' not in f]
    
    print(f"  元画像: {len(original_images)} 枚")
    print(f"  拡張画像: {len(augmented_images)} 枚")
    
    # 削除リストを作成（拡張画像を優先的に削除）
    files_to_delete = []
    
    # まず拡張画像から削除
    if len(augmented_images) >= to_delete:
        # 拡張画像だけで十分削除できる場合
        files_to_delete = random.sample(augmented_images, to_delete)
    else:
        # 拡張画像を全て削除し、さらに元画像からも削除
        files_to_delete = augmented_images.copy()
        remaining_to_delete = to_delete - len(augmented_images)
        files_to_delete.extend(random.sample(original_images, remaining_to_delete))
    
    # ファイルを削除
    deleted = 0
    for filename in files_to_delete:
        file_path = os.path.join(source_dir, filename)
        try:
            os.remove(file_path)
            deleted += 1
            if deleted % 50 == 0:
                print(f"  {deleted}/{to_delete} 枚削除完了")
        except Exception as e:
            print(f"エラー: {filename} の削除中にエラーが発生しました: {e}")
    
    final_count = len([f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f"完了: {source_dir} には現在 {final_count} 枚の画像があります")
    return final_count

if __name__ == "__main__":
    # データセットのパス
    fall_dir = r"c:\kennkyu1\dataset\train\fall"
    not_fall_dir = r"c:\kennkyu1\dataset\train\not_fall"
    
    target_count = 200
    
    print("=" * 60)
    print("データセットの削減を開始します")
    print("=" * 60)
    
    # fallフォルダを削減
    print("\n[1/2] fallフォルダの削減")
    fall_count = reduce_dataset(fall_dir, target_count)
    
    # not_fallフォルダを削減
    print("\n[2/2] not_fallフォルダの削減")
    not_fall_count = reduce_dataset(not_fall_dir, target_count)
    
    print("\n" + "=" * 60)
    print("データセット削減が完了しました")
    print(f"fall: {fall_count} 枚")
    print(f"not_fall: {not_fall_count} 枚")
    print("=" * 60)
