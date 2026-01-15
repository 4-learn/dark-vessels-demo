"""
Re-ID 船舶識別 Demo

使用 CLIP 提取圖片特徵，透過餘弦相似度比對船舶身份。
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


# 載入 CLIP 模型
print("載入 CLIP 模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("模型載入完成")


def get_embedding(image_path: str) -> np.ndarray:
    """提取圖片的特徵向量"""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # 正規化
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """計算餘弦相似度"""
    return float(np.dot(a, b))


# === 船舶資料庫 ===
ship_db = {}


def register_ship(name: str, image_path: str):
    """註冊船舶到資料庫"""
    embedding = get_embedding(image_path)
    ship_db[name] = embedding
    print(f"已註冊：{name}")


def identify_ship(image_path: str, threshold: float = 0.85) -> str:
    """識別船舶"""
    if not ship_db:
        return "資料庫為空，請先註冊船舶"

    query_emb = get_embedding(image_path)

    best_match = None
    best_score = 0

    for name, emb in ship_db.items():
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return f"識別為：{best_match}（相似度：{best_score:.2f}）"
    else:
        return f"未知船舶（最高相似度：{best_score:.2f}）"


def list_registered_ships():
    """列出已註冊的船舶"""
    if not ship_db:
        print("資料庫為空")
        return

    print("已註冊船舶：")
    for name in ship_db:
        print(f"  - {name}")


# === 使用範例 ===
if __name__ == "__main__":
    print("\n=== Re-ID 船舶識別 Demo ===\n")

    # 範例使用方式（需要自行準備圖片）
    print("使用方式：")
    print("1. 註冊船舶：register_ship('來發號', 'path/to/laifa.jpg')")
    print("2. 識別船舶：identify_ship('path/to/unknown.jpg')")
    print("3. 列出已註冊：list_registered_ships()")
    print("\n請在 Python 互動模式中使用，或修改此檔案加入你的圖片路徑。")
