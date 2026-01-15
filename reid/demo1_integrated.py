"""
Re-ID 整合暗船偵測系統 Demo

結合 YOLO 偵測 + AIS 比對 + Re-ID 識別的完整流程。
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# 如果有安裝 ultralytics，可以使用 YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("提示：未安裝 ultralytics，YOLO 功能不可用")


# === CLIP 模型 ===
print("載入 CLIP 模型...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP 模型載入完成")


# === Re-ID 函式 ===
def get_embedding(image) -> np.ndarray:
    """提取圖片的特徵向量（支援 PIL Image 或檔案路徑）"""
    if isinstance(image, str):
        image = Image.open(image)

    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """計算餘弦相似度"""
    return float(np.dot(a, b))


# === 資料庫 ===
# 船舶 Re-ID 資料庫
ship_reid_db = {}

# AIS 假資料（模擬）
ais_database = {
    "ship_001": {"name": "FISHING STAR", "type": "Fishing Boat"},
    "ship_002": {"name": "OCEAN TRADER", "type": "Container Ship"},
    "ship_003": {"name": "ISLAND FERRY", "type": "Ferry"},
}


def register_ship_reid(name: str, image_path: str):
    """註冊船舶到 Re-ID 資料庫"""
    embedding = get_embedding(image_path)
    ship_reid_db[name] = embedding
    print(f"[Re-ID] 已註冊：{name}")


def match_ais(ship_type: str) -> dict | None:
    """比對 AIS 資料庫"""
    for ship_id, info in ais_database.items():
        if info["type"] == ship_type:
            return {"ship_id": ship_id, **info}
    return None


def identify_ship_reid(image, threshold: float = 0.85) -> tuple[str | None, float]:
    """Re-ID 識別船舶"""
    if not ship_reid_db:
        return None, 0.0

    query_emb = get_embedding(image)

    best_match = None
    best_score = 0.0

    for name, emb in ship_reid_db.items():
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score


def crop_image(image_path: str, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """裁切圖片區域"""
    image = Image.open(image_path)
    return image.crop((x1, y1, x2, y2))


# === 整合偵測流程 ===
def detect_and_identify(image_path: str, yolo_model_path: str = None):
    """
    完整的偵測與識別流程：
    1. YOLO 偵測船舶
    2. AIS 比對
    3. 若 AIS 無資料，使用 Re-ID 比對
    """
    if not YOLO_AVAILABLE:
        print("錯誤：需要安裝 ultralytics 套件")
        return

    if yolo_model_path is None:
        print("錯誤：請提供 YOLO 模型路徑")
        return

    # 1. YOLO 偵測
    print(f"\n[偵測] 分析圖片：{image_path}")
    yolo = YOLO(yolo_model_path)
    results = yolo(image_path)

    for r in results:
        for box in r.boxes:
            # 取得類別
            ship_type = r.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            print(f"[YOLO] 偵測到：{ship_type}（信心值：{confidence:.2f}）")

            # 2. AIS 比對
            ais_match = match_ais(ship_type)

            if ais_match:
                print(f"[AIS] 已知船舶：{ais_match['name']}")
            else:
                print(f"[AIS] 查無 {ship_type} 的 AIS 紀錄")

                # 3. Re-ID 比對
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ship_crop = crop_image(image_path, x1, y1, x2, y2)

                reid_match, reid_score = identify_ship_reid(ship_crop)

                if reid_match:
                    print(f"[Re-ID] 識別為：{reid_match}（相似度：{reid_score:.2f}）")
                    print(f"[警告] {reid_match} 的 AIS 未開啟！")
                else:
                    print(f"[Re-ID] 未知船舶（最高相似度：{reid_score:.2f}）")
                    print(f"[告警] 🚨 暗船偵測！類型：{ship_type}")


# === 使用範例 ===
if __name__ == "__main__":
    print("\n=== Re-ID 整合暗船偵測系統 Demo ===\n")

    print("使用步驟：")
    print("1. 先註冊已知船舶：")
    print("   register_ship_reid('來發號', 'ships/laifa.jpg')")
    print()
    print("2. 執行偵測與識別：")
    print("   detect_and_identify('test.jpg', 'best.pt')")
    print()
    print("流程：YOLO 偵測 → AIS 比對 → Re-ID 識別 → 告警")
