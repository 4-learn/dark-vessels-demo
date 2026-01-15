"""
船舶偵測推論 Demo

使用訓練好的模型對圖片進行推論，輸出結構化資料。
"""

from ultralytics import YOLO


def detect_ships(image_path: str, model_path: str = "best.pt") -> list[dict]:
    """
    對圖片進行船舶偵測

    Args:
        image_path: 圖片路徑
        model_path: 模型路徑

    Returns:
        偵測結果列表
    """
    model = YOLO(model_path)
    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            det = {
                "class": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "bbox": {
                    "x1": round(box.xyxy[0][0].item(), 1),
                    "y1": round(box.xyxy[0][1].item(), 1),
                    "x2": round(box.xyxy[0][2].item(), 1),
                    "y2": round(box.xyxy[0][3].item(), 1)
                }
            }
            detections.append(det)

    return detections


if __name__ == "__main__":
    # 執行偵測
    ships = detect_ships("ship_photo.jpg")

    # 輸出結果
    print("=== 船舶偵測結果 ===\n")

    for i, ship in enumerate(ships, 1):
        print(f"[{i}] {ship['class']}")
        print(f"    信心值：{ship['confidence']}")
        print(f"    位置：({ship['bbox']['x1']}, {ship['bbox']['y1']}) - ({ship['bbox']['x2']}, {ship['bbox']['y2']})")
        print()

    # 輸出 JSON 格式
    import json
    print("=== JSON 格式 ===\n")
    print(json.dumps(ships, indent=2, ensure_ascii=False))
