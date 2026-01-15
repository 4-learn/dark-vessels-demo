"""
暗船偵知系統 - Workshop 解答

完整流程：YOLO 偵測 → AIS 比對 → 告警輸出
"""

import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from ais_database import get_ships_by_type, get_known_types


# 設定
MODEL_PATH = "best.pt"  # 訓練好的模型
CONFIDENCE_THRESHOLD = 0.5  # 信心值門檻


def detect_ships(image_path: str) -> list[dict]:
    """
    使用 YOLO 偵測圖片中的船舶

    Args:
        image_path: 圖片路徑

    Returns:
        偵測結果列表
    """
    model = YOLO(MODEL_PATH)
    results = model(image_path, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            # 過濾低信心值
            if conf < CONFIDENCE_THRESHOLD:
                continue

            detection = {
                "class": r.names[int(box.cls[0])],
                "confidence": conf,
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3])
                }
            }
            detections.append(detection)

    return detections


def match_with_ais(detection: dict) -> dict | None:
    """
    將偵測結果與 AIS 資料比對

    Args:
        detection: YOLO 偵測結果

    Returns:
        匹配到的 AIS 資料，或 None（暗船）
    """
    detected_type = detection["class"]
    matches = get_ships_by_type(detected_type)

    if matches:
        return matches[0]  # 簡化：回傳第一筆

    return None


def create_alert(detection: dict) -> dict:
    """
    建立暗船告警
    """
    return {
        "event_type": "dark_ship_detected",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "severity": "HIGH",
        "class": detection["class"],
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "message": f"偵測到不明船舶：{detection['class']}，信心值：{detection['confidence']:.2f}"
    }


def detect_and_alert(image_path: str) -> list[dict]:
    """
    主流程：偵測 → 比對 → 告警

    Args:
        image_path: 圖片路徑

    Returns:
        暗船告警列表
    """
    print(f"[偵測] 分析圖片：{image_path}")

    # 1. YOLO 偵測
    detections = detect_ships(image_path)

    if not detections:
        print("[偵測] 未偵測到任何船舶")
        return []

    print(f"[偵測] 共偵測到 {len(detections)} 艘船舶")

    # 2. AIS 比對 & 3. 產生告警
    alerts = []

    for det in detections:
        ais_match = match_with_ais(det)

        if ais_match:
            print(f"[比對] {det['class']} → 已知船舶：{ais_match['name']}")
        else:
            print(f"[比對] {det['class']} → 查無 AIS 紀錄")
            alert = create_alert(det)
            alerts.append(alert)

    # 輸出告警
    if alerts:
        print(f"[告警] 暗船偵測！共 {len(alerts)} 筆")
    else:
        print("[結果] 所有船舶均為已知，無告警")

    return alerts


if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("暗船偵知系統 - Workshop 解答")
    print("=" * 50)
    print()

    # 顯示 AIS 已知類型
    known_types = get_known_types()
    print(f"[AIS] 已知船舶類型：{known_types}")
    print()

    # 確認圖片路徑
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 預設測試圖片
        image_path = "test_ship.jpg"

    # 確認模型存在
    if not Path(MODEL_PATH).exists():
        print(f"[錯誤] 找不到模型：{MODEL_PATH}")
        print("[提示] 請先訓練模型或下載 best.pt")
        sys.exit(1)

    # 確認圖片存在
    if not Path(image_path).exists():
        print(f"[錯誤] 找不到圖片：{image_path}")
        sys.exit(1)

    # 執行偵測
    alerts = detect_and_alert(image_path)

    # 輸出 JSON
    if alerts:
        print()
        print("告警內容：")
        print(json.dumps(alerts, indent=2, ensure_ascii=False))
