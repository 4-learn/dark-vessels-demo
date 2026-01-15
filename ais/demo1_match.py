"""
偵測結果 × AIS 比對 Demo

示範如何將 YOLO 偵測結果與 AIS 資料比對。
"""

from ais_database import ais_database, get_ships_by_type


def match_detection_with_ais(detection: dict) -> dict | None:
    """
    將 YOLO 偵測結果與 AIS 資料比對

    Args:
        detection: YOLO 偵測結果 {"class": ..., "confidence": ..., "bbox": ...}

    Returns:
        匹配到的 AIS 資料，或 None（暗船）
    """
    detected_type = detection["class"]

    # 在 AIS 資料庫中尋找同類型的船
    matches = get_ships_by_type(detected_type)

    if matches:
        # 簡化：回傳第一個匹配的船
        return {
            "ship_id": matches[0]["ship_id"],
            "ais_info": matches[0],
            "matched_by": "type"
        }

    # 找不到 → 暗船
    return None


def process_detections(detections: list[dict]) -> dict:
    """
    處理所有偵測結果，分類為已知/暗船
    """
    result = {
        "known_ships": [],
        "dark_ships": []
    }

    for det in detections:
        match = match_detection_with_ais(det)

        if match:
            result["known_ships"].append({
                "detection": det,
                "ais": match["ais_info"]
            })
        else:
            result["dark_ships"].append(det)

    return result


if __name__ == "__main__":
    print("=== 偵測結果 × AIS 比對 Demo ===\n")

    # 模擬 YOLO 偵測結果
    detections = [
        {
            "class": "Ore Carrier",
            "confidence": 0.89,
            "bbox": {"x1": 317.8, "y1": 281.3, "x2": 450.7, "y2": 315.3}
        },
        {
            "class": "Fishing Boat",
            "confidence": 0.87,
            "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 200}
        },
        {
            "class": "Small Boat",
            "confidence": 0.72,
            "bbox": {"x1": 400, "y1": 100, "x2": 550, "y2": 250}
        }
    ]

    print("[輸入] YOLO 偵測結果：")
    for det in detections:
        print(f"  - {det['class']} (信心值: {det['confidence']})")
    print()

    # 逐一比對
    print("[比對過程]")
    for det in detections:
        match = match_detection_with_ais(det)

        if match:
            print(f"  ✓ {det['class']} → 已知船舶：{match['ais_info']['name']}")
        else:
            print(f"  ✗ {det['class']} → 查無 AIS 紀錄（暗船）")
    print()

    # 處理結果
    result = process_detections(detections)

    print("[結果統計]")
    print(f"  已知船舶：{len(result['known_ships'])} 艘")
    print(f"  暗船：{len(result['dark_ships'])} 艘")
    print()

    if result["dark_ships"]:
        print("[暗船清單]")
        for ship in result["dark_ships"]:
            print(f"  ⚠️  {ship['class']} (信心值: {ship['confidence']})")
