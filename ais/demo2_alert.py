"""
不明船舶告警觸發 Demo

示範完整流程：偵測 → 比對 → 告警。
"""

from datetime import datetime
import json
from ais_database import get_ships_by_type


def create_dark_ship_alert(detection: dict) -> dict:
    """
    建立暗船告警事件
    """
    return {
        "event_type": "dark_ship_detected",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "severity": "HIGH",
        "detection": {
            "class": detection["class"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"]
        },
        "message": f"偵測到不明船舶：{detection['class']}，信心值：{detection['confidence']:.2f}"
    }


def create_known_ship_log(detection: dict, ais_info: dict) -> dict:
    """
    建立已知船舶紀錄
    """
    return {
        "event_type": "known_ship_detected",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "severity": "INFO",
        "detection": {
            "class": detection["class"],
            "confidence": detection["confidence"]
        },
        "ais": {
            "name": ais_info["name"],
            "mmsi": ais_info["mmsi"],
            "flag": ais_info["flag"]
        },
        "message": f"已知船舶：{ais_info['name']} ({detection['class']})"
    }


def process_and_alert(detections: list[dict]) -> dict:
    """
    處理偵測結果並產生告警

    Returns:
        {"alerts": [...], "logs": [...]}
    """
    alerts = []
    logs = []

    for det in detections:
        # AIS 比對
        matches = get_ships_by_type(det["class"])

        if matches:
            # 已知船舶 → 紀錄
            log = create_known_ship_log(det, matches[0])
            logs.append(log)
            print(f"[正常] {log['message']}")
        else:
            # 暗船 → 告警
            alert = create_dark_ship_alert(det)
            alerts.append(alert)
            print(f"[告警] {alert['message']}")

    return {"alerts": alerts, "logs": logs}


if __name__ == "__main__":
    print("=== 不明船舶告警觸發 Demo ===\n")

    # 模擬 YOLO 偵測結果
    detections = [
        {
            "class": "Ore Carrier",
            "confidence": 0.89,
            "bbox": {"x1": 317.8, "y1": 281.3, "x2": 450.7, "y2": 315.3}
        },
        {
            "class": "Small Boat",
            "confidence": 0.72,
            "bbox": {"x1": 400, "y1": 100, "x2": 550, "y2": 250}
        },
        {
            "class": "Sailboat",
            "confidence": 0.65,
            "bbox": {"x1": 200, "y1": 150, "x2": 350, "y2": 280}
        }
    ]

    print("[處理中]")
    result = process_and_alert(detections)
    print()

    # 統計
    print(f"[統計] 已知船舶：{len(result['logs'])} 艘，暗船告警：{len(result['alerts'])} 筆\n")

    # 輸出告警 JSON
    if result["alerts"]:
        print("[告警 JSON]")
        print(json.dumps(result["alerts"], indent=2, ensure_ascii=False))
