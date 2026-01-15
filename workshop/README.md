# Workshop 解答：暗船偵知系統

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `ais_database.py` | AIS 假資料庫（5 艘船） |
| `dark_ship_detector.py` | 主程式：偵測 → 比對 → 告警 |

## 使用方式

```bash
# 1. 確保有訓練好的模型 best.pt
# 2. 準備測試圖片
# 3. 執行
python dark_ship_detector.py test_ship.jpg
```

## 預期輸出

```
==================================================
暗船偵知系統 - Workshop 解答
==================================================

[AIS] 已知船舶類型：['Fishing Boat', 'Container Ship', 'Ferry', 'Bulk Carrier', 'Ore Carrier']

[偵測] 分析圖片：test_ship.jpg
[偵測] 共偵測到 2 艘船舶
[比對] Fishing Boat → 已知船舶：FISHING STAR
[比對] Small Boat → 查無 AIS 紀錄
[告警] 暗船偵測！共 1 筆

告警內容：
[
  {
    "event_type": "dark_ship_detected",
    "timestamp": "2026-01-15T07:00:00Z",
    "severity": "HIGH",
    "class": "Small Boat",
    "confidence": 0.72,
    ...
  }
]
```

## 延伸挑戰

1. 加入信心值門檻設定（目前為 0.5）
2. 處理多艘暗船情境
3. 加入 GPS 座標比對邏輯
4. 實作告警聚合（相同類型合併）
