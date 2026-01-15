from ultralytics import YOLO

# 載入預訓練模型（Inference only）
model = YOLO("yolov5s.pt")

# 直接對影像檔案進行推論
results = model("image.jpg")

# 儲存偵測結果圖片（Server 環境適用）
results[0].save(filename="result.jpg")

# 取得結構化輸出（為 Event JSON 做準備）
boxes = results[0].boxes

for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    print({
        "class_id": cls_id,
        "confidence": conf,
        "bbox": [x1, y1, x2, y2]
    })
