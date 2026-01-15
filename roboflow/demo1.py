from ultralytics import YOLO

model = YOLO("best.pt")
results = model("ship_photo.jpg")
results[0].save(filename="result.jpg")

