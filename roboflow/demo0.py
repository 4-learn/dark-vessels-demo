import os
from dotenv import load_dotenv
from roboflow import Roboflow

# 載入 .env 檔案
load_dotenv()

# 從環境變數取得 API Key
api_key = os.getenv("ROBOFLOW_API_KEY")

# 初始化 Roboflow
rf = Roboflow(api_key=api_key)

# 下載船舶偵測資料集
project = rf.workspace("cnnonboard").project("boat-detection-model")
dataset = project.version(1).download("yolov8")

