"""
Re-ID Embedding 視覺化 Demo

用 t-SNE 將高維 Embedding 降到 2D，觀察船舶照片的群聚效果。
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# 檢查依賴
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from sklearn.manifold import TSNE
except ImportError as e:
    print(f"請先安裝依賴：pip install torch transformers scikit-learn matplotlib")
    print(f"缺少模組：{e}")
    exit(1)


def load_clip_model():
    """載入 CLIP 模型"""
    print("[載入] CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def extract_embedding(image_path: str, model, processor) -> np.ndarray:
    """提取單張圖片的 Embedding"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # 正規化
    embedding = features / features.norm(dim=-1, keepdim=True)
    return embedding.numpy().flatten()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: list[str],
    image_paths: list[str] = None,
    title: str = "Ship Embedding Visualization"
):
    """
    用 t-SNE 將 Embedding 降維並視覺化

    Args:
        embeddings: (n_samples, 512) 的 Embedding 矩陣
        labels: 每張圖的標籤（船名或類型）
        image_paths: 圖片路徑（可選，用於顯示縮圖）
        title: 圖表標題
    """
    n_samples = len(embeddings)

    if n_samples < 2:
        print("[錯誤] 至少需要 2 張圖片才能視覺化")
        return

    # t-SNE 降維
    print(f"[處理] t-SNE 降維中（{n_samples} 張圖片）...")

    # perplexity 必須小於樣本數
    perplexity = min(30, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # 設定顏色
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 繪圖
    plt.figure(figsize=(12, 8))

    for i, (x, y) in enumerate(embeddings_2d):
        color = label_to_color[labels[i]]
        plt.scatter(x, y, c=[color], s=200, alpha=0.7, edgecolors='white', linewidth=2)
        plt.annotate(
            labels[i],
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    # 圖例
    for label, color in label_to_color.items():
        plt.scatter([], [], c=[color], s=100, label=label)

    plt.legend(loc='upper right', title='船舶類型')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 儲存
    output_path = "embedding_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[儲存] 視覺化結果：{output_path}")

    plt.show()


def create_demo_data():
    """
    建立示範用的假 Embedding 資料
    （當沒有實際圖片時使用）
    """
    print("[示範] 使用模擬資料展示視覺化效果")

    np.random.seed(42)

    # 模擬 3 種船舶類型，每種 4 張照片
    ships = {
        "FISHING STAR": np.random.randn(4, 512) * 0.1 + np.array([1, 0] + [0]*510),
        "OCEAN TRADER": np.random.randn(4, 512) * 0.1 + np.array([-1, 1] + [0]*510),
        "ISLAND FERRY": np.random.randn(4, 512) * 0.1 + np.array([0, -1] + [0]*510),
    }

    # 加入一個「新偵測」
    new_detection = np.random.randn(1, 512) * 0.1 + np.array([0.9, 0.1] + [0]*510)

    embeddings = []
    labels = []

    for ship_name, emb in ships.items():
        embeddings.extend(emb)
        labels.extend([ship_name] * len(emb))

    # 新偵測
    embeddings.extend(new_detection)
    labels.append("❓ 新偵測")

    return np.array(embeddings), labels


def main():
    """主程式"""
    print("=" * 50)
    print("Re-ID Embedding 視覺化 Demo")
    print("=" * 50)
    print()

    # 尋找圖片
    image_dir = Path(".")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        f for f in image_dir.glob("*")
        if f.suffix.lower() in image_extensions
    ]

    if len(image_files) >= 3:
        # 使用實際圖片
        print(f"[找到] {len(image_files)} 張圖片")

        model, processor = load_clip_model()

        embeddings = []
        labels = []

        for img_path in image_files:
            print(f"[處理] {img_path.name}")
            emb = extract_embedding(str(img_path), model, processor)
            embeddings.append(emb)
            # 用檔名作為標籤
            labels.append(img_path.stem[:15])

        embeddings = np.array(embeddings)

    else:
        # 使用模擬資料
        print(f"[提示] 圖片不足（找到 {len(image_files)} 張），使用模擬資料")
        print("[提示] 若要使用實際圖片，請在當前目錄放置 3 張以上船舶照片")
        print()
        embeddings, labels = create_demo_data()

    # 視覺化
    visualize_embeddings(
        embeddings,
        labels,
        title="Ship Embedding 2D Visualization (t-SNE)"
    )

    print()
    print("[說明]")
    print("  - 同一艘船的不同照片會群聚在一起")
    print("  - 不同船舶會分散在不同區域")
    print("  - 新偵測會落在最相似的船舶附近")


if __name__ == "__main__":
    main()
