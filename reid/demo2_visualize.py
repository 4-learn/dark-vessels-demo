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

    plt.legend(loc='upper right', title='Ship')
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


def visualize_similarity_matrix(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Cosine Similarity Matrix"
):
    """
    繪製相似度矩陣熱力圖

    Args:
        embeddings: (n_samples, 512) 的 Embedding 矩陣
        labels: 每張圖的標籤
        title: 圖表標題
    """
    n_samples = len(embeddings)

    # 計算 cosine similarity 矩陣
    # 因為已經正規化，直接用 dot product
    similarity_matrix = np.dot(embeddings, embeddings.T)

    # 繪製熱力圖
    plt.figure(figsize=(10, 8))

    im = plt.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')

    # 設定軸標籤
    # 縮短標籤以便顯示
    short_labels = [f"{i}:{l[:8]}" for i, l in enumerate(labels)]
    plt.xticks(range(n_samples), short_labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(n_samples), short_labels, fontsize=8)

    # 在格子中顯示數值
    for i in range(n_samples):
        for j in range(n_samples):
            value = similarity_matrix[i, j]
            # 根據數值選擇文字顏色
            color = 'white' if value < 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()

    # 儲存
    output_path = "similarity_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[儲存] 相似度矩陣：{output_path}")

    plt.show()

    # 印出文字版摘要
    print_similarity_summary(similarity_matrix, labels)


def print_similarity_summary(similarity_matrix: np.ndarray, labels: list[str]):
    """印出相似度摘要"""
    print()
    print("=" * 50)
    print("Similarity Summary (512-dim cosine similarity)")
    print("=" * 50)

    # 找出每個樣本最相似的（排除自己）
    n = len(labels)
    for i in range(n):
        # 複製一份，把自己設成 -1
        sims = similarity_matrix[i].copy()
        sims[i] = -1
        best_j = np.argmax(sims)
        best_sim = sims[best_j]

        print(f"[{i}] {labels[i][:15]:15} -> Best match: [{best_j}] {labels[best_j][:15]:15} (sim={best_sim:.3f})")

    # 群聚內 vs 群聚間的平均相似度
    print()
    print("-" * 50)
    unique_labels = list(set(labels))

    if len(unique_labels) > 1:
        intra_sims = []  # 同群聚
        inter_sims = []  # 不同群聚

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    intra_sims.append(similarity_matrix[i, j])
                else:
                    inter_sims.append(similarity_matrix[i, j])

        if intra_sims:
            print(f"Intra-cluster avg similarity: {np.mean(intra_sims):.3f} (same ship)")
        if inter_sims:
            print(f"Inter-cluster avg similarity: {np.mean(inter_sims):.3f} (different ships)")

        if intra_sims and inter_sims:
            gap = np.mean(intra_sims) - np.mean(inter_sims)
            print(f"Gap: {gap:.3f} (larger = better separation)")


def normalize(vectors: np.ndarray) -> np.ndarray:
    """正規化向量（使 cosine similarity 在 -1 到 1 之間）"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def create_demo_data():
    """
    建立示範用的假 Embedding 資料
    （當沒有實際圖片時使用）
    """
    print("[示範] 使用模擬資料展示視覺化效果")

    np.random.seed(42)

    # 模擬 3 種船舶類型，每種 4 張照片
    # 每艘船有一個「中心點」，加上一些隨機噪音
    ships = {
        "FISHING STAR": np.random.randn(4, 512) * 0.1 + np.array([1, 0] + [0]*510),
        "OCEAN TRADER": np.random.randn(4, 512) * 0.1 + np.array([-1, 1] + [0]*510),
        "ISLAND FERRY": np.random.randn(4, 512) * 0.1 + np.array([0, -1] + [0]*510),
    }

    # 加入一個「新偵測」（靠近 FISHING STAR）
    new_detection = np.random.randn(1, 512) * 0.1 + np.array([0.9, 0.1] + [0]*510)

    embeddings = []
    labels = []

    for ship_name, emb in ships.items():
        # 正規化每個向量
        emb_normalized = normalize(emb)
        embeddings.extend(emb_normalized)
        labels.extend([ship_name] * len(emb))

    # 新偵測（也要正規化）
    new_normalized = normalize(new_detection)
    embeddings.extend(new_normalized)
    labels.append("? New Detection")

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

    # 視覺化 1: t-SNE 2D 散點圖
    visualize_embeddings(
        embeddings,
        labels,
        title="Ship Embedding 2D Visualization (t-SNE)"
    )

    # 視覺化 2: 相似度矩陣（真正用於判斷的數值）
    visualize_similarity_matrix(
        embeddings,
        labels,
        title="Cosine Similarity Matrix (512-dim)"
    )

    print()
    print("[說明]")
    print("  - t-SNE 圖：視覺化群聚效果（2D 投影，形狀會變形）")
    print("  - 相似度矩陣：實際比對用的數值（512 維，準確）")
    print("  - 同船相似度高（綠色），不同船相似度低（紅色）")


if __name__ == "__main__":
    main()
