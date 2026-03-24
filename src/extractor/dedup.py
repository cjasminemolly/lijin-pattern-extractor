"""
重复照片智能清理模块
使用感知哈希 + CLIP 特征向量双重检测重复黎锦图片
"""
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity


class LijinDeduplicator:
    """黎锦图片去重器 - 双重算法确保精准去重"""

    def __init__(self, hash_threshold: int = 8, similarity_threshold: float = 0.92):
        """
        Args:
            hash_threshold: 感知哈希汉明距离阈值（越小越严格）
            similarity_threshold: CLIP 余弦相似度阈值（越大越严格）
        """
        self.hash_threshold = hash_threshold
        self.similarity_threshold = similarity_threshold
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self):
        """懒加载 CLIP 模型"""
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            print("Loading CLIP model for feature extraction...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self._clip_model, self._clip_processor

    def compute_phash(self, image_path: str) -> imagehash.ImageHash:
        """计算感知哈希"""
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)

    def compute_clip_features(self, image_paths: List[str]) -> np.ndarray:
        """批量计算 CLIP 图像特征向量"""
        import torch
        model, processor = self._load_clip()
        features = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
            features.append(feat.squeeze().numpy())
        return np.array(features)

    def find_duplicates(self, image_dir: str) -> List[List[str]]:
        """
        在目录中查找重复图片组

        Returns:
            重复组列表，每组包含相似图片路径
        """
        image_paths = [
            str(p) for p in Path(image_dir).glob("**/*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        if len(image_paths) < 2:
            return []

        print(f"Scanning {len(image_paths)} images for duplicates...")

        # Step 1: 感知哈希粗筛
        hashes = {path: self.compute_phash(path) for path in image_paths}
        candidate_pairs = []
        paths = list(hashes.keys())
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                dist = hashes[paths[i]] - hashes[paths[j]]
                if dist <= self.hash_threshold:
                    candidate_pairs.append((paths[i], paths[j]))

        if not candidate_pairs:
            return []

        # Step 2: CLIP 精确验证
        candidate_set = list(set(p for pair in candidate_pairs for p in pair))
        features = self.compute_clip_features(candidate_set)
        feat_map = {path: feat for path, feat in zip(candidate_set, features)}

        duplicate_groups = []
        visited = set()
        for p1, p2 in candidate_pairs:
            sim = cosine_similarity(
                feat_map[p1].reshape(1, -1),
                feat_map[p2].reshape(1, -1)
            )[0][0]
            if sim >= self.similarity_threshold:
                # 合并到已有组或新建组
                merged = False
                for group in duplicate_groups:
                    if p1 in group or p2 in group:
                        group.update([p1, p2])
                        merged = True
                        break
                if not merged:
                    duplicate_groups.append({p1, p2})

        return [list(g) for g in duplicate_groups]

    def clean_duplicates(self, image_dir: str, keep_strategy: str = "largest") -> dict:
        """
        自动清理重复图片，保留最优质的一张

        Args:
            keep_strategy: "largest"(保留最大文件) | "newest"(保留最新) | "oldest"(保留最旧)

        Returns:
            清理报告
        """
        groups = self.find_duplicates(image_dir)
        removed = []
        kept = []

        for group in groups:
            if keep_strategy == "largest":
                best = max(group, key=lambda p: os.path.getsize(p))
            elif keep_strategy == "newest":
                best = max(group, key=lambda p: os.path.getmtime(p))
            else:
                best = min(group, key=lambda p: os.path.getmtime(p))

            kept.append(best)
            for path in group:
                if path != best:
                    os.remove(path)
                    removed.append(path)

        return {
            "total_groups": len(groups),
            "removed_count": len(removed),
            "kept_count": len(kept),
            "removed_files": removed,
            "kept_files": kept,
        }
