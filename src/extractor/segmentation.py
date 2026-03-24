"""
黎锦纹样分割提取模块
基于 SAM (Segment Anything Model) 精准分割纹样区域
"""
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import cv2


class LijinPatternSegmentor:
    """黎锦纹样分割器"""

    def __init__(self, use_sam: bool = True):
        self.use_sam = use_sam
        self._sam_model = None

    def _load_sam(self):
        """懒加载 SAM 模型"""
        if self._sam_model is None:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                print("Loading SAM model...")
                # 使用轻量级 vit_b 版本
                sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b.pth")
                self._sam_model = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.95,
                    min_mask_region_area=500,
                )
            except Exception:
                print("SAM not available, falling back to classical CV")
                self._sam_model = None
        return self._sam_model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：增强对比度，突出纹样"""
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def extract_patterns_classical(self, image: np.ndarray) -> List[Dict]:
        """
        经典 CV 方法提取纹样（SAM 不可用时的备选方案）
        使用颜色聚类 + 轮廓检测
        """
        from sklearn.cluster import KMeans

        # 颜色量化
        pixels = image.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        quantized = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)

        patterns = []
        for i in range(8):
            mask = (labels.reshape(image.shape[:2]) == i).astype(np.uint8) * 255
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # 过滤噪点
                    x, y, w, h = cv2.boundingRect(cnt)
                    pattern_crop = image[y:y+h, x:x+w]
                    color = kmeans.cluster_centers_[i].astype(int).tolist()
                    patterns.append({
                        "bbox": [x, y, w, h],
                        "area": area,
                        "dominant_color": color,
                        "crop": pattern_crop,
                        "mask": mask[y:y+h, x:x+w],
                    })

        # 按面积排序
        patterns.sort(key=lambda p: p["area"], reverse=True)
        return patterns

    def extract_patterns_sam(self, image: np.ndarray) -> List[Dict]:
        """使用 SAM 提取纹样区域"""
        mask_generator = self._load_sam()
        if mask_generator is None:
            return self.extract_patterns_classical(image)

        masks = mask_generator.generate(image)
        patterns = []
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            crop = image[y:y+h, x:x+w]
            # 应用 mask
            mask_crop = mask[y:y+h, x:x+w]
            crop_masked = crop.copy()
            crop_masked[mask_crop == 0] = 255  # 背景填白

            patterns.append({
                "bbox": [x, y, w, h],
                "area": mask_data["area"],
                "stability_score": mask_data["stability_score"],
                "crop": crop_masked,
                "mask": mask_crop,
            })

        patterns.sort(key=lambda p: p["area"], reverse=True)
        return patterns

    def extract(self, image_path: str) -> Dict:
        """
        主入口：从图片中提取所有黎锦纹样

        Returns:
            {
                "patterns": [...],
                "count": int,
                "dominant_colors": [...],
                "image_size": (w, h)
            }
        """
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        img_array = self.preprocess(img_array)

        if self.use_sam:
            patterns = self.extract_patterns_sam(img_array)
        else:
            patterns = self.extract_patterns_classical(img_array)

        # 提取主色调
        dominant_colors = []
        for p in patterns[:5]:
            crop = p["crop"]
            pixels = crop.reshape(-1, 3)
            avg_color = pixels.mean(axis=0).astype(int).tolist()
            dominant_colors.append(avg_color)

        return {
            "patterns": patterns,
            "count": len(patterns),
            "dominant_colors": dominant_colors,
            "image_size": img.size,
        }
