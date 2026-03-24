"""
纹样矢量化导出模块
将提取的纹样转换为 SVG 矢量图
"""
import numpy as np
from PIL import Image
import svgwrite
import cv2
from typing import Dict, List


class PatternVectorizer:
    """将位图纹样转换为 SVG 矢量图"""

    def bitmap_to_svg_paths(self, mask: np.ndarray, color: List[int]) -> List[str]:
        """将二值 mask 转换为 SVG path 数据"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        paths = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            # 简化轮廓点
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue
            d = f"M {points[0][0]} {points[0][1]} "
            d += " ".join(f"L {p[0]} {p[1]}" for p in points[1:])
            d += " Z"
            paths.append(d)
        return paths

    def export_pattern_svg(self, pattern: Dict, output_path: str) -> str:
        """
        导出单个纹样为 SVG

        Args:
            pattern: 纹样数据字典（来自 segmentation 模块）
            output_path: 输出 SVG 文件路径

        Returns:
            SVG 文件路径
        """
        crop = pattern["crop"]
        mask = pattern.get("mask")
        h, w = crop.shape[:2]
        color = pattern.get("dominant_color", [100, 100, 100])

        dwg = svgwrite.Drawing(output_path, size=(w, h))
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

        if mask is not None:
            svg_color = f"rgb({color[0]},{color[1]},{color[2]})"
            paths = self.bitmap_to_svg_paths(mask, color)
            for path_d in paths:
                dwg.add(dwg.path(d=path_d, fill=svg_color, stroke="none"))
        else:
            # 无 mask 时直接嵌入位图
            import base64, io
            img_pil = Image.fromarray(crop)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            dwg.add(dwg.image(
                href=f"data:image/png;base64,{b64}",
                insert=(0, 0), size=(w, h)
            ))

        dwg.save()
        return output_path

    def export_pattern_sheet(self, patterns: List[Dict], output_path: str, cols: int = 4) -> str:
        """
        将多个纹样导出为一张纹样图谱 SVG

        Args:
            patterns: 纹样列表
            output_path: 输出路径
            cols: 每行列数

        Returns:
            SVG 文件路径
        """
        cell_size = 120
        padding = 10
        rows = (len(patterns) + cols - 1) // cols
        total_w = cols * (cell_size + padding) + padding
        total_h = rows * (cell_size + padding) + padding

        dwg = svgwrite.Drawing(output_path, size=(total_w, total_h))
        dwg.add(dwg.rect(insert=(0, 0), size=(total_w, total_h), fill="#f5f0e8"))

        # 标题
        dwg.add(dwg.text(
            "黎锦纹样图谱",
            insert=(total_w // 2, 30),
            text_anchor="middle",
            font_size="18px",
            font_family="serif",
            fill="#333"
        ))

        for idx, pattern in enumerate(patterns):
            row = idx // cols
            col = idx % cols
            x = padding + col * (cell_size + padding)
            y = padding + row * (cell_size + padding) + 40

            # 纹样背景框
            dwg.add(dwg.rect(
                insert=(x, y), size=(cell_size, cell_size),
                fill="white", stroke="#ccc", stroke_width=1,
                rx=4, ry=4
            ))

            # 纹样编号
            dwg.add(dwg.text(
                f"#{idx+1}",
                insert=(x + 5, y + 15),
                font_size="10px",
                fill="#999"
            ))

            # 颜色块
            color = pattern.get("dominant_color", [150, 100, 80])
            svg_color = f"rgb({color[0]},{color[1]},{color[2]})"
            dwg.add(dwg.rect(
                insert=(x + cell_size - 20, y + 5),
                size=(15, 15),
                fill=svg_color,
                rx=2, ry=2
            ))

        dwg.save()
        return output_path
