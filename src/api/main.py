"""
黎锦纹样提取器 - FastAPI 后端
"""
import os
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from src.extractor.segmentation import LijinPatternSegmentor
from src.extractor.dedup import LijinDeduplicator
from src.extractor.vectorize import PatternVectorizer

app = FastAPI(
    title="黎锦纹样智能提取器",
    description="AI 驱动的黎锦传统纹样识别与提取工具",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

segmentor = LijinPatternSegmentor(use_sam=False)  # 默认用经典CV，SAM需要模型文件
deduplicator = LijinDeduplicator()
vectorizer = PatternVectorizer()


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = Path("src/web/index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>黎锦纹样提取器 API 运行中</h1><p>访问 /docs 查看 API 文档</p>"


@app.post("/extract")
async def extract_patterns(file: UploadFile = File(...)):
    """
    上传黎锦图片，提取纹样

    - **file**: 图片文件 (JPG/PNG/WEBP)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    # 保存上传文件
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".jpg"
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"

    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        # 提取纹样
        result = segmentor.extract(str(upload_path))

        # 导出纹样图谱 SVG
        svg_path = OUTPUT_DIR / f"{file_id}_patterns.svg"
        if result["patterns"]:
            vectorizer.export_pattern_sheet(result["patterns"], str(svg_path))

        return {
            "id": file_id,
            "pattern_count": result["count"],
            "image_size": result["image_size"],
            "dominant_colors": result["dominant_colors"],
            "svg_url": f"/export/{file_id}" if svg_path.exists() else None,
            "patterns": [
                {
                    "index": i,
                    "bbox": p["bbox"],
                    "area": p["area"],
                    "dominant_color": p.get("dominant_color"),
                }
                for i, p in enumerate(result["patterns"][:20])
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"提取失败: {str(e)}")


@app.post("/dedup")
async def deduplicate_folder(folder_path: str):
    """
    对指定文件夹中的黎锦图片进行去重清理

    - **folder_path**: 图片文件夹路径
    """
    if not Path(folder_path).exists():
        raise HTTPException(404, "文件夹不存在")

    try:
        report = deduplicator.clean_duplicates(folder_path)
        return report
    except Exception as e:
        raise HTTPException(500, f"去重失败: {str(e)}")


@app.get("/export/{file_id}")
async def export_svg(file_id: str):
    """下载提取的纹样 SVG 图谱"""
    svg_path = OUTPUT_DIR / f"{file_id}_patterns.svg"
    if not svg_path.exists():
        raise HTTPException(404, "SVG 文件不存在")
    return FileResponse(svg_path, media_type="image/svg+xml")


@app.get("/patterns")
async def list_patterns():
    """获取已提取的纹样列表"""
    svgs = list(OUTPUT_DIR.glob("*_patterns.svg"))
    return {
        "count": len(svgs),
        "patterns": [
            {"id": f.stem.replace("_patterns", ""), "url": f"/export/{f.stem.replace(\"_patterns\", \"\")}"}
            for f in svgs
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "lijin-pattern-extractor"}
