# 🧵 黎锦纹样智能提取器 (Lijin Pattern Extractor)

> 拍一张照片，AI 自动精准提取黎锦传统纹样

## ✨ 功能特性

- 📸 **拍照即提取** — 手机/相机拍摄黎锦织物，一键识别纹样
- 🎨 **纹样分割** — 基于深度学习精准分割纹样区域
- 🔍 **去重清理** — 自动识别并清理重复/相似纹样图片
- 📦 **纹样库管理** — 建立个人黎锦纹样数字档案库
- 🖼️ **矢量导出** — 提取结果可导出为 SVG/PNG 矢量图

## 🏗️ 技术架构

```
lijin-pattern-extractor/
├── src/
│   ├── extractor/          # 纹样提取核心模块
│   │   ├── segmentation.py # 图像分割
│   │   ├── dedup.py        # 重复检测
│   │   └── vectorize.py    # 矢量化导出
│   ├── api/                # FastAPI 后端
│   │   └── main.py
│   └── web/                # 前端界面
│       └── index.html
├── models/                 # 预训练模型权重
├── tests/
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 访问界面

打开浏览器访问 `http://localhost:8000`，上传黎锦图片即可开始提取。

## 📖 API 文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/extract` | POST | 上传图片，提取纹样 |
| `/dedup` | POST | 批量去重检测 |
| `/patterns` | GET | 获取纹样库列表 |
| `/export/{id}` | GET | 导出指定纹样为 SVG |

## 🧠 模型说明

本项目使用以下 AI 模型：

- **分割模型**: SAM (Segment Anything Model) — Meta AI
- **特征提取**: CLIP ViT-B/32 — 用于相似度计算去重
- **颜色分析**: K-Means 聚类 — 提取主色调

## 🌺 关于黎锦

黎锦是海南黎族传统纺织技艺，已有3000多年历史，2009年被列入联合国教科文组织非物质文化遗产名录。本工具旨在通过 AI 技术助力黎锦纹样的数字化保护与传承。

## 📄 License

MIT License
