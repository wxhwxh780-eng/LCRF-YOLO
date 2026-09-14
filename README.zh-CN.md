# LCRF-YOLO

论文：

**LCRF-YOLO：基于局部对比增强与响应校准跨尺度融合的轻量化工业缺陷检测方法**

## 简介

本仓库提供 LCRF-YOLO 的官方实现代码。

LCRF-YOLO 针对工业缺陷检测中低对比缺陷表征不足以及跨尺度特征融合后响应干扰的问题，在 YOLOv11n 基础上设计了一种轻量化检测框架。

该方法包含两个核心模块：

- **Local Contrast Residual Block (LCRB)**  
  通过局部邻域对比增强与轻量化通道校准，提高弱缺陷区域的局部判别能力。

- **Response-Calibrated Cross-Scale Fusion (RCSFusion)**  
  通过统计引导的响应校准与细节增强分支，缓解跨尺度融合过程中弱响应被削弱的问题。

LCRF-YOLO 在提升缺陷检测性能的同时保持较低的计算开销。

---

## 环境配置

实验环境：

- Python 3.8
- PyTorch 2.1.0
- Ultralytics 8.4.21

安装依赖：

```bash
pip install -r requirements.txt
