
内容：

```markdown
# LCRF-YOLO

**面向弱对比工业缺陷检测的轻量化 YOLO 网络**

本项目提供论文：

"LCRF-YOLO: Local Contrast Residual and Response-Calibrated Cross-Scale Fusion for Lightweight Industrial Defect Detection"

的官方代码实现。

---

## 简介

工业表面缺陷检测通常存在：

- 缺陷与背景对比度低；
- 小目标和细粒度纹理难以保持；
- 多尺度融合过程中响应容易被背景干扰。

针对上述问题，本文提出 LCRF-YOLO：

### 1. Local Contrast Residual Block (LCRB)

通过局部对比残差增强缺陷区域表征能力，提高弱纹理目标响应。

### 2. Response-Calibrated Cross-Scale Fusion (RCSFusion)

利用统计响应信息进行跨尺度特征校准，提高融合后的有效响应。

---

## 环境

- Python 3.8
- PyTorch 2.1.0
- Ultralytics YOLO

---

## 数据集

实验采用：

- GC10-DET
- NEU-DET
- HRIPCB

数据需转换为 YOLO 标注格式。

---

