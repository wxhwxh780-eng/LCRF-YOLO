# LCRF-YOLO

Official implementation of:

**LCRF-YOLO: Local Contrast Residual and Response-Calibrated Cross-Scale Fusion for Lightweight Industrial Defect Detection**

This repository provides the official implementation of LCRF-YOLO, a lightweight object detector designed for weak and low-contrast industrial surface defect detection.

The source code is released for research purposes.

---

## Overview

Industrial surface defects usually suffer from:

- low contrast between defects and background;
- irregular and small-scale appearances;
- insufficient feature responses after cross-scale fusion.

To address these challenges, LCRF-YOLO introduces two lightweight modules:

- **Local Contrast Residual Block (LCRB)**  
  Enhances local defect-aware representations by exploiting contrast differences.

- **Response-Calibrated Cross-Scale Fusion (RCSFusion)**  
  Refines cross-scale feature interaction through statistical response calibration.

The proposed framework is built upon YOLOv11n and achieves improved detection performance while maintaining lightweight computational complexity.

---

## Installation

The implementation is based on:

- Python 3.8
- PyTorch 2.1.0
- Ultralytics YOLO framework

Install dependencies:

```bash
pip install -r requirements.txt
