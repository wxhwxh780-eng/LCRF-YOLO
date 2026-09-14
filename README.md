# LCRF-YOLO

Official implementation of:

**LCRF-YOLO: Lightweight Industrial Defect Detection via Local Contrast Enhancement and Response-Calibrated Cross-Scale Fusion**

## Overview

This repository provides the official implementation of **LCRF-YOLO**, a lightweight detector designed for industrial defect detection under challenging conditions, including **low-contrast defects** and **background interference**.

LCRF-YOLO is built upon the YOLOv11n framework and introduces two lightweight feature refinement modules:

- **Local Contrast Residual Block (LCRB)**  
  Enhances local defect discriminability by exploiting neighborhood contrast differences and lightweight channel recalibration, improving the representation of weak defect cues.

- **Response-Calibrated Cross-Scale Fusion (RCSFusion)**  
  Calibrates feature responses after cross-scale aggregation through statistical-guided spatial gating and detail refinement, reducing response interference caused by feature fusion.

The proposed framework improves weak defect representation while maintaining a lightweight architecture suitable for efficient industrial deployment.

---

## Environment

The experiments are conducted under the following environment:

- Python 3.8
- PyTorch 2.1.0
- Ultralytics 8.4.21
- CUDA-enabled GPU

---

## Installation

Clone this repository:

```bash
git clone https://github.com/wxhwxh780-eng/LCRF-YOLO.git

cd LCRF-YOLO
