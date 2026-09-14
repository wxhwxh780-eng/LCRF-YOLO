#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrain GC10-DET YOLO11n baseline or LCRF-YOLO with fixed, easy-to-find paths.

Usage:
1) Baseline:
python train_gc10_fixed.py --mode baseline --seed 0

2) Full LCRF-YOLO:
python train_gc10_fixed.py --mode lcrf --seed 0

The final copied weights will always be saved under:
  /home/jiao/users/xiexy/ultralytics-main/GC10_RETRAIN_FOR_PAPER/final_weights/
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/jiao/users/xiexy/ultralytics-main")
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


DATA_YAML = REPO_ROOT / "AGE-YOLO-datasets/gc10_x4/gc10_data.yaml"

# Baseline uses the official YOLO11n architecture and trains from scratch.
BASELINE_MODEL = "yolo11n.yaml"

# IMPORTANT: change this only if your combined LCRB+RCSFusion YAML has another filename.
# The script checks that this path exists before training.
LCRF_MODEL_YAML = (
    REPO_ROOT
    / "ultralytics-main/ultralytics/cfg/models/11/yolo11-age.yaml"
)

OUTPUT_ROOT = REPO_ROOT / "GC10_RETRAIN_FOR_PAPER"
FINAL_WEIGHT_DIR = OUTPUT_ROOT / "final_weights"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "lcrf"],
        required=True,
        help="baseline = official YOLO11n; lcrf = LCRB+RCSFusion full model",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def check_paths(mode: str):
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data YAML not found:\n{DATA_YAML}")

    if mode == "lcrf" and not LCRF_MODEL_YAML.exists():
        raise FileNotFoundError(
            "LCRF model YAML not found. Please correct LCRF_MODEL_YAML at the top "
            f"of this script.\nCurrent path:\n{LCRF_MODEL_YAML}"
        )


def main():
    args = parse_args()
    check_paths(args.mode)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if args.mode == "baseline":
        model_source = BASELINE_MODEL
        run_name = f"gc10_yolo11n_baseline_seed{args.seed}"
        final_name = f"gc10_yolo11n_baseline_seed{args.seed}_best.pt"
    else:
        model_source = str(LCRF_MODEL_YAML)
        run_name = f"gc10_lcrf_yolo_seed{args.seed}"
        final_name = f"gc10_lcrf_yolo_seed{args.seed}_best.pt"

    run_dir = OUTPUT_ROOT / run_name
    eval_dir = OUTPUT_ROOT / "test_eval" / f"{run_name}_test"

    # Keep the path fixed. Do not silently create run_name2/run_name3.
    if run_dir.exists():
        raise FileExistsError(
            f"\nTraining directory already exists:\n{run_dir}\n\n"
            "To retrain from the beginning, rename or delete that directory first.\n"
            f"Example:\nrm -rf '{run_dir}'"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== TRAIN CONFIG ==========")
    print("Mode:", args.mode)
    print("Model:", model_source)
    print("Data:", DATA_YAML)
    print("Seed:", args.seed)
    print("Expected run dir:", run_dir)
    print("Final copied weight:", FINAL_WEIGHT_DIR / final_name)

    # Build the architecture from YAML; do not load pretrained .pt weights.
    model = YOLO(model_source)

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        device=args.device,

        deterministic=True,
        seed=args.seed,

        # Absolute and fixed output path.
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=True,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        close_mosaic=20,
        patience=50,
        pretrained=False,

        # Use the current gc10_x4 processed dataset as-is,
        # with the same online augmentation settings for both models.
        mosaic=1.0,
        scale=0.1,
        hsv_s=0.2,
        hsv_v=0.2,
        fliplr=0.0,
        flipud=0.0,

        plots=True,
    )

    save_dir = Path(model.trainer.save_dir).resolve()
    best_pt = save_dir / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found:\n{best_pt}")

    fixed_best = FINAL_WEIGHT_DIR / final_name
    shutil.copy2(best_pt, fixed_best)

    print("\n========== TRAIN FINISHED ==========")
    print("Actual save dir:", save_dir)
    print("Original best.pt:", best_pt)
    print("Copied best.pt:", fixed_best)

    # Test on the official test split.
    best_model = YOLO(str(fixed_best))

    print("\n========== ALPHA CHECK ==========")
    alpha_found = False
    for name, param in best_model.model.named_parameters():
        if "alpha" in name and param.numel() == 1:
            alpha_found = True
            alpha = param.detach().item()
            tanh_alpha = torch.tanh(param.detach()).item()
            print(f"{name}: alpha={alpha:.6f}, tanh(alpha)={tanh_alpha:.6f}")

    if args.mode == "baseline" and not alpha_found:
        print("Baseline correctly contains no learnable alpha parameter.")
    elif args.mode == "lcrf" and not alpha_found:
        print(
            "WARNING: No alpha parameter was found. Check whether the selected "
            "YAML really contains LCRB and RCSFusion."
        )

    metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=str(OUTPUT_ROOT / "test_eval"),
        name=f"{run_name}_test",
        exist_ok=True,
    )

    print("\n========== GC10 TEST METRICS ==========")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n========== USE THESE PATHS LATER ==========")
    if args.mode == "baseline":
        print(f'BASELINE="{fixed_best}"')
    else:
        print(f'FULL="{fixed_best}"')
    print(f"Test evaluation directory: {eval_dir}")


if __name__ == "__main__":
    main()