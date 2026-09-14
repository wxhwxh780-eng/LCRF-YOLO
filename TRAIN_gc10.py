import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/jiao/users/xiexy/ultralytics-main")

from ultralytics import YOLO


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data_yaml = "/home/jiao/users/xiexy/ultralytics-main//gc10_x4/gc10_data.yaml"
    model_yaml = "/home/jiao/users/xiexy/ultralytics-main/ultralytics-main/ultralytics/cfg/models/11/yolo11-gc.yaml"

    model = YOLO(model_yaml)

    train_results = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,

        deterministic=True,




        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        close_mosaic=20,
        patience=50,

        pretrained=False,


        plots=True
    )

    # 自动获取本次训练保存目录，避免手动写错 runs/detect/runs/... 路径
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    print("\n========== TRAIN FINISHED ==========")
    print(f"Save dir: {save_dir}")
    print(f"Best pt:  {best_pt}")

    # 训练完成后，加载 best.pt 单独测试 test split
    print("\n========== TEST RESULT ON OFFICIAL AGE-YOLO NEU-DET SPLIT ==========")
    best_model = YOLO(str(best_pt))

    print("\n========== ALPHA CHECK ==========")
    has_alpha = False
    for name, param in best_model.model.named_parameters():
        if "alpha" in name:
            has_alpha = True
            print(f"{name}: alpha={param.data.item():.6f}, tanh(alpha)={torch.tanh(param.data).item():.6f}")

    if not has_alpha:
        print("No alpha found. 当前模型里可能没有 RCSFusion/带 alpha 的模块。")
    test_metrics = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        name="age_x5+fcmanet_l6_seed42"
    )

    print("\n========== TEST METRICS SUMMARY ==========")
    print(f"mAP50:    {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision:{test_metrics.box.mp:.4f}")
    print(f"Recall:   {test_metrics.box.mr:.4f}")
