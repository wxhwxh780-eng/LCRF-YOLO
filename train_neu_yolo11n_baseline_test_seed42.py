import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/jiao/users/xiexy/ultralytics-main")

from ultralytics import YOLO


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data_yaml = "/home/jiao/users/xiexy/ultralytics-main/AGE-YOLO-datasets/neu_x5/neu_data.yaml"
    model_yaml = "/home/jiao/users/xiexy/ultralytics-main/ultralytics-main/ultralytics/cfg/models/11/yolo11n_neu.yaml"

    # 先用官方 AGE-YOLO 数据集 split 跑 YOLO11n baseline
    # 注意：这里是从头训练，不加载 yolo11n.pt
    model = YOLO(model_yaml, task="detect")

    train_results = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,

        deterministic=True,
        seed=42,

        project="runs/AGE/LAST/NEU",
        name="LCRB(backbone P4) + RCSFusion+seed=0",

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        close_mosaic=20,
        patience=100,

        pretrained=False,

        # 尽量贴近 AGE-YOLO 论文/源码里描述的 mixed augmentation
        # 先不做离线 x5，只使用在线增强
        mosaic=1.0,
        scale=0.1,

        hsv_s=0.2,
        hsv_v=0.2,
        fliplr=0.0,
        flipud=0.0,

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
        project="runs/AGE_eval",
        name="age+trc3k2+dcru"
    )

    print("\n========== TEST METRICS SUMMARY ==========")
    print(f"mAP50:    {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision:{test_metrics.box.mp:.4f}")
    print(f"Recall:   {test_metrics.box.mr:.4f}")