import sys
import torch

sys.path.insert(0, "/home/jiao/users/xiexy/ultralytics-main")

from ultralytics import YOLO


if __name__ == "__main__":
    best_pt = "/home/jiao/users/xiexy/ultralytics-main/runs/detect/runs/AGE/age_x5+rcsfusion_p4_seed422/weights/best.pt"

    model = YOLO(best_pt)

    print("\n========== ALPHA CHECK ==========")
    has_alpha = False

    for name, param in model.model.named_parameters():
        if "alpha" in name:
            has_alpha = True
            print(
                f"{name}: "
                f"alpha={param.data.item():.6f}, "
                f"tanh(alpha)={torch.tanh(param.data).item():.6f}"
            )

    if not has_alpha:
        print("No alpha found. 说明当前 best.pt 里没有带 alpha 的模块，或者 RCSFusion 没有正确加载。")