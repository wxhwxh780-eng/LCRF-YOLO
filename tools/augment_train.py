import cv2
import yaml
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(root) / p


def collect_images(img_dir):
    img_dir = Path(img_dir)
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(img_dir.glob(f"*{ext}"))
        imgs.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def img_to_label_path(img_path):
    """
    Convert:
        .../images/train/xxx.jpg
    to:
        .../labels/train/xxx.txt
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" not in parts:
        raise ValueError(f"Cannot infer label path because 'images' not in path: {img_path}")

    idx = parts.index("images")
    parts[idx] = "labels"

    return Path(*parts).with_suffix(".txt")


def read_yolo_label(label_path):
    label_path = Path(label_path)

    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    try:
        labels = np.loadtxt(str(label_path), dtype=np.float32)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32)

    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    labels = labels.reshape(-1, 5)
    return labels


def save_yolo_label(label_path, labels):
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        for row in labels:
            cls_id = int(row[0])
            x, y, w, h = row[1:]
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def random_scale(img, labels, scale_range=(0.8, 1.2)):
    """
    等比例缩放整张图。
    YOLO 标签是归一化坐标，所以整图缩放后 bbox 不需要改。
    """
    scale = random.uniform(scale_range[0], scale_range[1])

    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return img, labels


def random_horizontal_flip(img, labels, p=0.5):
    """
    水平翻转：
    x_center = 1 - x_center
    """
    if random.random() < p:
        img = img[:, ::-1].copy()

        if labels.size > 0:
            labels[:, 1] = 1.0 - labels[:, 1]
            labels[:, 1] = np.clip(labels[:, 1], 0.0, 1.0)

    return img, labels


def hsv_jitter(img, hsv_gain=0.2):
    """
    只扰动 S 和 V：
    S, V ∈ [0.8, 1.2]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    s_gain = random.uniform(1.0 - hsv_gain, 1.0 + hsv_gain)
    v_gain = random.uniform(1.0 - hsv_gain, 1.0 + hsv_gain)

    hsv[:, :, 1] *= s_gain
    hsv[:, :, 2] *= v_gain

    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def augment_one(img, labels):
    img, labels = random_scale(img, labels, scale_range=(0.8, 1.2))
    img, labels = random_horizontal_flip(img, labels, p=0.5)
    img = hsv_jitter(img, hsv_gain=0.2)

    return img, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="data yaml path")
    parser.add_argument("--mult", type=int, default=6, help="final train multiplier, e.g. 6 means original + 5 augmented")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="allow running even if aug files already exist")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_yaml = load_yaml(args.yaml)

    if "path" in data_yaml and data_yaml["path"] is not None:
        root = Path(data_yaml["path"])
    else:
        root = Path(args.yaml).parent

    train_img_dir = resolve_path(root, data_yaml["train"])

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Train image directory not found: {train_img_dir}")

    train_imgs = collect_images(train_img_dir)

    # 只对原始图片增强，避免重复增强增强图
    base_imgs = [
        p for p in train_imgs
        if "_aug" not in p.stem
    ]

    if len(base_imgs) == 0:
        raise RuntimeError(f"No base images found in: {train_img_dir}")

    # 防止重复运行导致数量爆炸
    existing_aug = [
        p for p in train_imgs
        if "_aug" in p.stem
    ]

    if len(existing_aug) > 0 and not args.overwrite:
        print("\nWARNING: Found existing augmented images.")
        print(f"Existing augmented images: {len(existing_aug)}")
        print("To avoid duplicate augmentation, stop now.")
        print("If you really want to overwrite/run again, add --overwrite.")
        return

    print("\n========== AGE-style Offline Augmentation ==========")
    print(f"YAML: {args.yaml}")
    print(f"Train image dir: {train_img_dir}")
    print(f"Original train images: {len(base_imgs)}")
    print(f"Multiplier: x{args.mult}")
    print(f"Will generate: {len(base_imgs) * (args.mult - 1)} augmented images")

    generated = 0

    for img_path in tqdm(base_imgs, desc="Augment train"):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        label_path = img_to_label_path(img_path)
        labels = read_yolo_label(label_path)

        for i in range(1, args.mult):
            aug_img, aug_labels = augment_one(img.copy(), labels.copy())

            out_img_path = img_path.parent / f"{img_path.stem}_aug{i}.jpg"
            out_label_path = label_path.parent / f"{label_path.stem}_aug{i}.txt"

            cv2.imwrite(str(out_img_path), aug_img)
            save_yolo_label(out_label_path, aug_labels)

            generated += 1

    final_imgs = collect_images(train_img_dir)

    print("\n========== Done ==========")
    print(f"Generated augmented images: {generated}")
    print(f"Final train images: {len(final_imgs)}")
    print(f"Expected final images: {len(base_imgs) * args.mult}")
    print("Val/test are unchanged because this script only modifies train directory.")


if __name__ == "__main__":
    main()