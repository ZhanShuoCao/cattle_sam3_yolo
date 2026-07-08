"""
数据加载和转换工具。
- 支持 YOLO 检测格式（class_id cx cy w h，归一化坐标）
- 支持 LabelMe JSON 格式（polygon → binary mask）
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ─── YOLO 检测数据 ────────────────────────────────────────────────

def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Dict]:
    """
    读取单个 YOLO 标注文件，返回检测框列表。
    YOLO 格式: class_id cx cy w h（归一化到 [0,1]）

    Returns:
        List[Dict]: 每个元素含 {"bbox_xyxy": [x1,y1,x2,y2], "class_id": int}
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            # 转回像素坐标 (xyxy)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            # 裁剪到图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            boxes.append({"bbox_xyxy": [x1, y1, x2, y2], "class_id": class_id})
    return boxes


def yolo_dataset_to_coco(
    image_dir: str,
    label_dir: str,
    class_names: List[str],
    output_json: Optional[str] = None,
) -> Dict:
    """
    将 YOLO 检测数据集转为 COCO JSON 格式，方便后续评估。

    Args:
        image_dir: 图像目录
        label_dir: 标注目录
        class_names: 类别名列表，如 ["cattle"]
        output_json: 可选，保存路径

    Returns:
        COCO 格式 dict
    """
    images, annotations = [], []
    ann_id = 0

    image_files = sorted(os.listdir(image_dir))
    for img_id, fname in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        label_fname = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_dir, label_fname)
        boxes = load_yolo_labels(label_path, w, h)
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            bw, bh = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box["class_id"] + 1,  # COCO 类别从 1 开始
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)],
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    return coco_dict


# ─── LabelMe 分割数据 ──────────────────────────────────────────────

def labelme_json_to_masks(json_path: str) -> List[Dict]:
    """
    读取单个 LabelMe JSON 文件，将 polygon 标注转为 binary mask。

    Returns:
        List[Dict]: 每个元素含 {
            "mask": np.ndarray (H, W) bool,
            "label": str,
            "group_id": int,
            "bbox_xyxy": [x1,y1,x2,y2]
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    instances = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        label = shape.get("label", "unknown")
        group_id = shape.get("group_id", 0)
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        # 将 points 转为整数坐标
        pts = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        mask_bool = mask.astype(bool)

        # 计算 bounding box
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        instances.append({
            "mask": mask_bool,
            "label": label,
            "group_id": group_id,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return instances


def load_labelme_dataset(
    data_dir: str,
) -> List[Dict]:
    """
    加载整个 LabelMe 分割数据集。

    Args:
        data_dir: 数据集根目录，递归搜索所有 .json 文件

    Returns:
        List[Dict]: 每个元素含 {
            "image_path": str,
            "image_size": (w, h),
            "instances": List[Dict]  # labelme_json_to_masks 的输出
        }
    """
    samples = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            json_path = os.path.join(root, f)
            # 查找同名图像
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = os.path.join(root, f.replace(".json", ext))
                if os.path.exists(img_path):
                    break
            else:
                # 尝试 json 中记录的 imagePath
                try:
                    with open(json_path, "r") as fh:
                        meta = json.load(fh)
                    img_name = meta.get("imagePath", "")
                    img_path = os.path.join(root, img_name)
                except Exception:
                    img_path = ""

            if not img_path or not os.path.exists(img_path):
                continue

            instances = labelme_json_to_masks(json_path)
            if not instances:
                continue

            img = Image.open(img_path)
            samples.append({
                "image_path": img_path,
                "image_size": img.size,  # (w, h)
                "instances": instances,
            })

    return samples


def labelme_to_coco(
    data_dir: str,
    category_names: Optional[List[str]] = None,
    output_json: Optional[str] = None,
) -> Dict:
    """
    将 LabelMe 分割数据集转换为 COCO JSON 格式。

    Args:
        data_dir: LabelMe 数据集根目录
        category_names: 类别名 → id 映射，默认 {"cattle": 1}
        output_json: 可选输出路径

    Returns:
        COCO 格式 dict
    """
    if category_names is None:
        category_names = ["cattle"]
    cat_map = {name: i + 1 for i, name in enumerate(category_names)}

    images, annotations = [], []
    ann_id = 0

    samples = load_labelme_dataset(data_dir)
    for img_id, sample in enumerate(samples, 1):
        w, h = sample["image_size"]
        images.append({
            "id": img_id,
            "file_name": os.path.basename(sample["image_path"]),
            "width": w,
            "height": h,
        })

        for inst in sample["instances"]:
            mask = inst["mask"]
            cat_id = cat_map.get(inst["label"], 1)

            # RLE 或 polygon 编码
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            segmentation = []
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                cnt = cnt.flatten().tolist()
                if len(cnt) >= 6:  # 至少 3 个点
                    segmentation.append(cnt)

            x1, y1, x2, y2 = inst["bbox_xyxy"]
            bw, bh = x2 - x1, y2 - y1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, bw, bh],
                "area": int(mask.sum()),
                "segmentation": segmentation,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(category_names)],
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    return coco_dict
