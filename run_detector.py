#!/usr/bin/env python
"""
独立运行奶牛检测头（YOLO），不依赖 SAM3 分割模块。

用法:
    # 单张图像
    python run_detector.py --image cow.jpg --model runs/detect/train/weights/best.pt

    # 批量目录
    python run_detector.py --dir ./test_images --model yolo11n.pt

    # 实时摄像头（按 q 退出）
    python run_detector.py --camera 0 --model best.pt

    # 输出结果到 JSON
    python run_detector.py --image cow.jpg --model best.pt --save-json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 检测器封装
# ═══════════════════════════════════════════════════════════════════

class CowDetector:
    """YOLO 奶牛检测器，轻量封装，仅负责推理。"""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 20,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        print(f"[CowDetector] 已加载: {model_path}")

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        返回检测结果列表，每个元素:
            {"bbox": np.ndarray (4,) xyxy, "score": float, "class_id": int}
        """
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes, scores, cls_ids):
                detections.append({"bbox": b, "score": float(s), "class_id": int(c)})
        return detections


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════

def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    class_names: dict = None,
) -> np.ndarray:
    """在原图上绘制检测框和标签。"""
    if class_names is None:
        class_names = {0: "cow"}
    out = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"].astype(int)
        score = det["score"]
        cls_name = class_names.get(det["class_id"], f"cls_{det['class_id']}")
        label = f"{cls_name} {score:.2f}"

        # 框
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标签背景
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


# ═══════════════════════════════════════════════════════════════════
# 处理入口
# ═══════════════════════════════════════════════════════════════════

def process_image(
    image_path: str,
    detector: CowDetector,
    output_dir: str,
    save_json: bool,
    show: bool,
) -> None:
    """处理单张图像。"""
    print(f"\n处理: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: 无法读取 {image_path}")
        return

    t0 = time.time()
    detections = detector.detect(image)
    elapsed = (time.time() - t0) * 1000

    print(f"  检测到 {len(detections)} 个目标")
    print(f"  耗时: {elapsed:.1f} ms")
    for i, det in enumerate(detections):
        print(f"    [{i}] bbox={det['bbox'].astype(int).tolist()}  "
              f"score={det['score']:.3f}  class={det['class_id']}")

    # 保存可视化
    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    vis = draw_detections(image, detections)
    vis_path = os.path.join(output_dir, f"{name}_det.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"  可视化已保存: {vis_path}")

    # 保存 JSON
    if save_json:
        json_path = os.path.join(output_dir, f"{name}_det.json")
        json_result = [
            {"bbox": det["bbox"].tolist(), "score": det["score"], "class_id": det["class_id"]}
            for det in detections
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        print(f"  JSON 已保存: {json_path}")

    if show:
        cv2.imshow("Detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(
    image_dir: str,
    detector: CowDetector,
    output_dir: str,
    save_json: bool,
) -> None:
    """批量处理目录。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"\n找到 {len(files)} 张图像")
    total_time = 0.0
    for fname in files:
        t0 = time.time()
        process_image(os.path.join(image_dir, fname), detector, output_dir, save_json, show=False)
        total_time += time.time() - t0
    print(f"\n批量完成，总耗时: {total_time * 1000:.1f} ms, "
          f"平均: {total_time / max(len(files), 1) * 1000:.1f} ms/张")


def process_camera(
    camera_id: int,
    detector: CowDetector,
) -> None:
    """实时摄像头检测（按 q 退出）。"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: 无法打开摄像头 {camera_id}")
        return
    print(f"摄像头 {camera_id} 已开启，按 'q' 退出")

    fps_smooth = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections = detector.detect(frame)
        fps = 1.0 / max(time.time() - t0, 0.001)
        fps_smooth = fps_smooth * 0.9 + fps * 0.1

        vis = draw_detections(frame, detections)
        cv2.putText(vis, f"FPS: {fps_smooth:.1f} | Count: {len(detections)}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Cow Detector", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="独立奶牛检测头（YOLO）")
    # 输入
    p.add_argument("--image", type=str, help="单张图像路径")
    p.add_argument("--dir", type=str, help="图像目录（批量）")
    p.add_argument("--camera", type=int, default=None, help="摄像头 ID")

    # 模型
    p.add_argument("--model", type=str, required=True, help="YOLO 模型路径 (.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    p.add_argument("--max-det", type=int, default=20, help="最多检测数")
    p.add_argument("--device", type=str, default="cuda", help="推理设备")

    # 输出
    p.add_argument("--output", type=str, default="./output", help="输出目录")
    p.add_argument("--save-json", action="store_true", help="保存检测结果为 JSON")
    p.add_argument("--show", action="store_true", help="显示检测结果窗口")

    return p.parse_args()


def main():
    args = parse_args()

    detector = CowDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
    )

    if args.image:
        process_image(args.image, detector, args.output, args.save_json, args.show)
    elif args.dir:
        process_directory(args.dir, detector, args.output, args.save_json)
    elif args.camera is not None:
        process_camera(args.camera, detector)
    else:
        print("请指定 --image、--dir 或 --camera。使用 --help 查看帮助。")


if __name__ == "__main__":
    main()
