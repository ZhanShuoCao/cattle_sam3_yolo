#!/usr/bin/env python
"""
SAM3 检测头独立运行脚本。

使用 SAM3 内置的文本提示检测头（forward_grounding）直接进行目标检测，
不依赖 YOLO 等外部检测器。输入文本提示（如 "cow"），SAM3 端到端输出
检测框、分割 mask 和置信度。

用法:
    # 单张图像
    python run_sam3_detector.py --image test.jpg --text "cow"

    # 指定模型路径和置信度
    python run_sam3_detector.py --image cow.jpg --text "cow" --conf 0.3

    # 多文本提示
    python run_sam3_detector.py --image farm.jpg --text "cow.person.tractor"

    # 批量目录
    python run_sam3_detector.py --dir ./images --text "cow"

    # 保存结果
    python run_sam3_detector.py --image test.jpg --text "cow" --output ./results --save-json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# 将 sam3 源码目录加入 Python 路径
_SAM3_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3-main")
if _SAM3_SRC not in sys.path:
    sys.path.insert(0, _SAM3_SRC)


# ═══════════════════════════════════════════════════════════════════
# SAM3 检测头封装
# ═══════════════════════════════════════════════════════════════════

class SAM3Detector:
    """SAM3 内置检测头（文本提示 → boxes + masks + scores）。"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None

        print(f"[SAM3Detector] 加载模型: {checkpoint_path}")
        from sam3.model_builder import build_sam3_image_model

        self._model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )

        from sam3.model.sam3_image_processor import Sam3Processor

        self._processor = Sam3Processor(
            model=self._model,
            resolution=1008,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        print("[SAM3Detector] 模型加载完成")

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict:
        """
        运行 SAM3 检测头。

        Args:
            image: (H, W, 3) BGR numpy array（OpenCV 格式）
            text_prompt: 文本提示，多个类别用 "." 分隔，如 "cow.person"

        Returns:
            dict: {
                "boxes": np.ndarray (N, 4) xyxy 像素坐标,
                "scores": np.ndarray (N,) 置信度,
                "masks": np.ndarray (N, H, W) bool 分割 mask,
                "masks_logits": np.ndarray (N, H, W) sigmoid 之前的 logits,
                "orig_size": (H, W),
                "text_prompt": str,
            }
        """
        # BGR → RGB → PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # SAM3 模型使用 bfloat16 混合精度，必须包裹 autocast
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            state = self._processor.set_image(pil_image)
            state = self._processor.set_text_prompt(text_prompt, state)

        boxes = state["boxes"].cpu().float().numpy()           # (N, 4) xyxy
        scores = state["scores"].cpu().float().numpy()        # (N,)
        masks = state["masks"].cpu().numpy()                   # (N, H, W) bool
        masks_logits = state["masks_logits"].cpu().float().numpy()  # (N, H, W) float

        # masks 维度 → squeeze channel
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks_logits.ndim == 4:
            masks_logits = masks_logits.squeeze(1)

        h, w = image.shape[:2]

        return {
            "boxes": boxes,
            "scores": scores,
            "masks": masks,
            "masks_logits": masks_logits,
            "orig_size": (h, w),
            "text_prompt": text_prompt,
        }


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════

def _generate_colors(n: int) -> List[tuple]:
    """生成 n 个 HSV 颜色（BGR 格式）。"""
    colors = []
    for i in range(max(n, 1)):
        hsv = np.uint8([[[int(i * 180 / max(n, 1)) % 180, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def draw_results(
    image: np.ndarray,
    result: dict,
    alpha: float = 0.4,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    将检测结果绘制到图像上。

    Args:
        image: (H, W, 3) BGR 图像
        result: detect() 返回的 dict
        alpha: mask 叠加透明度
        line_thickness: 框线宽度

    Returns:
        (H, W, 3) BGR 可视化图像
    """
    out = image.copy()
    boxes = result["boxes"]
    scores = result["scores"]
    masks = result["masks"]
    h, w = image.shape[:2]
    n = len(boxes)
    colors = _generate_colors(n)

    # 1. 绘制 mask 叠加
    overlay = out.copy()
    for i in range(n):
        if masks[i].sum() == 0:
            continue
        color = colors[i]
        overlay[masks[i]] = color
    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)

    # 2. 绘制检测框 + 标签
    for i in range(n):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = scores[i]
        color = colors[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        label = f"#{i} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        # 标签背景
        bg_y1 = max(y1 - th - 8, 0)
        cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # 3. 左上角统计信息
    info = f"prompt: '{result['text_prompt']}' | detections: {n}"
    cv2.putText(
        out, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )

    return out


# ═══════════════════════════════════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════════════════════════════════

def process_image(
    image_path: str,
    detector: SAM3Detector,
    text_prompt: str,
    output_dir: str,
    save_json: bool,
    show: bool,
) -> None:
    """处理单张图像。"""
    print(f"\n{'=' * 60}")
    print(f"图像: {image_path}")
    print(f"提示: '{text_prompt}'")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: 无法读取 {image_path}")
        return

    t0 = time.time()
    result = detector.detect(image, text_prompt)
    elapsed_ms = (time.time() - t0) * 1000

    boxes = result["boxes"]
    scores = result["scores"]

    print(f"检测到 {len(boxes)} 个目标（耗时 {elapsed_ms:.1f} ms）")
    for i in range(len(boxes)):
        b = boxes[i].astype(int)
        print(f"  [{i}] box=[{b[0]},{b[1]},{b[2]},{b[3]}] "
              f"score={scores[i]:.4f}  area={result['masks'][i].sum()}px")

    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    safe_prompt = text_prompt.replace(".", "_").replace("/", "_")

    # 可视化
    vis = draw_results(image, result)
    vis_path = os.path.join(output_dir, f"{name}_{safe_prompt}_det.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"可视化: {vis_path}")

    # 保存 mask 图像
    for i in range(len(boxes)):
        mask_path = os.path.join(output_dir, f"{name}_{safe_prompt}_mask_{i}.png")
        cv2.imwrite(mask_path, (result["masks"][i].astype(np.uint8)) * 255)

    # JSON
    if save_json:
        json_path = os.path.join(output_dir, f"{name}_{safe_prompt}_det.json")
        json_data = {
            "image": image_path,
            "text_prompt": text_prompt,
            "elapsed_ms": elapsed_ms,
            "detections": [
                {
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "mask_area": int(result["masks"][i].sum()),
                }
                for i in range(len(boxes))
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")

    if show:
        cv2.imshow("SAM3 Detector", vis)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(
    image_dir: str,
    detector: SAM3Detector,
    text_prompt: str,
    output_dir: str,
    save_json: bool,
) -> None:
    """批量处理目录中的图像。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not files:
        print(f"目录 {image_dir} 中未找到图像文件")
        return

    print(f"\n找到 {len(files)} 张图像，开始批量处理...")
    total_ms = 0.0
    for fname in files:
        t0 = time.time()
        process_image(
            os.path.join(image_dir, fname),
            detector, text_prompt, output_dir, save_json, show=False,
        )
        total_ms += (time.time() - t0) * 1000
    print(f"\n批量完成: {len(files)} 张, 总耗时 {total_ms:.0f} ms, "
          f"平均 {total_ms / len(files):.0f} ms/张")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SAM3 检测头独立推理 — 使用文本提示直接检测目标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_sam3_detector.py --image test.jpg --text "cow"
  python run_sam3_detector.py --image farm.jpg --text "cow.person" --conf 0.3
  python run_sam3_detector.py --dir ./images --text "cow" --save-json
        """,
    )
    # 输入
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", type=str, help="单张图像路径")
    inp.add_argument("--dir", type=str, help="图像目录（批量）")

    # 文本提示
    p.add_argument("--text", type=str, default="cow",
                   help="文本提示，多个类别用 '.' 分隔，如 'cow.person'（默认: cow）")

    # 模型
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3.pt"),
                   help="SAM3 checkpoint 路径（默认: ./sam3.pt）")
    p.add_argument("--device", type=str, default="cuda",
                   help="推理设备（默认: cuda）")
    p.add_argument("--conf", type=float, default=0.5,
                   help="置信度阈值（默认: 0.5）")

    # 输出
    p.add_argument("--output", type=str, default="./output",
                   help="输出目录（默认: ./output）")
    p.add_argument("--save-json", action="store_true",
                   help="同时保存 JSON 结果")
    p.add_argument("--show", action="store_true",
                   help="显示检测结果窗口")

    return p.parse_args()


def main():
    args = parse_args()

    # 检查模型文件
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] 模型文件不存在: {args.checkpoint}")
        print("请指定正确的 --checkpoint 路径，或从 HuggingFace 下载 sam3.pt")
        sys.exit(1)

    # 初始化检测器（只加载一次模型）
    print("=" * 60)
    print("SAM3 检测头独立推理")
    print(f"  模型: {args.checkpoint}")
    print(f"  置信度阈值: {args.conf}")
    print(f"  设备: {args.device}")
    print("=" * 60)

    detector = SAM3Detector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.conf,
    )

    if args.image:
        process_image(
            args.image, detector, args.text,
            args.output, args.save_json, args.show,
        )
    elif args.dir:
        process_directory(
            args.dir, detector, args.text,
            args.output, args.save_json,
        )

    print("\n完成。")


if __name__ == "__main__":
    main()
