"""
Debug 可视化模块 — 输出检测框和分割 mask 叠加图。
"""

import colorsys
import os
from typing import List, Tuple

import cv2
import numpy as np

from config import VisualizationConfig


def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def visualize_result(result, config: VisualizationConfig, image_name: str = "output") -> None:
    if not config.enabled:
        return

    os.makedirs(config.output_dir, exist_ok=True)
    image = result.image.copy()
    colors = _generate_colors(max(len(result.instances), 1))

    # Mask overlay
    overlay = image.copy()
    for i, inst in enumerate(result.instances):
        mask = inst["mask"]
        if mask.sum() > 0:
            overlay[mask] = colors[i % len(colors)]
    out = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    # 检测框 + id 标签
    for i, inst in enumerate(result.instances):
        bbox = inst.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            color = colors[i % len(colors)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # 框左上角显示 "id: 0", "id: 1" 标签
            label = f"id: {i}"
            font_scale = 1.0
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # 白底标签 + 黑色外框
            bg_x1, bg_y1 = x1, y1 - th - baseline - 8
            bg_x2, bg_y2 = x1 + tw + 8, y1
            bg_y1 = max(bg_y1, 0)
            cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1)
            # 标签文字（与检测框同色 + 加粗）
            cv2.putText(out, label, (x1 + 4, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

    out_path = os.path.join(config.output_dir, f"{image_name}.jpg")
    cv2.imwrite(out_path, out)
