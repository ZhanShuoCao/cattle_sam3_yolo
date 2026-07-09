"""
遮挡碎片合并模块。
负责将同一头牛被遮挡切开的多个 mask component 合并为一个实例。

核心原则：
- 一头牛 = 一个 instance，即使 mask 内部有多个不连通区域
- 不按连通域数量决定实例数量
- 以检测框为单位进行合并，不属于该框的 component 不合并
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as connected_components_label


@dataclass
class MergeConfig:
    """合并参数配置"""
    box_expand_ratio: float = 0.05
    mask_conf_threshold: float = 0.5
    min_component_area: int = 200
    containment_threshold: float = 0.3
    iou_threshold: float = 0.1
    nms_threshold: float = 0.5


def _expand_box(box: np.ndarray, ratio: float, img_w: int, img_h: int) -> np.ndarray:
    """扩张检测框，用于更完整地覆盖被遮挡牛。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dw, dh = w * ratio, h * ratio
    x1 = max(0, int(x1 - dw))
    y1 = max(0, int(y1 - dh))
    x2 = min(img_w, int(x2 + dw))
    y2 = min(img_h, int(y2 + dh))
    return np.array([x1, y1, x2, y2])


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 box (xyxy) 的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _compute_containment(mask: np.ndarray, box: np.ndarray) -> float:
    """
    计算 mask 像素有多少比例落在 box 内。
    containment = mask 在 box 内的像素数 / mask 总像素数
    """
    x1, y1, x2, y2 = box
    total = mask.sum()
    if total == 0:
        return 0.0
    # 裁剪 box 范围并计算内部像素
    h, w = mask.shape
    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(w, x2)
    y2_c = min(h, y2)
    inside = mask[y1_c:y2_c, x1_c:x2_c].sum()
    return float(inside) / float(total)


def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个 binary mask 的 IoU。"""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def extract_mask_components(
    mask: np.ndarray,
    min_area: int = 0,
) -> List[Tuple[np.ndarray, int]]:
    """
    从 mask 中提取各个连通域 component。

    Args:
        mask: (H, W) bool 或 int binary mask
        min_area: 最小面积过滤

    Returns:
        List[Tuple[np.ndarray, int]]: [(component_mask, area), ...] 按面积降序排列
    """
    labeled, num_features = connected_components_label(mask.astype(bool))
    components = []
    for i in range(1, num_features + 1):
        comp = (labeled == i)
        area = int(comp.sum())
        if area >= min_area:
            components.append((comp, area))
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def merge_fragments_by_box(
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    scores: Optional[List[float]] = None,
    config: Optional[MergeConfig] = None,
    img_size: Optional[Tuple[int, int]] = None,
) -> List[dict]:
    """
    将碎片化的 mask component 按检测框合并为每头牛一个 instance。

    核心逻辑：
    1. 对每个 cow box，收集主要位于该 box 内的 mask component
    2. 过滤：面积太小、containment 太低、或与其他 box 更匹配的 component
    3. 合并属于同一 cow box 的 component 为单个 binary mask
    4. 输出最终 instance masks

    Args:
        masks: 模型输出的 mask 候选列表，每个是 (H,W) bool/int 或分数 map
        boxes: 检测框列表，每个是 (4,) xyxy 格式
        scores: 可选，每个 mask 的置信度分数
        config: 合并参数
        img_size: 原始图像 (w, h)，用于扩张 box

    Returns:
        List[dict]: 每个元素含 {
            "mask": np.ndarray,     # 合并后的 instance mask (H,W) bool
            "bbox": np.ndarray,     # 对应检测框
            "num_components": int,  # 合并的 component 数量
            "was_merged": bool,     # 是否发生了合并
        }
    """
    if config is None:
        config = MergeConfig()

    h, w = None, None
    if masks:
        h, w = masks[0].shape[:2]
    if img_size is None and h is not None:
        img_size = (w, h)

    # ── 空输入保护 ──
    if len(boxes) == 0:
        return []
    if len(masks) == 0:
        # 没有 mask 输入时，返回空 mask
        return [
            {"mask": np.zeros((img_size[1], img_size[0]), dtype=bool),
             "bbox": box, "num_components": 0, "was_merged": False}
            for box in boxes
        ]

    # ── Step 1: 扩张检测框 ──
    expanded_boxes = [
        _expand_box(box, config.box_expand_ratio, img_size[0], img_size[1])
        for box in boxes
    ]

    # ── Step 2: 从所有 mask 中提取 component ──
    # 对每个 mask，在其对应的 dilated box 范围内提取 component，避免大图内存问题
    all_components: List[dict] = []
    # 将 mask 与最匹配的 expanded box 关联
    for i, mask in enumerate(masks):
        if mask.dtype != bool:
            mask = mask > config.mask_conf_threshold if config.mask_conf_threshold > 0 else mask.astype(bool)
        # 找到面积重叠最大的 box
        best_ebox = None
        best_overlap = 0
        for ebox in expanded_boxes:
            x1, y1, x2, y2 = ebox.astype(int)
            overlap = mask[y1:y2, x1:x2].sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_ebox = ebox
        # 在 best_ebox 范围内提取 component
        if best_ebox is not None and best_overlap > 0:
            x1, y1, x2, y2 = best_ebox.astype(int)
            crop = mask[y1:y2, x1:x2]
            comps = extract_mask_components(crop, min_area=config.min_component_area)
            for comp_mask, area in comps:
                full_comp = np.zeros_like(mask, dtype=bool)
                full_comp[y1:y2, x1:x2] = comp_mask
                ys, xs = np.where(full_comp)
                centroid = (float(xs.mean()), float(ys.mean()))
                all_components.append({
                    "mask": full_comp,
                    "area": area,
                    "centroid": centroid,
                    "score": scores[i] if scores else None,
                })

    if not all_components:
        return [
            {"mask": np.zeros((img_size[1], img_size[0]), dtype=bool),
             "bbox": box, "num_components": 0, "was_merged": False}
            for box in boxes
        ]

    # ── Step 3: 对每个 component 找最佳匹配 box ──
    # comp_to_best_box[i] = (box_idx, containment)
    comp_assignments: List[Tuple[int, float]] = []
    for comp in all_components:
        best_box_idx = -1
        best_containment = 0.0
        for j, ebox in enumerate(expanded_boxes):
            cont = _compute_containment(comp["mask"], ebox)
            if cont > best_containment:
                best_containment = cont
                best_box_idx = j
        comp_assignments.append((best_box_idx, best_containment))

    # ── Step 4: 分配 component 到各 box，并进行过滤 ──
    box_components: List[List[dict]] = [[] for _ in range(len(boxes))]
    for i, (box_idx, containment) in enumerate(comp_assignments):
        if box_idx < 0:
            continue
        if containment < config.containment_threshold:
            continue  # 与任何 box 都不够匹配，丢弃
        comp = all_components[i]

        # 检查是否与另一个 box 更匹配（防止把别的牛的碎片并进来）
        # 已在 assignment 阶段处理，此处检查 cross-box conflict
        conflict = False
        for j, other_box in enumerate(expanded_boxes):
            if j == box_idx:
                continue
            other_cont = _compute_containment(comp["mask"], other_box)
            # 如果这个 component 落入两个 box 且差异不大，分配给 containment 更高的
            if other_cont > containment and other_cont > config.containment_threshold:
                conflict = True
                # 重新分配给更匹配的 box
                box_idx = j
                containment = other_cont

        if not conflict:
            box_components[box_idx].append(comp)
        else:
            # 重新分配
            box_components[box_idx].append(comp)

    # ── Step 5: 每个 box 内合并 component ──
    results = []
    for i, (orig_box, comps) in enumerate(zip(boxes, box_components)):
        if not comps:
            results.append({
                "mask": np.zeros((img_size[1], img_size[0]), dtype=bool),
                "bbox": orig_box,
                "num_components": 0,
                "was_merged": False,
            })
        else:
            merged = np.zeros_like(comps[0]["mask"], dtype=bool)
            for c in comps:
                merged = np.logical_or(merged, c["mask"])
            results.append({
                "mask": merged,
                "bbox": orig_box,
                "num_components": len(comps),
                "was_merged": len(comps) > 1,
            })

    # ── Step 6: 跨 instance NMS（避免相邻牛 mask 重叠过多）──
    results = _apply_mask_nms(results, config.nms_threshold)

    return results


def _apply_mask_nms(
    results: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    对合并后的 instance mask 做 NMS。
    如果两 mask IoU 过高（可能合并了不属于该牛的成分），保留分数更高的。
    """
    if len(results) <= 1:
        return results

    # 用 mask 面积作为排序依据（优先保留大面积的）
    areas = [r["mask"].sum() for r in results]
    order = np.argsort(areas)[::-1]

    keep = []
    suppressed = set()
    for i in range(len(order)):
        idx_i = order[i]
        if idx_i in suppressed:
            continue
        keep.append(idx_i)
        for j in range(i + 1, len(order)):
            idx_j = order[j]
            if idx_j in suppressed:
                continue
            iou = _compute_mask_iou(results[idx_i]["mask"], results[idx_j]["mask"])
            if iou > iou_threshold:
                suppressed.add(idx_j)

    return [results[i] for i in sorted(keep)]
