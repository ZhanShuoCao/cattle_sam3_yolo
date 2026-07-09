"""
Detector-Guided Cow Segmentation Pipeline.

Pipeline 流程：
    输入图像
    → 奶牛检测器输出每头牛的 box
    → 每个 box 作为 prompt 输入 SAM3 分割模块
    → 得到该 box 内的 mask 候选
    → 对同一 box 内的多个 mask component 进行合并
    → 输出最终 cow instance masks

设计原则：
- 最小侵入，通过接口适配现有模型
- 所有 SAM3/detector 依赖通过抽象接口注入
- 缺失模型时提供清晰 fallback，不崩溃
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from config import (
    DetectorConfig,
    FragmentMergeConfig,
    PipelineConfig,
    SAM3Config,
    VisualizationConfig,
)
from fragment_merge import MergeConfig, merge_fragments_by_box


# ═══════════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════════

class BaseDetector:
    """检测器抽象接口。可替换为 YOLO、RT-DETR 等任意实现。"""

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        检测图像中的奶牛。

        Args:
            image: (H, W, 3) BGR/RGB numpy array

        Returns:
            List[dict]: 每个元素含 {
                "bbox": np.ndarray (4,) xyxy,
                "score": float,
                "class_id": int,
            }
        """
        raise NotImplementedError


class BaseSegmenter:
    """
    分割器抽象接口。可替换为 SAM3、SAM2、MobileSAM 等任意实现。

    设计为两阶段接口以利用 SAM3 的 image embedding 缓存：
    1. set_image() — 预计算图像特征
    2. segment() — 对单个 box 生成 mask
    """

    def set_image(self, image: np.ndarray) -> None:
        """预计算图像特征（利用 SAM 的图像 embedding 缓存）。"""
        raise NotImplementedError

    def segment(
        self,
        box: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        对指定 box 区域进行分割。

        Args:
            box: (4,) xyxy 格式
            image: 可选，如果 set_image 未被调用则直接传入

        Returns:
            List[Dict]: 每个元素含 {
                "mask": np.ndarray (H,W) bool,
                "score": float,          # IoU 预测分数
                "bbox": np.ndarray,      # 预测的 mask bbox
            }
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
# YOLO 检测器实现
# ═══════════════════════════════════════════════════════════════════

class YOLODetector(BaseDetector):
    """
    基于 ultralytics YOLO 的奶牛检测器。

    支持：
    - 加载 .pt 权重进行推理
    - 自动处理 YOLO 输出格式
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载 YOLO 模型。"""
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            print(
                f"[YOLODetector] WARNING: 模型路径无效或不存在: {self.config.model_path}\n"
                f"  将使用 YOLO 预训练权重作为 fallback。"
            )
            # 使用 ultralytics 预训练模型作为 fallback
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolo11n.pt")  # 最小的预训练模型作为 fallback
                print("  → 已加载 yolo11n.pt（预训练权重，非牛场专用模型）")
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 YOLO 模型。请先训练奶牛检测模型或提供正确的模型路径。\n"
                    f"  训练命令参考: yolo train data=data.yaml model=yolo11n.pt epochs=100\n"
                    f"  原始错误: {e}"
                )
        else:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_path)
            print(f"[YOLODetector] 已加载: {self.config.model_path}")

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        YOLO 推理，返回检测框列表。

        Args:
            image: (H, W, 3) numpy array（BGR 或 RGB 均可，YOLO 内部处理）

        Returns:
            检测结果列表
        """
        results = self.model(
            image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
            augment=self.config.augment,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                detections.append({
                    "bbox": xyxy,
                    "score": conf,
                    "class_id": cls_id,
                })

        return detections


# ═══════════════════════════════════════════════════════════════════
# OWLv2 检测器实现
# ═══════════════════════════════════════════════════════════════════

class OwlV2Detector(BaseDetector):
    """
    基于 google/owlv2-large 的零样本奶牛检测器，适配 BaseDetector 接口。

    用法:
        detector = OwlV2Detector(config)
        detections = detector.detect(image)  # image 是 np.ndarray
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            from owl_detector import OwlDetector

            self._detector = OwlDetector(
                model_id=self.config.owl_model_id,
                threshold=self.config.owl_threshold,
                device=self.config.device,
            )
            print(
                f"[OwlV2Detector] 已加载: {self.config.owl_model_id}\n"
                f"  text_prompt='{self.config.owl_text_prompt}'  "
                f"threshold={self.config.owl_threshold}"
            )
        return self._detector

    def set_threshold(self, threshold: float) -> None:
        """动态更新 OWL 检测阈值（无需重建 pipeline）。"""
        if self._detector is not None:
            self._detector.pipe._postprocess_params["threshold"] = threshold

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        OWLv2 推理，返回检测框列表。

        Args:
            image: (H, W, 3) numpy array（BGR 或 RGB）

        Returns:
            List[dict]: 每个元素含 {"bbox": np.ndarray (4,) xyxy, "score": float, "class_id": int}
        """
        from PIL import Image

        # numpy → PIL（OWL pipeline 内部用 PIL）
        if image.shape[2] == 3:
            # BGR → RGB（cv2 读取的是 BGR）
            image_rgb = image[..., ::-1].copy()
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)

        # OWL 检测
        nms_iou = self.config.owl_nms_iou if self.config.owl_nms_iou > 0 else None
        detections_raw = self.detector.detect(
            pil_image,
            text_prompt=self.config.owl_text_prompt,
            nms_iou=nms_iou,
        )

        # 适配输出格式
        detections = []
        for d in detections_raw:
            box = d["box"]  # [xmin, ymin, xmax, ymax]
            detections.append({
                "bbox": np.array(box),
                "score": d["score"],
                "class_id": 0,  # OWL 是开放词汇检测，统一 class_id=0
            })
        return detections


# ═══════════════════════════════════════════════════════════════════
# SAM3 分割器实现
# ═══════════════════════════════════════════════════════════════════

class SAM3Segmenter(BaseSegmenter):
    """
    基于 Ultralytics SAM3 API 的分割器。

    使用 ultralytics 封装的 SAM 模型，支持 box prompt 分割。
    参考: https://docs.ultralytics.com/zh/models/sam-3/

    用法:
        model = SAM("sam3.pt")
        results = model.predict("image.jpg", bboxes=[x1, y1, x2, y2])
    """

    def __init__(self, config: SAM3Config):
        self.config = config
        self._model = None
        self._image = None  # 缓存当前图像路径/array
        self._is_available = False

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import SAM

            model_path = self.config.checkpoint_path or "sam3.pt"
            self._model = SAM(model_path)
            self._is_available = True
            print("[SAM3Segmenter] SAM3 已加载（Ultralytics SAM API）")

        except ImportError:
            print("[SAM3Segmenter] WARNING: ultralytics 未安装，使用 fallback")
            self._is_available = False
        except Exception as e:
            print(f"[SAM3Segmenter] WARNING: 加载 SAM3 失败 ({e})，使用 fallback")
            self._is_available = False

    def set_image(self, image: np.ndarray) -> None:
        self._ensure_model()
        self._image = image
        self._image_shape = image.shape[:2]
        # SAM.predict 需要文件路径，把 numpy array 存为临时文件
        import tempfile, cv2 as _cv2
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        _cv2.imwrite(self._tmpfile.name, image)
        self._tmpfile.close()

    def segment(
        self,
        box: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        if not self._is_available:
            return self._segment_fallback(box)
        return self._segment_with_sam3(box)

    def _segment_with_sam3(self, box: np.ndarray) -> List[Dict]:
        """使用 Ultralytics SAM API 进行 box-prompt 分割。"""
        bboxes = box.tolist()
        results = self._model.predict(
            source=self._tmpfile.name,
            bboxes=[bboxes],
            verbose=False,
        )
        all_results = []
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy().astype(bool)
                for i, mask in enumerate(masks):
                    if mask.sum() == 0:
                        continue
                    ys, xs = np.where(mask)
                    pred_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                    all_results.append({
                        "mask": mask,
                        "score": 1.0,
                        "bbox": pred_bbox,
                    })
        return all_results

    def segment_batch(self, boxes: List[np.ndarray]) -> List[List[Dict]]:
        """批量 box 分割。"""
        if not self._is_available or not boxes:
            return [self._segment_fallback(b) for b in boxes]

        bboxes = [b.tolist() for b in boxes]
        results = self._model.predict(
            source=self._tmpfile.name,
            bboxes=bboxes,
            verbose=False,
        )
        all_results = []
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy().astype(bool)
                box_results = []
                for mask in masks:
                    if mask.sum() == 0:
                        continue
                    ys, xs = np.where(mask)
                    pred_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                    box_results.append({"mask": mask, "score": 1.0, "bbox": pred_bbox})
                all_results.append(box_results)
            else:
                all_results.append([])
        return all_results

    def _segment_fallback(self, box: np.ndarray) -> List[Dict]:
        h, w = self._image_shape if self._image_shape else (2160, 3840)
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = box.astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return [{"mask": mask, "score": 0.0, "bbox": box.copy()}]


# ═══════════════════════════════════════════════════════════════════
# 主 Pipeline
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CowSegmentationResult:
    """单张图像的 pipeline 结果。"""
    image: np.ndarray                       # 原始图像 (H,W,3)
    instances: List[Dict]                   # 最终 instance masks 列表
    boxes: List[np.ndarray]                 # 检测框
    raw_masks: List[np.ndarray]             # 合并前的原始 mask
    num_merges: int                         # 发生合并的 instance 数
    elapsed_ms: float                       # 推理耗时 (ms)


class CowSegmentationPipeline:
    """
    Detector-Guided Cow Segmentation Pipeline.

    使用方式:
        pipeline = CowSegmentationPipeline(config)
        result = pipeline.process(image)

    Pipeline 流程:
        1. YOLO 检测 → cow boxes
        2. 每个 box 输入 SAM3 → mask 候选
        3. fragment merge → 最终 instance masks
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # 根据配置选择检测器
        detector_type = config.detector_config.detector_type.lower()
        if detector_type == "owlv2":
            self.detector: BaseDetector = OwlV2Detector(config.detector_config)
        else:
            self.detector: BaseDetector = YOLODetector(config.detector_config)
        self.segmenter: BaseSegmenter = SAM3Segmenter(config.sam3_config)

    # ---- 公开接口 ----

    def process(self, image: np.ndarray) -> CowSegmentationResult:
        """
        对单张图像执行完整 pipeline。

        Args:
            image: (H, W, 3) numpy array (BGR 或 RGB)

        Returns:
            CowSegmentationResult
        """
        t_start = time.time()

        # Step 1: 检测
        detections = self._detect(image)
        boxes = [d["bbox"] for d in detections]
        scores = [d["score"] for d in detections]

        # Step 2: SAM3 分割（每个 box 一个 prompt）
        raw_masks = self._segment_per_box(image, boxes)

        # Step 3: Fragment 合并
        merged = self._merge_fragments(raw_masks, boxes, scores, image)

        elapsed_ms = (time.time() - t_start) * 1000

        return CowSegmentationResult(
            image=image,
            instances=merged,
            boxes=boxes,
            raw_masks=raw_masks,
            num_merges=sum(1 for m in merged if m.get("was_merged", False)),
            elapsed_ms=elapsed_ms,
        )

    def process_batch(self, images: List[np.ndarray]) -> List[CowSegmentationResult]:
        """批量处理多张图像。"""
        return [self.process(img) for img in images]

    # ---- 内部步骤 ----

    def _detect(self, image: np.ndarray) -> List[dict]:
        """Step 1: 检测奶牛 + 去重。"""
        if not self.config.detector_guided:
            return []
        detections = self.detector.detect(image)
        if self.config.fragment_merge.box_dedup_enabled:
            detections = self._deduplicate_boxes(detections)
        return detections

    def _deduplicate_boxes(
        self, detections: List[dict], iou_threshold: float = 0.3
    ) -> List[dict]:
        """
        合并高度重叠的检测框（牛头+牛身 → 整牛）。

        策略：高分框优先，低分框若 IoU ≥ threshold 则与高分框取并集。
        """
        if len(detections) <= 1:
            return detections

        iou_threshold = self.config.fragment_merge.box_dedup_iou
        # 按 score 降序
        order = sorted(range(len(detections)), key=lambda i: detections[i]["score"], reverse=True)

        merged = []
        suppressed = set()

        for i in order:
            if i in suppressed:
                continue
            best = detections[i]
            best_box = best["bbox"].copy()

            # 找与当前框高度重叠的低分框，合并
            for j in order:
                if j <= i or j in suppressed:
                    continue
                box_j = detections[j]["bbox"]
                iou = self._box_iou(best_box, box_j)
                if iou >= iou_threshold:
                    # 取并集
                    best_box[0] = min(best_box[0], box_j[0])
                    best_box[1] = min(best_box[1], box_j[1])
                    best_box[2] = max(best_box[2], box_j[2])
                    best_box[3] = max(best_box[3], box_j[3])
                    suppressed.add(j)

            merged.append({
                "bbox": best_box,
                "score": best["score"],
                "class_id": best.get("class_id", 0),
            })

        return merged

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _segment_per_box(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Step 2: 对每个 box 调用 SAM3 分割（批量模式）。"""
        if not boxes:
            return []

        self.segmenter.set_image(image)

        # 使用批量 API（如果可用），否则逐个处理
        if hasattr(self.segmenter, 'segment_batch'):
            batch_results = self.segmenter.segment_batch(boxes)
            all_masks = []
            for results in batch_results:
                for r in results:
                    all_masks.append(r["mask"])
            return all_masks
        else:
            all_masks = []
            for box in boxes:
                seg_results = self.segmenter.segment(box, image)
                for r in seg_results:
                    all_masks.append(r["mask"])
            return all_masks

    def _merge_fragments(
        self,
        raw_masks: List[np.ndarray],
        boxes: List[np.ndarray],
        scores: List[float],
        image: np.ndarray,
    ) -> List[Dict]:
        """Step 3: 合并碎片化 mask。"""
        if not self.config.fragment_merge.enabled:
            # 不合并，直接返回 raw masks
            return [
                {"mask": m, "bbox": boxes[i] if i < len(boxes) else None,
                 "num_components": 1, "was_merged": False}
                for i, m in enumerate(raw_masks)
            ]

        merge_cfg = MergeConfig(
            box_expand_ratio=self.config.fragment_merge.box_expand_ratio,
            mask_conf_threshold=self.config.fragment_merge.mask_conf_threshold,
            min_component_area=self.config.fragment_merge.min_component_area,
            containment_threshold=self.config.fragment_merge.containment_threshold,
            iou_threshold=self.config.fragment_merge.iou_threshold,
            nms_threshold=self.config.fragment_merge.nms_threshold,
        )

        h, w = image.shape[:2]
        return merge_fragments_by_box(
            masks=raw_masks,
            boxes=boxes,
            scores=scores,
            config=merge_cfg,
            img_size=(w, h),
        )

    # ---- 便利方法 ----

    def get_instance_masks(self, result: CowSegmentationResult) -> List[np.ndarray]:
        """从结果中提取纯 mask 列表。"""
        return [inst["mask"] for inst in result.instances]

    def get_instance_boxes(self, result: CowSegmentationResult) -> List[np.ndarray]:
        """从结果中提取 bounding box 列表。"""
        boxes = []
        for inst in result.instances:
            mask = inst["mask"]
            if mask.sum() > 0:
                ys, xs = np.where(mask)
                boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()]))
            else:
                boxes.append(inst.get("bbox", np.zeros(4)))
        return boxes
