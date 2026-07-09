"""
Standalone OWL-V2 cow detector — extracted from the DazzleCowIdentifier pipeline.

Usage:
    python owl_detector.py --image_dir ./images --text_prompt "cow"
    python owl_detector.py --image_dir ./images --text_prompt "cow" --output json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# ---------------------------------------------------------------------------
# Detection helpers (inlined from utils.py / AutoMaskTrackPrompt2.py)
# ---------------------------------------------------------------------------

def preprocess_outputs(output: list[dict]) -> tuple:
    """Convert HuggingFace OWL output into (scores, labels, boxes)."""
    scores = [x["score"] for x in output]
    labels = [x["label"] for x in output]
    boxes = [[*x["box"].values()] for x in output]   # each: [xmin, ymin, xmax, ymax]
    return scores, labels, [boxes]


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(boxes: list, scores: list, iou_threshold: float = 0.5) -> tuple:
    """Non-maximum suppression — keep highest-score box per cluster."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [i for i in order if compute_iou(boxes[current], boxes[i]) <= iou_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def filter_by_area(outputs: list[dict], image_size: tuple,
                   min_ratio: float = 0.025, max_ratio: float = 0.075) -> list[dict]:
    """Discard bounding boxes whose area ratio falls outside [min_ratio, max_ratio]."""
    w, h = image_size
    frame_area = w * h
    kept = []
    for q in outputs:
        x1, y1, x2, y2 = q["box"].values()
        ratio = ((x2 - x1) * (y2 - y1)) / frame_area
        if min_ratio <= ratio <= max_ratio:
            kept.append(q)
    return kept


# ---------------------------------------------------------------------------
# Core detector class
# ---------------------------------------------------------------------------

class OwlDetector:
    """Thin wrapper around google/owlv2-large for zero-shot cow detection."""

    def __init__(
        self,
        model_id: str = "google/owlv2-large-patch14-finetuned",
        threshold: float = 0.3,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            model=model_id,
            task="zero-shot-object-detection",
            device=device,
            threshold=threshold,
        )

    def detect(
        self,
        image: Image.Image | str,
        text_prompt: str = "cow",
        min_area_ratio: float | None = None,
        max_area_ratio: float | None = None,
        nms_iou: float | None = 0.5,
    ) -> list[dict]:
        """
        Run detection on a single image.

        Returns a list of dicts, each containing:
            box      : [xmin, ymin, xmax, ymax]
            score    : float   (confidence)
        """
        if isinstance(image, str):
            image = Image.open(image)

        raw = self.pipe(image, candidate_labels=[text_prompt])

        # Optional area-ratio filter
        if min_area_ratio is not None and max_area_ratio is not None:
            raw = filter_by_area(raw, image.size, min_area_ratio, max_area_ratio)

        scores, labels, boxes_list = preprocess_outputs(raw)
        boxes = boxes_list[0]   # single-image format

        # Optional NMS
        if nms_iou is not None and len(boxes) > 1:
            boxes, scores = nms(boxes, scores, iou_threshold=nms_iou)

        return [
            {"box": [round(v, 1) for v in b], "score": round(s, 4)}
            for b, s in zip(boxes, scores)
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone OWL-V2 cow detector")
    parser.add_argument("--image_dir", required=True, help="Directory of images to process")
    parser.add_argument("--text_prompt", default="cow", help="OWL text prompt (default: 'cow')")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--min_area", type=float, default=0.025, help="Min bbox area ratio (0=off)")
    parser.add_argument("--max_area", type=float, default=0.075, help="Max bbox area ratio (0=off)")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold (0=off)")
    parser.add_argument("--output", choices=["json", "print"], default="print", help="Output format")
    parser.add_argument("--device", default=None, help="Torch device (auto-detect by default)")
    args = parser.parse_args()

    detector = OwlDetector(threshold=args.threshold, device=args.device)

    min_r = args.min_area if args.min_area > 0 else None
    max_r = args.max_area if args.max_area > 0 else None
    nms_val = args.nms if args.nms > 0 else None

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    results = {}

    for name in sorted(os.listdir(args.image_dir)):
        path = os.path.join(args.image_dir, name)
        if not os.path.isfile(path) or Path(name).suffix.lower() not in exts:
            continue
        dets = detector.detect(
            path, args.text_prompt,
            min_area_ratio=min_r, max_area_ratio=max_r, nms_iou=nms_val,
        )
        results[name] = dets

        if args.output == "print":
            print(f"\n{name}  ({len(dets)} detections)")
            for d in dets:
                print(f"  box={d['box']}  score={d['score']}")

    if args.output == "json":
        print(json.dumps(results, indent=2))

    total = sum(len(v) for v in results.values())
    print(f"\nDone. {total} detections across {len(results)} images.")


if __name__ == "__main__":
    main()
