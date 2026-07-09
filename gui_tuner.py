"""
交互式参数调优面板 — 调整 OWL 检测 + SAM3 分割 + Fragment Merge 参数并实时预览。

用法:
    python gui_tuner.py
    python gui_tuner.py --port 7860 --share
"""

import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from config import PipelineConfig
from cow_segmentation_pipeline import CowSegmentationPipeline
from visualizer import visualize_result


# ═══════════════════════════════════════════════════════════════════
# Pipeline wrapper — 缓存中间结果以加速调参
# ═══════════════════════════════════════════════════════════════════

class CachedPipeline:
    """两阶段缓存：检测结果缓存 + mask 缓存，调参时避免重复计算。"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline = CowSegmentationPipeline(config)
        self._detections = None
        self._image = None
        self._raw_masks = None
        self._cache_key = None

    def detect_only(self, image: np.ndarray) -> list:
        """仅运行 OWL 检测（快速），缓存结果。"""
        self._image = image
        self._detections = self.pipeline.detector.detect(image)
        if self.config.fragment_merge.box_dedup_enabled:
            self._detections = self.pipeline._deduplicate_boxes(self._detections)
        self._raw_masks = None  # 清空 mask 缓存
        return self._detections

    def segment_and_merge(self) -> list:
        """用缓存的 detections 跑 SAM3 + merge。"""
        if self._detections is None:
            return []
        boxes = [d["bbox"] for d in self._detections]
        scores = [d["score"] for d in self._detections]
        raw_masks = self.pipeline._segment_per_box(self._image, boxes)
        self._raw_masks = raw_masks
        merged = self.pipeline._merge_fragments(raw_masks, boxes, scores, self._image)
        return merged

    def full_pipeline(self, image: np.ndarray):
        """完整 pipeline 并缓存所有中间结果。"""
        self._image = image
        self._detections = self.pipeline.detector.detect(image)
        if self.config.fragment_merge.box_dedup_enabled:
            self._detections = self.pipeline._deduplicate_boxes(self._detections)

        boxes = [d["bbox"] for d in self._detections]
        scores = [d["score"] for d in self._detections]

        self._raw_masks = self.pipeline._segment_per_box(image, boxes)
        merged = self.pipeline._merge_fragments(self._raw_masks, boxes, scores, image)
        return self._detections, merged

    def re_merge_only(self):
        """仅用缓存的 mask 重新 merge（极快）。"""
        if self._raw_masks is None or self._detections is None:
            return []
        boxes = [d["bbox"] for d in self._detections]
        scores = [d["score"] for d in self._detections]
        return self.pipeline._merge_fragments(self._raw_masks, boxes, scores, self._image)

    def draw_boxes(self, image: np.ndarray, detections: list) -> np.ndarray:
        """绘制检测框叠加图。"""
        from visualizer import _generate_colors
        out = image.copy()
        colors = _generate_colors(max(len(detections), 1))
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"].astype(int)
            c = colors[i % len(colors)]
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            cv2.putText(out, f"{d['score']:.2f}", (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return out

    def draw_instances(self, image: np.ndarray, instances: list) -> np.ndarray:
        """绘制 instance mask 叠加图。"""
        from visualizer import _generate_colors
        overlay = image.copy()
        colors = _generate_colors(max(len(instances), 1))
        for i, inst in enumerate(instances):
            mask = inst["mask"]
            if mask.sum() > 0:
                overlay[mask] = colors[i % len(colors)]
            bbox = inst.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
                cv2.putText(overlay, f"cow{i}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)
        out = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        return out


# ═══════════════════════════════════════════════════════════════════
# Session state — 每个用户一个
# ═══════════════════════════════════════════════════════════════════

class SessionState:
    def __init__(self):
        self.cached: CachedPipeline | None = None
        self.current_image: np.ndarray | None = None
        self.current_mask_cache: list | None = None


# ═══════════════════════════════════════════════════════════════════
# Gradio 回调
# ═══════════════════════════════════════════════════════════════════

def to_bgr(img: np.ndarray) -> np.ndarray:
    """将 RGB numpy 转为 BGR（管线内部统一用 BGR）。"""
    if img.shape[2] == 3:
        return img[..., ::-1].copy()
    return img


def to_rgb(img: np.ndarray) -> np.ndarray:
    """将 BGR numpy 转为 RGB（Gradio 显示需要 RGB）。"""
    if img.shape[2] == 3:
        return img[..., ::-1].copy()
    return img


def on_image_upload(image):
    if image is None:
        return image, "请上传图片"
    arr = np.array(image)
    h, w = arr.shape[:2]
    return image, f"已加载 {w}×{h}  |  点击「运行检测+分割」开始"


def on_run_detection(
    image,
    owl_threshold, owl_text_prompt, owl_nms_iou,
    box_dedup_enabled, box_dedup_iou,
    state: SessionState,
):
    if image is None:
        return None, None, "❌ 请先上传图片", state

    t0 = time.time()
    arr = to_bgr(np.array(image))                     # RGB→BGR
    state.current_image = arr

    # 更新配置
    state.cached.pipeline.config.detector_config.owl_threshold = owl_threshold
    state.cached.pipeline.config.detector_config.owl_text_prompt = owl_text_prompt
    state.cached.pipeline.config.detector_config.owl_nms_iou = owl_nms_iou
    state.cached.pipeline.config.fragment_merge.box_dedup_enabled = box_dedup_enabled
    state.cached.pipeline.config.fragment_merge.box_dedup_iou = box_dedup_iou

    state.cached.pipeline.detector.set_threshold(owl_threshold)

    detections = state.cached.detect_only(arr)
    t = (time.time() - t0) * 1000

    if not detections:
        return image, image, f"⚠️ 未检测到任何目标 ({t:.0f} ms)", state

    box_img = state.cached.draw_boxes(arr, detections)
    msg = f"✅ 检测到 {len(detections)} 个目标 ({t:.0f} ms)  |  点击「运行分割」继续"
    return to_rgb(box_img), image, msg, state


def on_run_segmentation(
    owl_threshold, owl_text_prompt, owl_nms_iou,
    box_dedup_enabled, box_dedup_iou,
    merge_enabled, box_expand_ratio, mask_conf_threshold,
    min_component_area, containment_threshold, iou_threshold, nms_threshold,
    state: SessionState,
):
    if state.current_image is None:
        return None, "❌ 请先上传图片并运行检测", state

    t0 = time.time()

    # 更新 merge 参数
    cfg = state.cached.pipeline.config
    cfg.fragment_merge.box_expand_ratio = box_expand_ratio
    cfg.fragment_merge.mask_conf_threshold = mask_conf_threshold
    cfg.fragment_merge.min_component_area = int(min_component_area)
    cfg.fragment_merge.containment_threshold = containment_threshold
    cfg.fragment_merge.iou_threshold = iou_threshold
    cfg.fragment_merge.nms_threshold = nms_threshold
    cfg.fragment_merge.enabled = merge_enabled

    instances = state.cached.segment_and_merge()
    state.current_mask_cache = instances
    t = (time.time() - t0) * 1000

    if not instances:
        return to_rgb(state.cached.draw_boxes(state.current_image, state.cached._detections)), \
               f"⚠️ SAM3 未生成 mask ({t:.0f} ms)", state

    merged_count = sum(1 for inst in instances if inst.get("was_merged", False))
    out_img = state.cached.draw_instances(state.current_image, instances)
    msg = f"✅ {len(instances)} 个实例, {merged_count} 被合并  |  耗时 {t/1000:.1f}s"
    return to_rgb(out_img), msg, state


def on_remerge(
    merge_enabled, box_expand_ratio, mask_conf_threshold,
    min_component_area, containment_threshold, iou_threshold, nms_threshold,
    state: SessionState,
):
    """仅重新 merge，不重新跑 SAM3（快）。"""
    if state.cached is None or state.cached._raw_masks is None:
        return None, "⚠️ 请先运行分割", state

    t0 = time.time()

    cfg = state.cached.pipeline.config
    cfg.fragment_merge.box_expand_ratio = box_expand_ratio
    cfg.fragment_merge.mask_conf_threshold = mask_conf_threshold
    cfg.fragment_merge.min_component_area = int(min_component_area)
    cfg.fragment_merge.containment_threshold = containment_threshold
    cfg.fragment_merge.iou_threshold = iou_threshold
    cfg.fragment_merge.nms_threshold = nms_threshold
    cfg.fragment_merge.enabled = merge_enabled

    instances = state.cached.re_merge_only()
    t = (time.time() - t0) * 1000

    merged_count = sum(1 for inst in instances if inst.get("was_merged", False))
    out_img = state.cached.draw_instances(state.current_image, instances)
    msg = f"🔄 重新合并: {len(instances)} 实例, {merged_count} 合并  |  {t:.0f} ms"
    return out_img, msg, state


def on_full_run(
    image,
    owl_threshold, owl_text_prompt, owl_nms_iou,
    box_dedup_enabled, box_dedup_iou,
    merge_enabled, box_expand_ratio, mask_conf_threshold,
    min_component_area, containment_threshold, iou_threshold, nms_threshold,
    state: SessionState,
):
    """一键运行完整 pipeline。"""
    if image is None:
        return None, "❌ 请先上传图片", state

    t0 = time.time()
    arr = np.array(image)
    state.current_image = arr

    # 更新所有配置
    cfg = state.cached.pipeline.config
    cfg.detector_config.owl_threshold = owl_threshold
    cfg.detector_config.owl_text_prompt = owl_text_prompt
    cfg.detector_config.owl_nms_iou = owl_nms_iou
    cfg.fragment_merge.box_dedup_enabled = box_dedup_enabled
    cfg.fragment_merge.box_dedup_iou = box_dedup_iou
    cfg.fragment_merge.box_expand_ratio = box_expand_ratio
    cfg.fragment_merge.mask_conf_threshold = mask_conf_threshold
    cfg.fragment_merge.min_component_area = int(min_component_area)
    cfg.fragment_merge.containment_threshold = containment_threshold
    cfg.fragment_merge.iou_threshold = iou_threshold
    cfg.fragment_merge.nms_threshold = nms_threshold
    cfg.fragment_merge.enabled = merge_enabled

    # 更新 OWL threshold (直接影响 pipeline 内部)
    state.cached.pipeline.detector.set_threshold(owl_threshold)

    detections, instances = state.cached.full_pipeline(arr)
    t = (time.time() - t0) * 1000

    merged_count = sum(1 for inst in instances if inst.get("was_merged", False))
    out_img = state.cached.draw_instances(arr, instances)
    msg = (f"✅ 检测: {len(detections)} 框  →  {len(instances)} 实例"
           f"  |  合并: {merged_count}  |  耗时 {t/1000:.1f}s")
    return out_img, msg, state


def on_save_config(
    owl_threshold, owl_text_prompt, owl_nms_iou,
    box_dedup_enabled, box_dedup_iou,
    merge_enabled, box_expand_ratio, mask_conf_threshold,
    min_component_area, containment_threshold, iou_threshold, nms_threshold,
):
    """将所有 GUI 参数写回 config.yaml。"""
    from config import PipelineConfig
    import yaml

    path = "config.yaml"
    # 读现有配置（保留 YAML 原有结构和注释）
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dc = data.setdefault("detector_config", {})
    dc["owl_threshold"] = owl_threshold
    dc["owl_text_prompt"] = owl_text_prompt
    dc["owl_nms_iou"] = owl_nms_iou

    fm = data.setdefault("fragment_merge", {})
    fm["enabled"] = merge_enabled
    fm["box_dedup_enabled"] = box_dedup_enabled
    fm["box_dedup_iou"] = box_dedup_iou
    fm["box_expand_ratio"] = box_expand_ratio
    fm["mask_conf_threshold"] = mask_conf_threshold
    fm["min_component_area"] = int(min_component_area)
    fm["containment_threshold"] = containment_threshold
    fm["iou_threshold"] = iou_threshold
    fm["nms_threshold"] = nms_threshold

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return "💾 参数已保存到 config.yaml  |  下次 python run_inference.py 直接生效"


# ═══════════════════════════════════════════════════════════════════
# UI 构建
# ═══════════════════════════════════════════════════════════════════

def build_ui():
    # 从 config.yaml 读取默认值，确保 GUI 和 CLI 参数一致
    cfg = PipelineConfig.from_yaml("config.yaml")
    dc = cfg.detector_config
    fm = cfg.fragment_merge

    theme = gr.themes.Soft(primary_hue="orange")

    with gr.Blocks(title="Cow Segmentation Tuner") as app:
        state = gr.State(SessionState())

        gr.Markdown("# 🐄 Cow Segmentation Pipeline Tuner")
        gr.Markdown("OWLv2 检测 + SAM3 分割 + Fragment Merge 参数调节")

        with gr.Row():
            # ── 左侧：输入 ──
            with gr.Column(scale=1):
                gr.Markdown("### 📷 输入")
                input_image = gr.Image(type="pil", label="上传图片", height=300)

                gr.Markdown("### 🎯 OWLv2 检测")
                owl_threshold = gr.Slider(0.1, 0.9, value=dc.owl_threshold, step=0.05, label="检测阈值")
                owl_text_prompt = gr.Textbox(value=dc.owl_text_prompt, label="文本提示")
                owl_nms_iou = gr.Slider(0.0, 1.0, value=dc.owl_nms_iou, step=0.05, label="OWL NMS IoU (0=禁用)")
                box_dedup_enabled = gr.Checkbox(value=fm.box_dedup_enabled, label="Box 去重 (合并牛头+牛身)")
                box_dedup_iou = gr.Slider(0.05, 0.8, value=fm.box_dedup_iou, step=0.05, label="Box 去重 IoU")

                gr.Markdown("### 🧩 Fragment Merge")
                merge_enabled = gr.Checkbox(value=fm.enabled, label="启用 Fragment Merge")
                box_expand_ratio = gr.Slider(0.0, 0.2, value=fm.box_expand_ratio, step=0.01, label="Box 扩张比例")
                mask_conf_threshold = gr.Slider(0.0, 1.0, value=fm.mask_conf_threshold, step=0.05, label="Mask 置信度阈值")
                min_component_area = gr.Slider(0, 1000, value=fm.min_component_area, step=50, label="最小 component 面积")
                containment_threshold = gr.Slider(0.0, 0.8, value=fm.containment_threshold, step=0.05, label="Containment 阈值")
                iou_threshold = gr.Slider(0.0, 0.5, value=fm.iou_threshold, step=0.05, label="Component-Box IoU 阈值")
                nms_threshold = gr.Slider(0.0, 1.0, value=fm.nms_threshold, step=0.05, label="Mask NMS 阈值")

                with gr.Row():
                    btn_detect = gr.Button("🔍 1. 运行检测", variant="secondary", scale=1)
                    btn_segment = gr.Button("🧬 2. 运行分割", variant="secondary", scale=1)
                btn_full = gr.Button("⚡ 一键运行全部", variant="primary")
                btn_remerge = gr.Button("🔄 仅重新合并 (不跑SAM3)", variant="secondary", size="sm")
                btn_save = gr.Button("💾 保存参数到配置文件", variant="stop", size="sm")
                status_save = gr.Markdown("")

            # ── 右侧：输出 ──
            with gr.Column(scale=2):
                status = gr.Markdown("上传图片后点击按钮开始")

                with gr.Tabs():
                    with gr.Tab("分割结果"):
                        output_seg = gr.Image(type="numpy", label="Instance Masks", height=600)
                    with gr.Tab("检测框"):
                        output_box = gr.Image(type="numpy", label="Detection Boxes", height=600)

        # ── 事件绑定 ──

        # Pipeline 初始化
        def init_pipeline(state: SessionState):
            cfg = PipelineConfig.from_yaml("config.yaml")
            state.cached = CachedPipeline(cfg)
            return state

        app.load(fn=init_pipeline, inputs=[state], outputs=[state])

        # 上传图片
        input_image.change(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[input_image, status],
        )

        # 仅检测
        detect_inputs = [
            input_image,
            owl_threshold, owl_text_prompt, owl_nms_iou,
            box_dedup_enabled, box_dedup_iou,
            state,
        ]
        btn_detect.click(
            fn=on_run_detection,
            inputs=detect_inputs,
            outputs=[output_box, output_seg, status, state],
        )

        # 检测 + SAM3
        segment_inputs = [
            owl_threshold, owl_text_prompt, owl_nms_iou,
            box_dedup_enabled, box_dedup_iou,
            merge_enabled, box_expand_ratio, mask_conf_threshold,
            min_component_area, containment_threshold, iou_threshold, nms_threshold,
            state,
        ]
        btn_segment.click(
            fn=on_run_segmentation,
            inputs=segment_inputs,
            outputs=[output_seg, status, state],
        )

        # 一键运行
        full_inputs = [
            input_image,
            owl_threshold, owl_text_prompt, owl_nms_iou,
            box_dedup_enabled, box_dedup_iou,
            merge_enabled, box_expand_ratio, mask_conf_threshold,
            min_component_area, containment_threshold, iou_threshold, nms_threshold,
            state,
        ]
        btn_full.click(
            fn=on_full_run,
            inputs=full_inputs,
            outputs=[output_seg, status, state],
        )

        # 仅重新 merge
        remerge_inputs = [
            merge_enabled, box_expand_ratio, mask_conf_threshold,
            min_component_area, containment_threshold, iou_threshold, nms_threshold,
            state,
        ]
        btn_remerge.click(
            fn=on_remerge,
            inputs=remerge_inputs,
            outputs=[output_seg, status, state],
        )

        # 保存配置
        save_inputs = [
            owl_threshold, owl_text_prompt, owl_nms_iou,
            box_dedup_enabled, box_dedup_iou,
            merge_enabled, box_expand_ratio, mask_conf_threshold,
            min_component_area, containment_threshold, iou_threshold, nms_threshold,
        ]
        btn_save.click(
            fn=on_save_config,
            inputs=save_inputs,
            outputs=[status_save],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Cow Segmentation Parameter Tuner")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="127.0.0.1",
    )


if __name__ == "__main__":
    main()
