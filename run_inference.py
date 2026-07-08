#!/usr/bin/env python
"""
命令行推理入口。

用法示例:
    # 单张图像推理
    python run_inference.py --image cow.jpg --config config.yaml

    # 批量推理
    python run_inference.py --dir ./test_images --config config.yaml

    # 开启可视化 debug
    python run_inference.py --image cow.jpg --config config.yaml --debug
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from cow_segmentation_pipeline import CowSegmentationPipeline
from visualizer import visualize_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector-Guided Cow Segmentation Inference"
    )
    # 输入
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, help="图像目录（批量推理）")
    parser.add_argument("--video", type=str, help="视频路径（实验性）")

    # 配置
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML 配置文件路径")

    # 覆盖参数
    parser.add_argument("--detector-model", type=str, default=None,
                        help="检测器模型路径（覆盖 config）")
    parser.add_argument("--sam3-checkpoint", type=str, default=None,
                        help="SAM3 checkpoint 路径（覆盖 config）")
    parser.add_argument("--conf", type=float, default=None,
                        help="检测置信度阈值")
    parser.add_argument("--no-detector-guided", action="store_true",
                        help="关闭 detector-guided 模式")
    parser.add_argument("--debug", action="store_true",
                        help="开启可视化 debug")

    # 输出
    parser.add_argument("--output", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--save-mask", action="store_true",
                        help="保存二值 mask 图像")

    return parser.parse_args()


def load_config(args) -> PipelineConfig:
    """加载配置，支持命令行覆盖。"""
    cfg_path = args.config
    if os.path.exists(cfg_path):
        config = PipelineConfig.from_yaml(cfg_path)
    else:
        print(f"[WARNING] 配置文件 {cfg_path} 不存在，使用默认配置")
        config = PipelineConfig()

    # 命令行覆盖
    if args.detector_model:
        config.detector_config.model_path = args.detector_model
    if args.sam3_checkpoint:
        config.sam3_config.checkpoint_path = args.sam3_checkpoint
    if args.conf is not None:
        config.detector_config.conf_threshold = args.conf
    if args.no_detector_guided:
        config.detector_guided = False
    if args.debug:
        config.vis_config.enabled = True
    if args.output:
        config.output_dir = args.output

    return config


def process_image(image_path: str, pipeline: CowSegmentationPipeline, args, config):
    """处理单张图像。"""
    print(f"\n处理: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: 无法读取 {image_path}")
        return

    result = pipeline.process(image)

    print(f"  检测到 {len(result.boxes)} 头牛")
    print(f"  分割出 {len(result.instances)} 个实例")
    print(f"  耗时: {result.elapsed_ms:.1f} ms")

    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    name = Path(image_path).stem

    if args.save_mask:
        for i, inst in enumerate(result.instances):
            mask_path = os.path.join(config.output_dir, f"{name}_cow_{i}.png")
            mask_img = (inst["mask"].astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_img)
        print(f"  已保存 {len(result.instances)} 个 mask 到 {config.output_dir}")

    # 可视化
    if config.vis_config.enabled:
        visualize_result(result, config.vis_config, image_name=name)
        print(f"  可视化结果已保存到 {config.vis_config.output_dir}")


def main():
    args = parse_args()
    config = load_config(args)

    # 初始化 pipeline
    print("=" * 50)
    print("初始化 Cow Segmentation Pipeline...")
    print(f"  Detector-guided: {config.detector_guided}")
    print(f"  Debug vis:        {config.vis_config.enabled}")
    print("=" * 50)

    pipeline = CowSegmentationPipeline(config)

    # 推理
    if args.image:
        process_image(args.image, pipeline, args, config)
    elif args.dir:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(
            f for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in image_exts
        )
        print(f"\n找到 {len(image_files)} 张图像")
        for fname in image_files:
            process_image(os.path.join(args.dir, fname), pipeline, args, config)
    elif args.video:
        print("[WARNING] 视频推理暂未实现，请逐帧提取后使用 --dir 模式")
    else:
        print("请指定 --image、--dir 或 --video。使用 --help 查看帮助。")

    print("\n完成。")


if __name__ == "__main__":
    main()
