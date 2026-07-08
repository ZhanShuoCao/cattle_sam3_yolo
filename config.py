"""
项目配置模块。使用 dataclass + YAML 管理所有可配置参数。
尽量贴合项目现有配置体系，不引入复杂框架。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    """奶牛检测器配置（YOLO）"""
    model_path: str = ""                          # YOLO 模型权重路径（.pt），空字符串表示需要先训练
    conf_threshold: float = 0.35                  # 检测置信度阈值
    iou_threshold: float = 0.45                   # NMS IoU 阈值
    device: str = "cuda"                          # 推理设备
    imgsz: int = 640                              # 推理图像尺寸
    max_det: int = 20                             # 每张图最多检测数
    augment: bool = False                         # 是否启用测试时增强


@dataclass
class SAM3Config:
    """SAM3 分割模块配置"""
    model_type: str = "sam3"                      # "sam3" 或 "sam3.1"
    checkpoint_path: str = ""                     # 本地权重路径，空则自动从 HF 下载
    device: str = "cuda"
    # --- 提示模式 ---
    prompt_mode: str = "box"                      # "box" | "box+point" | "text"
    multimask_output: bool = False                # SAM3 是否输出多个 mask（单 box 场景通常 False）
    mask_threshold: float = 0.0                   # mask 二值化阈值


@dataclass
class VisualizationConfig:
    """可视化 Debug 配置"""
    enabled: bool = False
    output_dir: str = "./debug_vis"
    draw_box: bool = True
    draw_mask: bool = True
    save_format: str = "jpg"                       # jpg | png


@dataclass
class PipelineConfig:
    """主 Pipeline 配置，聚合所有子配置"""
    # --- 模式 ---
    detector_guided: bool = True                   # 是否启用 detector-guided segmentation

    # --- 路径 ---
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    sam3_config: SAM3Config = field(default_factory=SAM3Config)
    vis_config: VisualizationConfig = field(default_factory=VisualizationConfig)

    # --- 数据路径 ---
    detection_data_dir: str = ""                   # YOLO 检测数据集根目录
    segmentation_data_dir: str = ""                # LabelMe 分割数据集根目录
    output_dir: str = "./output"                   # 输出目录

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineConfig":
        """递归从字典构建配置对象"""
        def _populate(obj, d):
            for k, v in d.items():
                if hasattr(obj, k):
                    attr = getattr(obj, k)
                    if isinstance(attr, object) and hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                        _populate(attr, v)
                    else:
                        setattr(obj, k, v)

        cfg = cls()
        _populate(cfg, data)
        return cfg

    def to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, allow_unicode=True)
