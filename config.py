"""
项目配置模块 — config.py 只定义字段结构和兜底默认值。
日常调参请直接改 config.yaml 或用 gui_tuner.py。
"""

from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    """检测器配置（YOLO / OWLv2）。实际参数从 config.yaml 加载。"""
    detector_type: str = "owlv2"
    # YOLO
    model_path: str = ""
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cuda"
    imgsz: int = 640
    max_det: int = 20
    augment: bool = False
    # OWLv2
    owl_model_id: str = "google/owlv2-large-patch14-finetuned"
    owl_threshold: float = 0.3
    owl_text_prompt: str = "cow"
    owl_nms_iou: float = 0.0


@dataclass
class SAM3Config:
    """SAM3 分割模块配置。"""
    model_type: str = "sam3"
    checkpoint_path: str = ""
    device: str = "cuda"
    prompt_mode: str = "box"
    multimask_output: bool = False
    mask_threshold: float = 0.0


@dataclass
class FragmentMergeConfig:
    """遮挡碎片合并配置。"""
    enabled: bool = True
    box_expand_ratio: float = 0.0
    mask_conf_threshold: float = 0.3
    min_component_area: int = 150
    containment_threshold: float = 0.15
    iou_threshold: float = 0.05
    nms_threshold: float = 0.5
    box_dedup_enabled: bool = False
    box_dedup_iou: float = 0.15


@dataclass
class VisualizationConfig:
    """可视化 Debug 配置。"""
    enabled: bool = False
    output_dir: str = "./debug_vis"


@dataclass
class PipelineConfig:
    """主 Pipeline 配置，聚合所有子配置。"""
    detector_guided: bool = True
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    sam3_config: SAM3Config = field(default_factory=SAM3Config)
    fragment_merge: FragmentMergeConfig = field(default_factory=FragmentMergeConfig)
    vis_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    output_dir: str = "./output"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineConfig":
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
        import yaml
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, allow_unicode=True)
