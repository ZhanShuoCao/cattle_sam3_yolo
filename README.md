# 🐄 Cow Segmentation Pipeline — Detector-Guided 奶牛实例分割

基于 **YOLO + SAM3** 的牛场工况奶牛实例分割方案。

## 核心思路

1. **先检测后分割**：YOLO 检测器先定位每头牛，输出检测框
2. **检测框引导 SAM3**：每个框作为 prompt 输入 SAM3 进行实例分割
3. **一个框 = 一头牛**：每个检测框对应一个 cow instance

## 项目结构

```
├── config.py                      # 配置 dataclass 定义
├── config.yaml                    # 默认配置文件
├── cow_segmentation_pipeline.py   # 主 Pipeline 模块（核心）
├── data_utils.py                  # 数据加载工具
├── visualizer.py                  # Debug 可视化
├── run_inference.py               # 推理命令行入口
├── sam3-main/                     # SAM3 源码
├── test.jpg                       # 测试图片
└── README.md
```

## 环境准备

```bash
# 1. 激活已有 conda 环境
conda activate omg

# 2. 安装依赖（如果缺失）
pip install ultralytics pycocotools scipy opencv-python pillow numpy pyyaml

# 3. 将 sam3-main 加入 PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD/sam3-main"
```

## 快速开始

### 推理

```bash
# 单张图像推理
python run_inference.py --image test.jpg --config config.yaml

# 开启 debug 可视化（输出 mask 叠加 + 检测框 + id 标签）
python run_inference.py --image test.jpg --config config.yaml --debug

# 批量推理
python run_inference.py --dir ./test_images --config config.yaml

# 使用自定义检测模型
python run_inference.py --image test.jpg --detector-model ./cow_yolo.pt
```

### 数据转换

```python
from data_utils import labelme_to_coco, yolo_dataset_to_coco

# LabelMe 分割数据 → COCO JSON
labelme_to_coco("D:/BaiduNetdiskDownload/分割数据集/已标注1-800",
                output_json="./seg_gt.json")

# YOLO 检测数据 → COCO JSON
yolo_dataset_to_coco("./train/images", "./train/labels", ["cattle"],
                     output_json="./det_train.json")
```

## 配置说明

编辑 `config.yaml`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `detector_guided` | 是否启用 detector-guided 模式 | `true` |
| `detector_config.model_path` | YOLO 权重路径（空=预训练fallback） | `""` |
| `detector_config.conf_threshold` | 检测置信度阈值 | `0.35` |
| `detector_config.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `sam3_config.checkpoint_path` | SAM3 权重路径 | `""` |
| `sam3_config.prompt_mode` | SAM3 提示模式 | `"box"` |
| `vis_config.enabled` | 是否保存 debug 可视化（命令行 `--debug` 会覆盖） | `false` |
| `vis_config.output_dir` | 可视化输出目录 | `"./debug_vis"` |

## API 使用

```python
from config import PipelineConfig
from cow_segmentation_pipeline import CowSegmentationPipeline
import cv2

# 加载配置
config = PipelineConfig.from_yaml("config.yaml")

# 初始化 pipeline
pipeline = CowSegmentationPipeline(config)

# 推理
image = cv2.imread("cow.jpg")
result = pipeline.process(image)

# 获取结果
for i, inst in enumerate(result.instances):
    mask = inst["mask"]   # binary mask (H, W)
    bbox = inst["bbox"]   # 检测框 (4,) xyxy
    print(f"Cow {i}: box={bbox}")

print(f"检测到 {len(result.boxes)} 头牛, 耗时 {result.elapsed_ms:.1f} ms")
```

## 常见问题

### Q: SAM3 无法加载怎么办？
1. 确保 `sam3-main` 在 PYTHONPATH 中
2. 确保已安装 `triton` 等 SAM3 依赖
3. 首次运行需联网从 HuggingFace 下载权重
4. Fallback 模式会使用 bbox mask，不影响代码运行

### Q: YOLO 模型不存在怎么办？
1. 在检测数据集上训练奶牛检测模型：
   ```bash
   yolo train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
   ```
2. 训练完成后将 `detector_config.model_path` 指向生成的 `.pt` 文件
3. 训练好之前，pipeline 使用预训练 YOLO 作为 fallback

### Q: 可视化没有检测框和 id 标签？
确保加了 `--debug` 参数。可视化输出在 `./debug_vis/` 目录下，包含：
- 彩色半透明 mask 覆盖
- 检测框（与 mask 同色）
- 框左上角白底黑框 `id: 0`、`id: 1` 标签
