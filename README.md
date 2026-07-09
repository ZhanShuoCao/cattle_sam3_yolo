# Cattle SAM3 — OWLv2 + SAM3 奶牛实例分割

基于 **OWLv2** 零样本检测 + **SAM3** 分割的牛场奶牛实例分割 pipeline。

## 流程

```
输入图像 → OWLv2 检测每头牛的 box → SAM3 box-prompt 分割 → Fragment Merge 合并遮挡碎片 → 输出 instance masks
```

## 环境

```bash
conda create -n omg python=3.10
conda activate omg
pip install torch transformers ultralytics gradio opencv-python pillow numpy scipy pyyaml
```

## 模型权重

| 模型 | 大小 | 说明 |
|------|------|------|
| `sam3.pt` | ~3.2 GB | SAM3 分割模型，需自行下载放到项目根目录 |
| `google/owlv2-large-patch14-finetuned` | ~2.5 GB | OWLv2 检测模型，首次运行自动从 HuggingFace 下载 |

## 配置

所有参数在 `config.yaml` 中调整，也可以通过 GUI 调节后保存。

## 使用

### CLI

```bash
# 单张推理
python run_inference.py --image test.jpg --debug

# 关闭 fragment merge
python run_inference.py --image test.jpg --debug --no-merge

# 切换检测器
python run_inference.py --image test.jpg --detector yolo

# 保存 mask
python run_inference.py --image test.jpg --save-mask
```

### GUI 调参

```bash
python gui_tuner.py
```

浏览器打开 `http://127.0.0.1:7860`，实时调节参数并预览效果，满意后点击「保存参数到配置文件」。

### 离线运行

```bash
# 模型已缓存到本地后，加这个跳过 HuggingFace 联网
HF_HUB_OFFLINE=1 python run_inference.py --image test.jpg --debug
```

## 文件结构

```
├── owl_detector.py              # OWLv2 零样本检测器
├── cow_segmentation_pipeline.py # 主 pipeline（检测→SAM3→合并）
├── fragment_merge.py            # 遮挡碎片合并
├── config.py / config.yaml      # 配置系统
├── run_inference.py             # CLI 推理入口
├── gui_tuner.py                 # Gradio GUI 调参面板
├── visualizer.py                # 可视化叠加图
└── sam3-main/                   # SAM3 模型源码
```
