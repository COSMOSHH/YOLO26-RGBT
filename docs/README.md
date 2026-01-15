# YOLO26-RGBT

本项目是基于 YOLO26 进行 RGBT 红外+可见光多模态目标检测的深度学习实现，参考 [YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT) 及相关论文/工程方案，支持多种融合方式、高效训练与推理，并适配新版 Ultralytics 框架。

---

## 项目简介

- **模型结构**：YOLO26 结合 RGBT MidFusion/ScoreFusion/FastFusion 等多模态融合模块，支持 4 通道（RGB+T）输入。
- **主要功能**：
  - 支持红外与可见光数据的多模态检测与训练
  - 可配置不同融合策略
  - 兼容多种主流多光谱/双光谱公开数据集（如 KAIST/LLVIP/M3FD 等）
  - 高效推理与验证接口，支持自动化实验和可视化

---

## 环境准备

```bash
# 推荐Python版本
conda create -n py311cu128 python=3.11
conda activate py311cu128

# 安装CUDA、PyTorch相关依赖
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装Ultralytics/YOLO
pip install ultralytics==8.4.1

# 安装其他依赖
pip install opencv-python tqdm
```

---

## 主要用法

### 1. 数据集准备

- 推荐数据格式：  
  ```
  datasets/
      rgbt3m_tinyfire_enhance/
          split_data/
              images/
                  visible/
                  infrared/
              labels/
                  visible/
                  infrared/
  ```
- DATA.yaml 示例见 `datasets/rgbt3m_tinyfire_enhance.yaml`，需配置 `names`, `train`, `val`, `channels` 等字段。

### 2. 训练

```bash
python my_train_RGBT.py
```

主要训练参数可在脚本中通过`channels`, `use_simotm`, `data`, `batch`, `epochs`等设置。

### 3. 验证与推理

```bash
python my_val_RGBT.py
```
结果保存在项目 `runs/` 目录下。

---

## 核心文件说明

- `my_train_RGBT.py` —— 训练入口脚本，集成 YOLO26-RGBT 各种参数和增强接口
- `cfg/yolo26-RGBT-midfusion.yaml` —— 模型配置文件，适用于 4 通道 RGBT 输入与 mid-fusion 结构
- `ultralytics/nn/modules/conv.py`、`block.py`、`tasks.py` —— 自定义 RGBT、融合模块及YOLO网络结构实现
- `ultralytics/data/base.py`、`augment.py` —— 数据读取、预处理与增强，支持多模态数据

---

## 注意事项和常见报错

- 若遇到“expected input to have X channels but got Y channels”等问题，需确保 `channels` 参数与数据实际通道、模型 yaml 的 `ch` 字段一致。
- 若遇到 `AttributeError: 'YOLODataset' object has no attribute 'load_and_preprocess_image'`，请在 `BaseDataset` 中补充该方法实现。
- 如需切换为 3/6/8+通道或灰度数据，参考 wandahangFY/YOLOv11-RGBT 仓库和 readme 说明调整参数和脚本。

---

## 致谢与参考

- [wandahangFY/YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- 相关论文及技术博客：https://arxiv.org/abs/2506.14696、知乎/CSDN 技术专栏

---

## 贡献与交流

如有问题或建议，欢迎提交 Issues 或 PR，或加入相关 RGBT/YOLOv11 多模态交流群。
