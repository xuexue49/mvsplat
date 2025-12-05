# MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

> ⚠️ **学习版说明**
> 
> 这是 MVSplat 的**简化学习版本**，为了方便理解模型核心逻辑，我们对代码进行了以下简化：
> 
> - ✅ **保留核心模块**：Encoder (Cost Volume), Decoder (CUDA Splatting), MSE Loss
> - ✅ **添加中文注释**：详细解释 Cost Volume、Gaussian Splatting 等关键概念
> - ✅ **简化训练流程**：移除复杂的视频渲染和多余的可视化
> - ✅ **最小可视化**：保留基本的图像对比保存功能
> 
> 原始完整版本请访问：[donydchen/mvsplat](https://github.com/donydchen/mvsplat)

---

## 核心架构

```
Input Images (B, V, 3, H, W)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Encoder (EncoderCostVolume)                    │
│  ├── Multi-View Transformer: 多视角特征融合      │
│  ├── Cost Volume: 深度假设匹配代价计算           │
│  └── Gaussian Adapter: 预测 3D Gaussian 参数    │
└─────────────────────────────────────────────────┘
      │
      ▼
3D Gaussians (means, covariances, colors, opacities)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Decoder (CUDA Splatting)                       │
│  └── 3D Gaussian Splatting 渲染目标视角          │
└─────────────────────────────────────────────────┘
      │
      ▼
Output Image (B, V, 3, H, W)
```

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `src/main.py` | 训练入口，包含完整流程注释 |
| `src/model/model_wrapper.py` | 模型封装，处理训练/验证/测试 |
| `src/model/encoder/encoder_costvolume.py` | **核心**：Cost Volume 编码器 |
| `src/model/decoder/cuda_splatting.py` | **核心**：CUDA Gaussian Splatting |
| `src/loss/__init__.py` | MSE 损失函数 |

## 快速开始

### 环境配置

```bash
conda activate mvsplat_medical
```

### 训练

```bash
# 下载预训练 backbone 权重
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints

# 开始训练
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

### 测试

```bash
python -m src.main +experiment=re10k \
  checkpointing.load=checkpoints/re10k.ckpt \
  mode=test \
  dataset/view_sampler=evaluation \
  test.compute_scores=true
```

## 原始项目信息

**论文**: [MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](https://arxiv.org/abs/2403.14627) (ECCV 2024 Oral)

**作者**: Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, Jianfei Cai

**引用**:
```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```

## 致谢

本项目基于 [pixelSplat](https://github.com/dcharatan/pixelsplat) 和 [UniMatch](https://github.com/autonomousvision/unimatch)。
