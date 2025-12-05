"""
MVSplat 主入口文件

MVSplat 是一个基于 3D Gaussian Splatting 的多视图立体重建模型，
能够从少量输入视图生成新视角合成。

核心流程:
1. Encoder: 提取多视图特征，构建 Cost Volume，预测 3D Gaussian 参数
2. Decoder: 使用 CUDA Splatting 渲染目标视角图像
3. Loss: 计算渲染结果与真实图像的 MSE 损失
"""

import os
from pathlib import Path
import warnings

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def get_output_dir(cfg_dict: DictConfig) -> Path:
    """获取输出目录路径"""
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] 输出目录: {output_dir}")
    return output_dir


def create_callbacks(output_dir: Path, save_every_n_steps: int) -> list:
    """
    创建训练回调函数
    
    Args:
        output_dir: 输出目录
        save_every_n_steps: 每隔多少步保存检查点
        
    Returns:
        回调函数列表 [LearningRateMonitor, ModelCheckpoint, VisualizationCallback]
    """
    from src.callbacks import VisualizationCallback
    
    callbacks = [
        LearningRateMonitor("step", True),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,  # 保存所有检查点
            monitor="info/global_step",
            mode="max",
        ),
        VisualizationCallback(output_dir=str(output_dir / "visualizations")),
    ]
    
    # 修复检查点文件名中的特殊字符
    for cb in callbacks:
        if hasattr(cb, 'CHECKPOINT_EQUALS_CHAR'):
            cb.CHECKPOINT_EQUALS_CHAR = '_'
    
    return callbacks


def create_model(cfg, cfg_dict: dict, checkpoint_path: str = None) -> ModelWrapper:
    """
    创建 MVSplat 模型
    
    模型架构:
    - Encoder (EncoderCostVolume): 多视图特征提取 + Cost Volume + Gaussian 预测
    - Decoder (DecoderSplattingCUDA): CUDA 加速的 3D Gaussian Splatting 渲染
    
    Args:
        cfg: 类型化配置对象
        cfg_dict: 原始配置字典
        checkpoint_path: 预训练模型路径 (可选)
        
    Returns:
        ModelWrapper 实例
    """
    # 创建编码器: 负责从多视图图像预测 3D Gaussians
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, cfg_dict)
    
    # 创建解码器: 负责将 3D Gaussians 渲染成图像
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    
    # 创建损失函数
    losses = get_losses(cfg.loss)
    
    # 组装模型
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": decoder,
        "losses": losses,
        "step_tracker": StepTracker(),
        "cfg_dict": cfg_dict,
    }
    
    if checkpoint_path is not None:
        print(f"[INFO] 从检查点加载模型: {checkpoint_path}")
        model = ModelWrapper.load_from_checkpoint(checkpoint_path, **model_kwargs, strict=True)
    else:
        model = ModelWrapper(**model_kwargs)
    
    return model


def create_trainer(cfg, output_dir: Path, callbacks: list) -> Trainer:
    """
    创建 PyTorch Lightning Trainer
    
    Args:
        cfg: 配置对象
        output_dir: 输出目录
        callbacks: 回调函数列表
        
    Returns:
        Trainer 实例
    """
    # 根据 GPU 数量选择策略
    num_gpus = torch.cuda.device_count()
    strategy = "ddp" if num_gpus > 1 else "auto"
    
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=LocalLogger(),
        devices="auto",
        strategy=strategy,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        enable_progress_bar=(cfg.mode == "test"),
    )
    
    return trainer


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg_dict: DictConfig):
    """
    主训练/测试入口
    
    使用方式:
        训练: python -m src.main +experiment=re10k
        测试: python -m src.main +experiment=re10k mode=test checkpointing.load=<path>
    """
    # 1. 加载配置
    cfg = load_typed_root_config(cfg_dict)
    cfg_dict_container = OmegaConf.to_container(cfg_dict)
    
    # 2. 设置输出目录和随机种子
    output_dir = get_output_dir(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    
    # 3. 创建训练组件
    callbacks = create_callbacks(output_dir, cfg.checkpointing.every_n_train_steps)
    checkpoint_path = cfg_dict.checkpointing.load
    
    model = create_model(cfg, cfg_dict_container, 
                        checkpoint_path if cfg.mode == "train" and not cfg.checkpointing.resume else None)
    
    trainer = create_trainer(cfg, output_dir, callbacks)
    
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        model.step_tracker,
        global_rank=trainer.global_rank,
    )
    
    # 4. 开始训练或测试
    if cfg.mode == "train":
        print("[INFO] 开始训练...")
        trainer.fit(
            model, 
            datamodule=data_module, 
            ckpt_path=checkpoint_path if cfg.checkpointing.resume else None
        )
    else:
        print("[INFO] 开始测试...")
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    main()
