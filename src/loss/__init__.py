"""
损失函数模块

目前只保留 MSE (均方误差) 损失，用于图像重建。
"""

from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    """MSE 损失配置"""
    weight: float  # 损失权重


@dataclass
class LossMseCfgWrapper:
    """MSE 损失配置包装器 (用于 Hydra 配置)"""
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    """
    均方误差 (MSE) 损失
    
    计算渲染图像与真实图像的像素级 MSE 损失。
    
    公式: L = weight * mean((pred - gt)^2)
    """
    
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        """
        计算 MSE 损失
        
        Args:
            prediction: 解码器输出 (包含渲染图像)
            batch: 数据批次 (包含真实图像)
            gaussians: 3D Gaussians (未使用)
            global_step: 当前训练步数 (未使用)
            
        Returns:
            loss: 标量损失值
        """
        pred = prediction.color      # (B, V, C, H, W)
        gt = batch["target"]["image"] # (B, V, C, H, W)
        
        delta = pred - gt
        return self.cfg.weight * (delta ** 2).mean()


# 损失配置类型
LossCfgWrapper = LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    """
    根据配置创建损失函数列表
    
    Args:
        cfgs: 损失配置列表
        
    Returns:
        损失函数实例列表
    """
    losses = []
    for cfg in cfgs:
        if isinstance(cfg, LossMseCfgWrapper):
            losses.append(LossMse(cfg))
        else:
            raise NotImplementedError(f"未知的损失类型: {type(cfg)}")
    return losses
