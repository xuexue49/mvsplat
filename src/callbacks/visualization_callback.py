"""
可视化回调模块

在验证时自动保存对比图像，方便观察模型效果。
"""

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
import torch
from pathlib import Path

from ..dataset.types import BatchedExample
from ..misc.image_io import prep_image, save_image


class VisualizationCallback(Callback):
    """
    简单的可视化回调
    
    功能:
    - 在验证时保存输入图像、真值和预测结果的对比图
    - 图像保存到 outputs/visualizations/ 目录
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Args:
            output_dir: 可视化输出目录
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: BatchedExample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        验证批次结束时保存可视化
        
        保存格式:
        - step_{global_step}_batch_{batch_idx}_context.png: 输入视图
        - step_{global_step}_batch_{batch_idx}_gt.png: 真值
        - step_{global_step}_batch_{batch_idx}_pred.png: 预测结果
        """
        if outputs is None:
            return
        
        # 只保存前几个批次的可视化
        if batch_idx >= 3:
            return
        
        step = pl_module.global_step
        prefix = f"step_{step:06d}_batch_{batch_idx}"
        
        # 获取图像数据
        context_imgs = outputs.get("context_images")  # (V, C, H, W)
        rgb_gt = outputs.get("rgb_gt")                # (V, C, H, W)
        rgb_pred = outputs.get("rgb_pred")            # (V, C, H, W)
        
        if context_imgs is None or rgb_gt is None or rgb_pred is None:
            return
        
        # 保存第一个目标视角的对比图
        if rgb_gt.shape[0] > 0:
            # 保存输入视图 (第一个 context)
            if context_imgs.shape[0] > 0:
                save_image(
                    context_imgs[0],
                    self.output_dir / f"{prefix}_context.png"
                )
            
            # 保存真值
            save_image(
                rgb_gt[0],
                self.output_dir / f"{prefix}_gt.png"
            )
            
            # 保存预测
            save_image(
                rgb_pred[0],
                self.output_dir / f"{prefix}_pred.png"
            )
            
            # 创建并保存对比图 (context | gt | pred 横向拼接)
            comparison = self._create_comparison(
                context_imgs[0] if context_imgs.shape[0] > 0 else None,
                rgb_gt[0],
                rgb_pred[0]
            )
            if comparison is not None:
                save_image(
                    comparison,
                    self.output_dir / f"{prefix}_comparison.png"
                )
    
    def _create_comparison(
        self,
        context: torch.Tensor | None,
        gt: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        创建横向对比图
        
        Args:
            context: 输入图像 (C, H, W) 或 None
            gt: 真值图像 (C, H, W)
            pred: 预测图像 (C, H, W)
            
        Returns:
            comparison: 拼接后的对比图 (C, H, W*2 or W*3)
        """
        images = []
        
        if context is not None:
            # 确保尺寸一致
            if context.shape[-2:] != gt.shape[-2:]:
                context = torch.nn.functional.interpolate(
                    context.unsqueeze(0),
                    size=gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            images.append(context)
        
        images.extend([gt, pred])
        
        # 横向拼接
        return torch.cat(images, dim=-1)
