from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from ..dataset.types import BatchedExample
from ..misc.image_io import prep_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections


class VisualizationCallback(Callback):
    """
    可视化回调，负责在验证步骤中生成和记录图像。
    
    将可视化逻辑从ModelWrapper中分离出来，遵循单一职责原则。
    """
    
    def __init__(self, extended_visualization: bool = False):
        """
        初始化可视化回调。
        
        Args:
            extended_visualization: 是否启用扩展可视化（额外的视频渲染）。
        """
        super().__init__()
        self.extended_visualization = extended_visualization
    
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
        验证批次结束时的回调。
        
        Args:
            trainer: PyTorch Lightning Trainer实例。
            pl_module: LightningModule实例（ModelWrapper）。
            outputs: 验证步骤的输出。
            batch: 验证数据批次。
            batch_idx: 批次索引。
            dataloader_idx: 数据加载器索引。
        """
        if outputs is None:
            return
            
        # Extract data from outputs
        gaussians_softmax = outputs["gaussians"]
        rgb_softmax = outputs["rgb_pred"]
        rgb_gt = outputs["rgb_gt"]
        
        # Construct comparison image
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        pl_module.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=pl_module.global_step,
            caption=batch["scene"],
        )
        
        # Render projections and construct projection image
        projections = hcat(*render_projections(
            gaussians_softmax,
            256,
            extra_label="(Softmax)",
        )[0])
        pl_module.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=pl_module.global_step,
        )
        
        # Draw cameras
        cameras = hcat(*render_cameras(batch, 256))
        pl_module.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=pl_module.global_step
        )
        
        # Encoder visualizations
        if hasattr(pl_module, 'encoder_visualizer') and pl_module.encoder_visualizer is not None:
            for k, image in pl_module.encoder_visualizer.visualize(
                batch["context"], pl_module.global_step
            ).items():
                pl_module.logger.log_image(k, [prep_image(image)], step=pl_module.global_step)
        
        # Run video validation step
        pl_module.render_video_interpolation(batch)
        pl_module.render_video_wobble(batch)
        if self.extended_visualization:
            pl_module.render_video_interpolation_exaggerated(batch)
