"""
MVSplat 模型包装器

封装 Encoder 和 Decoder，处理训练/验证/测试流程。

模型结构:
    Input Images (B, V, C, H, W)
         │
         ▼
    ┌─────────────┐
    │   Encoder   │  提取特征 → 构建 Cost Volume → 预测 Gaussians
    └─────────────┘
         │
         ▼
    Gaussians (means, covariances, colors, opacities)
         │
         ▼
    ┌─────────────┐
    │   Decoder   │  3D Gaussian Splatting 渲染
    └─────────────┘
         │
         ▼
    Output Image (B, V, C, H, W)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..loss import Loss
from ..misc.image_io import save_image
from ..misc.step_tracker import StepTracker

from .decoder.decoder import Decoder
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..config import OptimizerCfg, TestCfg, TrainCfg


class ModelWrapper(LightningModule):
    """
    MVSplat 模型封装类
    
    职责:
    1. 封装 Encoder + Decoder 的前向推理
    2. 管理训练/验证/测试流程
    3. 计算损失和评估指标
    4. 保存输出结果
    
    Attributes:
        encoder: 多视图编码器，输出 3D Gaussians
        decoder: Splatting 渲染器，输出目标视角图像
        losses: 损失函数列表
    """
    
    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        cfg_dict: dict,
    ) -> None:
        """
        初始化模型
        
        Args:
            optimizer_cfg: 优化器配置 (学习率、warmup 等)
            test_cfg: 测试配置 (输出路径、是否保存图像等)
            train_cfg: 训练配置 (深度模式、日志频率等)
            encoder: 编码器实例
            encoder_visualizer: 编码器可视化器 (可选)
            decoder: 解码器实例
            losses: 损失函数列表
            step_tracker: 训练步数追踪器
            cfg_dict: 完整配置字典
        """
        super().__init__()
        
        # 保存配置
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.cfg_dict = cfg_dict

        # 核心模型组件
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)
        
        # 数据预处理函数 (由 encoder 定义)
        self.data_shim = get_data_shim(self.encoder)
        
        # 测试指标存储
        self.test_step_outputs = {}

    def forward(
        self,
        context_images: Tensor,
        context_extrinsics: Tensor,
        context_intrinsics: Tensor,
        context_near: Tensor,
        context_far: Tensor,
        target_extrinsics: Tensor,
        target_intrinsics: Tensor,
        target_near: Tensor,
        target_far: Tensor,
        target_size: tuple[int, int],
    ) -> Tensor:
        """
        前向推理: 从 context 视图生成 target 视角的图像
        
        Pipeline:
            1. Encoder: context_images → 3D Gaussians
            2. Decoder: Gaussians + target_camera → rendered_images
        
        Args:
            context_images: 输入图像 (B, V_ctx, C, H, W)
            context_extrinsics: 输入相机外参 (B, V_ctx, 4, 4)
            context_intrinsics: 输入相机内参 (B, V_ctx, 3, 3)
            context_near/far: 深度范围
            target_extrinsics: 目标相机外参 (B, V_tgt, 4, 4)
            target_intrinsics: 目标相机内参 (B, V_tgt, 3, 3)
            target_near/far: 目标深度范围
            target_size: 输出图像尺寸 (H, W)
            
        Returns:
            rendered_images: 渲染结果 (B, V_tgt, C, H, W)
        """
        # Step 1: 编码 - 预测 3D Gaussians
        gaussians = self.encoder(
            images=context_images,
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            near=context_near,
            far=context_far,
            global_step=self.global_step,
            deterministic=True,
        )
        
        # Step 2: 解码 - 渲染目标视角
        output = self.decoder.forward(
            gaussians,
            target_extrinsics,
            target_intrinsics,
            target_near,
            target_far,
            target_size,
        )
        
        return output.color

    def training_step(self, batch: BatchedExample, batch_idx: int) -> Tensor:
        """
        训练步骤
        
        流程:
            1. 数据预处理
            2. 编码器: 输入图像 → 3D Gaussians
            3. 解码器: Gaussians → 渲染图像
            4. 计算损失 (MSE)
            5. 记录日志
        """
        batch = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # 编码: 预测 3D Gaussians
        gaussians = self.encoder(
            images=batch["context"]["image"],
            extrinsics=batch["context"]["extrinsics"],
            intrinsics=batch["context"]["intrinsics"],
            near=batch["context"]["near"],
            far=batch["context"]["far"],
            global_step=self.global_step,
            deterministic=False,
            scene_names=batch["scene"],
        )
        
        # 解码: 渲染目标视角
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        
        # 计算损失
        target_gt = batch["target"]["image"]
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)
        
        # 计算 PSNR 作为训练监控
        psnr = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr.mean())
        
        # 打印训练日志
        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            scene_names = [x[:20] for x in batch['scene']]
            print(f"[Step {self.global_step}] scene={scene_names}, loss={total_loss:.4f}, psnr={psnr.mean():.2f}")
        
        # 更新步数追踪器
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            
        self.log("info/global_step", self.global_step)
        return total_loss

    def validation_step(self, batch: BatchedExample, batch_idx: int) -> dict:
        """
        验证步骤
        
        Returns:
            包含预测结果和真值的字典，供可视化使用
        """
        batch = self.data_shim(batch)
        b, _, _, h, w = batch["target"]["image"].shape

        # 前向推理
        gaussians = self.encoder(
            images=batch["context"]["image"],
            extrinsics=batch["context"]["extrinsics"],
            intrinsics=batch["context"]["intrinsics"],
            near=batch["context"]["near"],
            far=batch["context"]["far"],
            global_step=self.global_step,
            deterministic=False,
        )
        
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        
        rgb_pred = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        
        # 计算验证指标
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        
        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.log("val/lpips", lpips)
        
        if self.global_rank == 0:
            print(f"[Val] scene={batch['scene'][0][:20]}, psnr={psnr:.2f}, ssim={ssim:.4f}")
        
        # 返回数据供可视化回调使用
        return {
            "gaussians": gaussians,
            "rgb_pred": rgb_pred,
            "rgb_gt": rgb_gt,
            "context_images": batch["context"]["image"][0],
        }

    def test_step(self, batch: BatchedExample, batch_idx: int) -> None:
        """
        测试步骤
        
        功能:
            1. 渲染测试视角
            2. 保存渲染结果图像
            3. 计算评估指标
        """
        batch = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape

        # 前向推理
        gaussians = self.encoder(
            images=batch["context"]["image"],
            extrinsics=batch["context"]["extrinsics"],
            intrinsics=batch["context"]["intrinsics"],
            near=batch["context"]["near"],
            far=batch["context"]["far"],
            global_step=self.global_step,
            deterministic=False,
        )
        
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        scene = batch["scene"][0]
        name = self.cfg_dict["wandb"]["name"]
        out_path = self.test_cfg.output_path / name
        
        rgb_pred = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # 保存图像
        if self.test_cfg.save_image:
            for idx, color in zip(batch["target"]["index"][0], rgb_pred):
                save_image(color, out_path / scene / f"color/{idx:06d}.png")

        # 计算并存储指标
        if self.test_cfg.compute_scores:
            for metric_name, metric_fn in [
                ("psnr", compute_psnr),
                ("ssim", compute_ssim),
                ("lpips", compute_lpips),
            ]:
                if metric_name not in self.test_step_outputs:
                    self.test_step_outputs[metric_name] = []
                self.test_step_outputs[metric_name].append(metric_fn(rgb_gt, rgb_pred).mean().item())

    def on_test_end(self) -> None:
        """测试结束时汇总并保存指标"""
        if not self.test_cfg.compute_scores:
            return
            
        name = self.cfg_dict["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for metric_name, scores in self.test_step_outputs.items():
            avg_score = sum(scores) / len(scores)
            results[metric_name] = avg_score
            print(f"[Test] {metric_name}: {avg_score:.4f}")
        
        # 保存结果
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[Test] 结果已保存到: {out_dir / 'results.json'}")

    def configure_optimizers(self) -> dict:
        """
        配置优化器和学习率调度器
        
        使用 Adam 优化器 + Cosine Annealing 学习率调度
        """
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        
        if self.optimizer_cfg.cosine_lr:
            # Cosine Annealing with Warm-up
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy='cos',
            )
        else:
            # Linear Warm-up
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / self.optimizer_cfg.warm_up_steps,
                end_factor=1,
                total_iters=self.optimizer_cfg.warm_up_steps,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
