"""
Cost Volume 编码器

这是 MVSplat 的核心模块，负责从多视图图像预测 3D Gaussians。

原理说明:
==========

1. Multi-View Transformer Backbone (多视图 Transformer 主干网络)
   - 输入: 多个视角的图像
   - 使用 cross-attention 在不同视角间交换信息
   - 输出: 每个视角的特征图

2. Cost Volume (代价体积)
   - 在多个深度假设上计算特征匹配代价
   - 通过比较不同视角在相同 3D 位置的特征相似度
   - 预测每个像素的深度分布

3. Gaussian Adapter (高斯适配器)
   - 将深度和特征转换为 3D Gaussian 参数
   - 预测: 位置(xyz), 协方差(covariance), 颜色(SH), 不透明度(opacity)

Pipeline:
    Images (B, V, 3, H, W)
         │
         ▼
    ┌──────────────────────┐
    │ Multi-View Backbone  │  特征提取 + 跨视角注意力
    └──────────────────────┘
         │
         ▼
    Features (B, V, C, H', W')
         │
         ▼
    ┌──────────────────────┐
    │    Depth Predictor   │  Cost Volume + U-Net
    └──────────────────────┘
         │
         ▼
    Depths + Raw Gaussians
         │
         ▼
    ┌──────────────────────┐
    │   Gaussian Adapter   │  参数转换
    └──────────────────────┘
         │
         ▼
    3D Gaussians (means, covariances, colors, opacities)
"""

from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import BackboneMultiview
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class OpacityMappingCfg:
    """不透明度映射配置 - 控制 PDF 到 opacity 的转换函数"""
    initial: float  # 初始指数
    final: float    # 最终指数
    warm_up: int    # 预热步数


@dataclass
class EncoderCostVolumeCfg:
    """Cost Volume 编码器配置"""
    name: Literal["costvolume"]
    
    # 特征维度
    d_feature: int              # Transformer 特征维度
    
    # 深度预测配置
    num_depth_candidates: int   # 深度候选数量
    num_surfaces: int           # 每个像素预测的表面数
    gaussians_per_pixel: int    # 每个像素的 Gaussian 数量
    
    # 可视化和适配器配置
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    
    # 预训练权重
    unimatch_weights_path: str | None
    
    # 下采样倍数
    downscale_factor: int
    shim_patch_size: int
    
    # Transformer 配置
    multiview_trans_attn_split: int
    
    # Cost Volume U-Net 配置
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    
    # Depth U-Net 配置
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    
    # 消融实验开关
    wo_depth_refine: bool           # 不使用深度精炼
    wo_cost_volume: bool            # 不使用 cost volume
    wo_backbone_cross_attn: bool    # 不使用跨视角注意力
    wo_cost_volume_refine: bool     # 不使用 cost volume 精炼
    use_epipolar_trans: bool        # 使用对极线 Transformer


# ============================================================================
# 主编码器类
# ============================================================================

class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    """
    Cost Volume 多视图编码器
    
    将多视图图像编码为 3D Gaussians，用于新视角合成。
    
    核心组件:
    - backbone: 多视图 Transformer，提取并融合多视角特征
    - depth_predictor: 基于 Cost Volume 的深度预测器
    - gaussian_adapter: 将预测转换为 3D Gaussian 参数
    """
    
    backbone: BackboneMultiview
    depth_predictor: DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg, cfg_dict: dict) -> None:
        """
        初始化编码器
        
        Args:
            cfg: 编码器配置
            cfg_dict: 完整配置字典 (用于获取数据集相关参数)
        """
        super().__init__(cfg)
        self.cfg_dict = cfg_dict
        
        # ===== 1. 可选的对极线采样器 =====
        if cfg.use_epipolar_trans:
            num_views = cfg_dict["dataset"]["view_sampler"]["num_context_views"]
            self.epipolar_sampler = EpipolarSampler(
                num_views=num_views,
                num_samples=32,
            )
            # 深度位置编码
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        
        # ===== 2. 多视图 Transformer 主干网络 =====
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        
        # 加载预训练权重 (仅训练模式)
        if cfg_dict["mode"] == 'train' and cfg.unimatch_weights_path is not None:
            self._load_backbone_weights(cfg.unimatch_weights_path)
        
        # ===== 3. Gaussian 参数适配器 =====
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        
        # ===== 4. 基于 Cost Volume 的深度预测器 =====
        num_views = cfg_dict["dataset"]["view_sampler"]["num_context_views"]
        gaussian_channels = cfg.num_surfaces * (self.gaussian_adapter.d_in + 2)
        
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=gaussian_channels,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=num_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )
    
    def _load_backbone_weights(self, ckpt_path: str) -> None:
        """加载 Backbone 预训练权重"""
        print(f"[Encoder] 加载预训练权重: {ckpt_path}")
        pretrained = torch.load(ckpt_path)["model"]
        
        # 只加载匹配的权重
        state_dict = OrderedDict({
            k: v for k, v in pretrained.items()
            if k in self.backbone.state_dict()
        })
        
        strict = not self.cfg.wo_backbone_cross_attn
        self.backbone.load_state_dict(state_dict, strict=strict)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        """
        将概率密度映射到不透明度
        
        使用 warm-up 策略逐渐调整映射函数:
        - 训练初期: 更平滑的映射
        - 训练后期: 更锐利的映射
        
        公式: opacity = 0.5 * (1 - (1-pdf)^exp + pdf^(1/exp))
        
        Args:
            pdf: 概率密度 [0, 1]
            global_step: 当前训练步数
            
        Returns:
            opacity: 不透明度 [0, 1]
        """
        cfg = self.cfg.opacity_mapping
        
        # 计算指数 (随训练进度增大)
        progress = min(global_step / cfg.warm_up, 1.0)
        x = cfg.initial + progress * (cfg.final - cfg.initial)
        exponent = 2 ** x
        
        # 映射公式
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        images: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        """
        前向推理: 从多视图图像预测 3D Gaussians
        
        Args:
            images: 输入图像 (B, V, 3, H, W)
            extrinsics: 相机外参 (B, V, 4, 4) - world-to-camera
            intrinsics: 相机内参 (B, V, 3, 3)
            near: 近平面深度 (B, V)
            far: 远平面深度 (B, V)
            global_step: 当前训练步数
            deterministic: 推理时使用确定性预测
            visualization_dump: 可视化输出字典 (可选)
            scene_names: 场景名称 (可选, 用于调试)
            
        Returns:
            gaussians: 3D Gaussian 参数
                - means: 中心位置 (B, N, 3)
                - covariances: 协方差矩阵 (B, N, 3, 3)
                - harmonics: 球谐系数 (B, N, 3, D_sh)
                - opacities: 不透明度 (B, N)
        """
        device = images.device
        b, v, _, h, w = images.shape

        # ===== Step 1: 多视图特征提取 =====
        epipolar_kwargs = None
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "near": near,
                "far": far,
            }
        
        trans_features, cnn_features = self.backbone(
            images,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # ===== Step 2: 深度预测 (Cost Volume) =====
        extra_info = {
            'images': rearrange(images, "b v c h w -> (v b) c h w"),
            'scene_names': scene_names,
        }
        
        depths, densities, raw_gaussians = self.depth_predictor(
            trans_features,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=self.cfg.gaussians_per_pixel,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )

        # ===== Step 3: 转换为 Gaussian 参数 =====
        # 采样像素坐标
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        
        # 解析原始 Gaussian 输出
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        
        # 提取并应用 xy 偏移
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        
        # 通过适配器转换参数
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
        )

        # ===== 可视化输出 (可选) =====
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # ===== 返回最终的 3D Gaussians =====
        return Gaussians(
            means=rearrange(
                gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"
            ),
            covariances=rearrange(
                gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"
            ),
            harmonics=rearrange(
                gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"
            ),
            opacities=rearrange(
                gaussians.opacities, "b v r srf spp -> b (v r srf spp)"
            ),
        )

    def get_data_shim(self) -> DataShim:
        """
        获取数据预处理函数
        
        应用 patch 剪裁以确保图像尺寸与下采样因子兼容
        """
        def data_shim(batch: BatchedExample) -> BatchedExample:
            return apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )
        return data_shim

    @property
    def sampler(self):
        """兼容可视化器的属性"""
        return None
