#!/usr/bin/env python3
"""
Debug training script for fast iteration and testing.
Uses dummy data and minimal training steps to verify the pipeline works.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import warnings

import torch
from colorama import Fore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import Logger

# Standard imports
from src.dataset.config import DatasetCfg, DataLoaderCfg, DataLoaderStageCfg
from src.dataset.data_module import DataModule
from src.factory import get_decoder, get_encoder, get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.model.model_wrapper import ModelWrapper, OptimizerCfg, TestCfg, TrainCfg
from src.model.encoder.encoder_costvolume import EncoderCostVolumeCfg, OpacityMappingCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.decoder.decoder import DepthRenderingMode
from src.loss.loss_mse import LossMseCfg, LossMseCfgWrapper
from src.loss.loss_depth import LossDepthCfg, LossDepthCfgWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@dataclass
class DebugConfig:
    """Debug configuration for fast testing with dummy data."""
    
    # Dataset configuration - USING DUMMY DATA
    dataset_path: str = "data/dummy_medical"
    image_shape: list[int] = field(default_factory=lambda: [256, 256])  # Smaller for speed
    num_context_views: int = 2
    
    # Data loader configuration - MINIMAL SETTINGS
    batch_size_train: int = 1
    batch_size_val: int = 1
    batch_size_test: int = 1
    num_workers: int = 0  # Avoid multiprocessing issues during debug
    persistent_workers: bool = False  # Must be False when num_workers=0
    seed: int = 42
    
    # Training configuration - FAST DEV RUN
    mode: Literal["train", "test"] = "train"
    max_steps: int = 10  # Only 10 steps for debugging
    val_check_interval: int = 5  # Validate after 5 steps
    gradient_clip_val: float = 1.0
    num_sanity_val_steps: int = 0
    num_nodes: int = 1
    
    # Optimizer configuration
    lr: float = 3e-4
    warm_up_steps: int = 5  # Reduced for debugging
    cosine_lr: bool = True
    
    # Checkpointing configuration
    checkpoints_dir: str = "outputs/debug_checkpoints"
    every_n_train_steps: int = 5
    save_top_k: int = 1  # Only keep 1 checkpoint
    load_checkpoint: str | None = None
    resume_training: bool = False
    
    # Output configuration
    output_dir: str | None = None
    experiment_name: str = "debug-dry-run"
    
    # Test configuration
    test_output_path: str = "outputs/debug_test_results"
    compute_scores: bool = False  # Skip score computation for speed
    save_image: bool = True
    save_video: bool = False
    eval_time_skip_steps: int = 1
    
    # Train visualization configuration
    depth_mode: DepthRenderingMode | None = "depth"
    extended_visualization: bool = False
    print_log_every_n_steps: int = 1  # Print every step for debugging
    
    # Model configuration - Encoder (reduced complexity)
    encoder_name: Literal["costvolume"] = "costvolume"
    d_feature: int = 32  # Reduced from 64
    num_depth_candidates: int = 32  # Reduced from 64
    num_surfaces: int = 2
    gaussians_per_pixel: int = 1
    unimatch_weights_path: str | None = None
    downscale_factor: int = 4
    shim_patch_size: int = 7
    multiview_trans_attn_split: int = 1
    costvolume_unet_feat_dim: int = 32  # Reduced from 64
    costvolume_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4])
    costvolume_unet_attn_res: list[int] = field(default_factory=lambda: [])
    depth_unet_feat_dim: int = 32  # Reduced from 64
    depth_unet_attn_res: list[int] = field(default_factory=lambda: [])
    depth_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4])
    wo_depth_refine: bool = False
    wo_cost_volume: bool = False
    wo_backbone_cross_attn: bool = False
    wo_cost_volume_refine: bool = False
    use_epipolar_trans: bool = True
    
    # Gaussian adapter configuration
    gaussian_scale_min: float = 0.0
    gaussian_scale_max: float = 1.5
    sh_degree: int = 0
    
    # Opacity mapping configuration
    opacity_initial: float = 0.5
    opacity_final: float = 1.0
    opacity_warm_up: int = 5  # Reduced for debugging
    
    # Encoder visualizer configuration
    visualizer_num_samples: int = 64  # Reduced from 128
    visualizer_min_resolution: int = 64
    visualizer_export_ply: bool = False
    
    # Decoder configuration
    decoder_name: Literal["splatting_cuda"] = "splatting_cuda"
    
    # Loss configuration
    loss_mse_weight: float = 1.0
    loss_depth_weight: float = 0.0
    loss_depth_sigma_image: float | None = None
    loss_depth_use_second_derivative: bool = False


def setup_infrastructure(cfg: DebugConfig) -> tuple[Logger, list]:
    """Set up output directories, logging, and callbacks."""
    
    # Set up the output directory
    if cfg.output_dir is None:
        output_dir = Path(cfg.checkpoints_dir).parent / "debug_runs" / f"run_{cfg.experiment_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(cyan(f"[DEBUG] Saving outputs to {output_dir}."))
    
    # Set up logging with local logger only
    callbacks = []
    logger = LocalLogger()
    print(cyan("[DEBUG] Using LocalLogger for experiment tracking."))
    
    # Set up checkpointing
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(
        ModelCheckpoint(
            checkpoint_dir,
            every_n_train_steps=cfg.every_n_train_steps,
            save_top_k=cfg.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'
    
    return logger, callbacks


def build_configs(cfg: DebugConfig):
    """Build all configuration objects from the flat DebugConfig."""
    
    # Dataset configuration
    dataset_cfg = DatasetCfg(
        dataset_path=cfg.dataset_path,
        image_shape=cfg.image_shape,
        num_context_views=cfg.num_context_views,
    )
    
    # Data loader configuration
    train_loader_cfg = DataLoaderStageCfg(
        batch_size=cfg.batch_size_train,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )
    
    val_loader_cfg = DataLoaderStageCfg(
        batch_size=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )
    
    test_loader_cfg = DataLoaderStageCfg(
        batch_size=cfg.batch_size_test,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )
    
    data_loader_cfg = DataLoaderCfg(
        train=train_loader_cfg,
        val=val_loader_cfg,
        test=test_loader_cfg,
    )
    
    # Encoder configuration
    visualizer_cfg = EncoderVisualizerCostVolumeCfg(
        num_samples=cfg.visualizer_num_samples,
        min_resolution=cfg.visualizer_min_resolution,
        export_ply=cfg.visualizer_export_ply,
    )
    
    gaussian_adapter_cfg = GaussianAdapterCfg(
        gaussian_scale_min=cfg.gaussian_scale_min,
        gaussian_scale_max=cfg.gaussian_scale_max,
        sh_degree=cfg.sh_degree,
    )
    
    opacity_mapping_cfg = OpacityMappingCfg(
        initial=cfg.opacity_initial,
        final=cfg.opacity_final,
        warm_up=cfg.opacity_warm_up,
    )
    
    encoder_cfg = EncoderCostVolumeCfg(
        name=cfg.encoder_name,
        d_feature=cfg.d_feature,
        num_depth_candidates=cfg.num_depth_candidates,
        num_surfaces=cfg.num_surfaces,
        visualizer=visualizer_cfg,
        gaussian_adapter=gaussian_adapter_cfg,
        opacity_mapping=opacity_mapping_cfg,
        gaussians_per_pixel=cfg.gaussians_per_pixel,
        unimatch_weights_path=cfg.unimatch_weights_path,
        downscale_factor=cfg.downscale_factor,
        shim_patch_size=cfg.shim_patch_size,
        multiview_trans_attn_split=cfg.multiview_trans_attn_split,
        costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
        costvolume_unet_channel_mult=cfg.costvolume_unet_channel_mult,
        costvolume_unet_attn_res=cfg.costvolume_unet_attn_res,
        depth_unet_feat_dim=cfg.depth_unet_feat_dim,
        depth_unet_attn_res=cfg.depth_unet_attn_res,
        depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        wo_depth_refine=cfg.wo_depth_refine,
        wo_cost_volume=cfg.wo_cost_volume,
        wo_backbone_cross_attn=cfg.wo_backbone_cross_attn,
        wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        use_epipolar_trans=cfg.use_epipolar_trans,
        num_context_views=cfg.num_context_views,
    )
    
    # Decoder configuration
    decoder_cfg = DecoderSplattingCUDACfg(
        name=cfg.decoder_name,
    )
    
    # Loss configuration
    loss_cfgs = []
    if cfg.loss_mse_weight > 0:
        loss_cfgs.append(LossMseCfgWrapper(
            mse=LossMseCfg(weight=cfg.loss_mse_weight)
        ))
    if cfg.loss_depth_weight > 0:
        loss_cfgs.append(LossDepthCfgWrapper(
            depth=LossDepthCfg(
                weight=cfg.loss_depth_weight,
                sigma_image=cfg.loss_depth_sigma_image,
                use_second_derivative=cfg.loss_depth_use_second_derivative,
            )
        ))
    
    # Optimizer configuration
    optimizer_cfg = OptimizerCfg(
        lr=cfg.lr,
        warm_up_steps=cfg.warm_up_steps,
        cosine_lr=cfg.cosine_lr,
    )
    
    # Test configuration
    test_cfg = TestCfg(
        output_path=Path(cfg.test_output_path),
        compute_scores=cfg.compute_scores,
        save_image=cfg.save_image,
        save_video=cfg.save_video,
        eval_time_skip_steps=cfg.eval_time_skip_steps,
        experiment_name=cfg.experiment_name,
    )
    
    # Train configuration
    train_cfg = TrainCfg(
        depth_mode=cfg.depth_mode,
        extended_visualization=cfg.extended_visualization,
        print_log_every_n_steps=cfg.print_log_every_n_steps,
    )
    
    return {
        'dataset_cfg': dataset_cfg,
        'data_loader_cfg': data_loader_cfg,
        'encoder_cfg': encoder_cfg,
        'decoder_cfg': decoder_cfg,
        'loss_cfgs': loss_cfgs,
        'optimizer_cfg': optimizer_cfg,
        'test_cfg': test_cfg,
        'train_cfg': train_cfg,
    }


def train_debug():
    """Debug training entry point."""
    
    print(cyan("="*80))
    print(cyan("DEBUG MODE: Fast dry run with dummy data"))
    print(cyan("="*80))
    
    # Initialize debug configuration
    cfg = DebugConfig()
    
    print(cyan(f"\n[DEBUG] Configuration:"))
    print(cyan(f"  - Dataset: {cfg.dataset_path}"))
    print(cyan(f"  - Image shape: {cfg.image_shape}"))
    print(cyan(f"  - Batch size: {cfg.batch_size_train}"))
    print(cyan(f"  - Max steps: {cfg.max_steps}"))
    print(cyan(f"  - Num workers: {cfg.num_workers}"))
    print(cyan(f"  - Context views: {cfg.num_context_views}\n"))
    
    # Set up infrastructure (logging, callbacks, output directories)
    logger, callbacks = setup_infrastructure(cfg)
    
    # Build all configuration objects
    configs = build_configs(cfg)
    
    # Set up the trainer with fast_dev_run equivalent settings
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=1,  # Use only 1 GPU for debugging
        num_nodes=cfg.num_nodes,
        strategy="auto",  # No DDP for single GPU
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        enable_progress_bar=True,  # Show progress bar for debugging
        gradient_clip_val=cfg.gradient_clip_val,
        max_steps=cfg.max_steps,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
    )
    
    # Set random seed
    torch.manual_seed(cfg.seed + trainer.global_rank)
    
    print(cyan("[DEBUG] Initializing components..."))
    
    # Initialize step tracker
    step_tracker = StepTracker()
    
    # Initialize encoder
    print(cyan("  - Encoder..."))
    encoder, encoder_visualizer = get_encoder(configs['encoder_cfg'])
    
    # Initialize decoder
    print(cyan("  - Decoder..."))
    decoder = get_decoder(configs['decoder_cfg'], configs['dataset_cfg'])
    
    # Initialize losses
    print(cyan("  - Losses..."))
    losses = get_losses(configs['loss_cfgs'])
    
    # Build model wrapper
    print(cyan("  - Model wrapper..."))
    model_kwargs = {
        "optimizer_cfg": configs['optimizer_cfg'],
        "test_cfg": configs['test_cfg'],
        "train_cfg": configs['train_cfg'],
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": decoder,
        "losses": losses,
        "step_tracker": step_tracker,
    }
    
    model_wrapper = ModelWrapper(**model_kwargs)
    
    # Initialize data module
    print(cyan("  - Data module..."))
    data_module = DataModule(
        configs['dataset_cfg'],
        configs['data_loader_cfg'],
        step_tracker,
        global_rank=trainer.global_rank,
    )
    
    print(cyan("\n[DEBUG] Starting training loop...\n"))
    
    # Run training
    try:
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
        )
        print(cyan("\n" + "="*80))
        print(cyan("✓ DEBUG RUN COMPLETED SUCCESSFULLY!"))
        print(cyan("="*80))
        return True
    except Exception as e:
        print(cyan("\n" + "="*80))
        print(cyan(f"✗ DEBUG RUN FAILED: {e}"))
        print(cyan("="*80))
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    
    success = train_debug()
    exit(0 if success else 1)
