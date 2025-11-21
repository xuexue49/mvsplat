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

# Standard imports (previously inside install_import_hook)
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
class ExperimentConfig:
    """Simple Python configuration for the experiment.
    Edit these values directly to configure your training run."""
    
    # Dataset configuration
    dataset_path: str = "data/medical_dataset"
    image_shape: list[int] = field(default_factory=lambda: [512, 512])
    num_context_views: int = 2
    
    # Data loader configuration
    batch_size_train: int = 1
    batch_size_val: int = 1
    batch_size_test: int = 1
    num_workers: int = 4
    persistent_workers: bool = True
    seed: int = 42
    
    # Training configuration
    mode: Literal["train", "test"] = "train"
    max_steps: int = 300000
    val_check_interval: int = 1000
    gradient_clip_val: float = 1.0
    num_sanity_val_steps: int = 0
    num_nodes: int = 1
    
    # Optimizer configuration
    lr: float = 3e-4
    warm_up_steps: int = 2000
    cosine_lr: bool = True
    
    # Checkpointing configuration
    checkpoints_dir: str = "outputs/checkpoints"
    every_n_train_steps: int = 1000
    save_top_k: int = 5
    load_checkpoint: str | None = None
    resume_training: bool = False
    
    # Output configuration
    output_dir: str | None = None
    experiment_name: str = "medical-experiment"  # Name for organizing outputs
    
    # Test configuration
    test_output_path: str = "outputs/test_results"
    compute_scores: bool = True
    save_image: bool = True
    save_video: bool = False
    eval_time_skip_steps: int = 1
    
    # Train visualization configuration
    depth_mode: DepthRenderingMode | None = "depth"
    extended_visualization: bool = False
    print_log_every_n_steps: int = 100
    
    # Model configuration - Encoder
    encoder_name: Literal["costvolume"] = "costvolume"
    d_feature: int = 64
    num_depth_candidates: int = 64
    num_surfaces: int = 2
    gaussians_per_pixel: int = 1
    unimatch_weights_path: str | None = None
    downscale_factor: int = 4
    shim_patch_size: int = 7
    multiview_trans_attn_split: int = 1
    costvolume_unet_feat_dim: int = 64
    costvolume_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4])
    costvolume_unet_attn_res: list[int] = field(default_factory=lambda: [])
    depth_unet_feat_dim: int = 64
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
    opacity_warm_up: int = 5000
    
    # Encoder visualizer configuration
    visualizer_num_samples: int = 128
    visualizer_min_resolution: int = 64
    visualizer_export_ply: bool = False
    
    # Decoder configuration
    decoder_name: Literal["splatting_cuda"] = "splatting_cuda"
    
    # Loss configuration
    loss_mse_weight: float = 1.0
    loss_depth_weight: float = 0.0
    loss_depth_sigma_image: float | None = None
    loss_depth_use_second_derivative: bool = False



def setup_infrastructure(cfg: ExperimentConfig) -> tuple[Logger, list]:
    """Set up output directories, logging, and callbacks."""
    
    # Set up the output directory
    if cfg.output_dir is None:
        output_dir = Path(cfg.checkpoints_dir).parent / "runs" / f"run_{cfg.experiment_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(cyan(f"Saving outputs to {output_dir}."))
    
    # Create a symlink to the latest run
    latest_run = output_dir.parent / "latest-run"
    if latest_run.exists() or latest_run.is_symlink():
        latest_run.unlink()
    try:
        latest_run.symlink_to(output_dir)
    except Exception as e:
        print(f"Warning: Could not create symlink to latest run: {e}")
    
    # Set up logging with local logger only
    callbacks = []
    logger = LocalLogger()
    print(cyan("Using LocalLogger for experiment tracking."))
    
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


def build_configs(cfg: ExperimentConfig):
    """Build all configuration objects from the flat ExperimentConfig."""
    
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


def train():
    """Main training/testing entry point."""
    
    # Initialize configuration
    cfg = ExperimentConfig()
    
    # Set up infrastructure (logging, callbacks, output directories)
    logger, callbacks = setup_infrastructure(cfg)
    
    # Build all configuration objects
    configs = build_configs(cfg)
    
    # Set up the trainer
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.gradient_clip_val,
        max_steps=cfg.max_steps,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
    )
    
    # Set random seed
    torch.manual_seed(cfg.seed + trainer.global_rank)
    
    # Initialize step tracker
    step_tracker = StepTracker()
    
    # Initialize encoder
    encoder, encoder_visualizer = get_encoder(configs['encoder_cfg'])
    
    # Initialize decoder
    decoder = get_decoder(configs['decoder_cfg'], configs['dataset_cfg'])
    
    # Initialize losses
    losses = get_losses(configs['loss_cfgs'])
    
    # Build model wrapper
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
    
    # Load from checkpoint if specified
    if cfg.mode == "train" and cfg.load_checkpoint is not None and not cfg.resume_training:
        # Just load model weights, without optimizer states
        model_wrapper = ModelWrapper.load_from_checkpoint(
            cfg.load_checkpoint, **model_kwargs, strict=True
        )
        print(cyan(f"Loaded weights from {cfg.load_checkpoint}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)
    
    # Initialize data module
    data_module = DataModule(
        configs['dataset_cfg'],
        configs['data_loader_cfg'],
        step_tracker,
        global_rank=trainer.global_rank,
    )
    
    # Run training or testing
    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=(cfg.load_checkpoint if cfg.resume_training else None)
        )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=cfg.load_checkpoint,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    
    train()
