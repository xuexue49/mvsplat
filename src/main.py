import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from src.config import RootCfg, load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def get_output_dir(cfg_dict: DictConfig) -> Path:
    """
    获取输出目录。
    
    Args:
        cfg_dict: Hydra的DictConfig对象。
        
    Returns:
        output_dir: 输出目录路径。
    """
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    return output_dir


def get_logger(cfg_dict: DictConfig, output_dir: Path) -> WandbLogger | LocalLogger:
    """
    获取日志记录器。
    
    Args:
        cfg_dict: Hydra的DictConfig对象。
        output_dir: 输出目录路径。
        
    Returns:
        logger: WandbLogger或LocalLogger实例。
    """
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    return logger


def get_callbacks(cfg: RootCfg, output_dir: Path) -> list[Callback]:
    """
    获取回调列表。
    
    Args:
        cfg: 类型化的RootCfg对象。
        output_dir: 输出目录路径。
        
    Returns:
        callbacks: 回调列表。
    """
    callbacks = []
    if cfg.wandb["mode"] != "disabled":
        callbacks.append(LearningRateMonitor("step", True))

    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'
    return callbacks


def setup_training_components(
    cfg: RootCfg,
    cfg_dict: DictConfig,
    logger: WandbLogger | LocalLogger,
    callbacks: list[Callback],
    checkpoint_path: str | None,
) -> tuple[Trainer, ModelWrapper, DataModule]:
    """
    初始化训练组件，包括Trainer、ModelWrapper和DataModule。
    
    Args:
        cfg: 类型化的RootCfg对象。
        cfg_dict: Hydra的DictConfig对象。
        logger: 日志记录器。
        callbacks: 回调列表。
        checkpoint_path: 检查点路径。
        
    Returns:
        trainer: PyTorch Lightning Trainer实例。
        model_wrapper: 模型包装器实例。
        data_module: 数据模块实例。
    """
    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    
    return trainer, model_wrapper, data_module


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    """
    训练入口函数。
    
    Args:
        cfg_dict: Hydra配置对象。
    """
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = get_output_dir(cfg_dict)
    logger = get_logger(cfg_dict, output_dir)
    callbacks = get_callbacks(cfg, output_dir)
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    trainer, model_wrapper, data_module = setup_training_components(
        cfg, cfg_dict, logger, callbacks, checkpoint_path
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
