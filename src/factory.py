from typing import List, Union, Tuple, Optional
from src.model.encoder.encoder import Encoder
from src.model.encoder.encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from src.model.decoder.decoder import Decoder
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.loss.loss import Loss
from src.loss.loss_mse import LossMse, LossMseCfgWrapper
from src.loss.loss_depth import LossDepth, LossDepthCfgWrapper

def get_encoder(cfg: EncoderCostVolumeCfg) -> Tuple[Encoder, None]:
    if cfg.name == "costvolume":
        return EncoderCostVolume(cfg), None
    raise ValueError(f"Unknown encoder: {cfg.name}")

def get_decoder(cfg: DecoderSplattingCUDACfg, dataset_cfg) -> Decoder:
    if cfg.name == "splatting_cuda":
        return DecoderSplattingCUDA(cfg, dataset_cfg)
    raise ValueError(f"Unknown decoder: {cfg.name}")

def get_losses(cfgs: List[Union[LossMseCfgWrapper, LossDepthCfgWrapper]]) -> List[Loss]:
    losses = []
    for cfg in cfgs:
        if hasattr(cfg, "mse"):
            losses.append(LossMse(cfg))
        elif hasattr(cfg, "depth"):
            losses.append(LossDepth(cfg))
    return losses
