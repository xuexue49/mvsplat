from typing import Optional

from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume



EncoderCfg = EncoderCostVolumeCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    if cfg.name == "costvolume":
        encoder = EncoderCostVolume(cfg)
        visualizer = EncoderVisualizerCostVolume(cfg.visualizer, encoder)
        return encoder, visualizer
    raise NotImplementedError(f"Encoder {cfg.name} not implemented")
