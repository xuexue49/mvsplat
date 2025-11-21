from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg



DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    if decoder_cfg.name == "splatting_cuda":
        return DecoderSplattingCUDA(decoder_cfg, dataset_cfg)
    raise NotImplementedError(f"Decoder {decoder_cfg.name} not implemented")
