from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper



LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    losses = []
    for cfg in cfgs:
        if isinstance(cfg, LossDepthCfgWrapper):
            losses.append(LossDepth(cfg))
        elif isinstance(cfg, LossLpipsCfgWrapper):
            losses.append(LossLpips(cfg))
        elif isinstance(cfg, LossMseCfgWrapper):
            losses.append(LossMse(cfg))
        else:
            raise NotImplementedError(f"Loss {type(cfg)} not implemented")
    return losses
