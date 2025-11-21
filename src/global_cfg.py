"""Global configuration storage - simplified for non-Hydra usage."""
from typing import Optional, Any

cfg: Optional[dict[str, Any]] = None


def get_cfg() -> dict[str, Any]:
    """Get the global configuration dictionary."""
    global cfg
    if cfg is None:
        # Return a minimal default config to prevent crashes
        return {
            "mode": "train",
            "wandb": {"name": "default"},
            "dataset": {"view_sampler": {"num_context_views": 2}},
        }
    return cfg


def set_cfg(new_cfg: dict[str, Any]) -> None:
    """Set the global configuration dictionary."""
    global cfg
    cfg = new_cfg


def get_seed() -> int:
    """Get the random seed from configuration."""
    return cfg.get("seed", 42) if cfg else 42
