from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetCfg:
    dataset_path: str
    image_shape: list[int]
    num_context_views: int
    background_color: list[float] = None  # RGB background color for rendering
    
    def __post_init__(self):
        # Set default white background if not specified
        if self.background_color is None:
            self.background_color = [1.0, 1.0, 1.0]

@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None

@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg
