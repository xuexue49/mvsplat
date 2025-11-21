from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetCfg:
    dataset_path: str
    image_shape: list[int]
    num_context_views: int

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
