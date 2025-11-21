from dataclasses import dataclass


@dataclass
class DatasetCfgCommon:
    """Legacy dataset configuration - kept for backward compatibility.
    
    Note: This is not used by the new MedicalDataset.
    """
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
