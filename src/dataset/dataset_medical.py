import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import imageio.v3 as iio
from torch.utils.data import Dataset
from einops import rearrange

from src.dataset.types import UnbatchedExample, UnbatchedViews

class MedicalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_shape: list[int],
        num_context_views: int,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.dataset_path = Path(dataset_path)
        self.image_shape = tuple(image_shape)
        self.num_context_views = num_context_views
        self.split = split

        with open(self.dataset_path / "transforms.json", "r") as f:
            self.meta = json.load(f)

        self.frames = self.meta["frames"]
        
        # Sort frames by file_path to ensure determinism
        self.frames.sort(key=lambda x: x["file_path"])

        # Basic intrinsics from meta
        self.w = self.meta.get("w", 800)
        self.h = self.meta.get("h", 800)
        angle_x = self.meta.get("camera_angle_x", 0.0)
        if "fl_x" in self.meta:
            self.fl_x = self.meta["fl_x"]
            self.fl_y = self.meta.get("fl_y", self.fl_x)
            self.cx = self.meta.get("cx", self.w / 2)
            self.cy = self.meta.get("cy", self.h / 2)
        else:
            # Assuming square pixels if only angle_x is given
            self.fl_x = 0.5 * self.w / np.tan(0.5 * angle_x)
            self.fl_y = self.fl_x
            self.cx = self.w / 2
            self.cy = self.h / 2

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index) -> UnbatchedExample:
        # Target view
        target_frame = self.frames[index]
        target_view = self.load_view(target_frame, index)

        # Context views
        # Deterministic selection: take indices (index + 1, index + 2, ...) modulo len
        context_indices = [
            (index + i + 1) % len(self.frames) 
            for i in range(self.num_context_views)
        ]
        
        context_views_list = [
            self.load_view(self.frames[i], i) 
            for i in context_indices
        ]
        
        # Collate context views
        context_views = self.collate_views(context_views_list)
        
        # Target view (batched as 1 view)
        target_views = self.collate_views([target_view])

        return {
            "context": context_views,
            "target": target_views,
            "scene": self.dataset_path.name,
        }

    def load_view(self, frame, index) -> UnbatchedViews:
        # Load image
        fname = self.dataset_path / frame["file_path"]
        
        # Handle potential relative paths or missing extensions if needed
        # Assuming standard format where file_path is relative to dataset_path
        
        image = iio.imread(fname)
        
        image = torch.from_numpy(image).float() / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3] # Drop alpha
        
        image = rearrange(image, "h w c -> c h w")
        
        # Resize
        if (image.shape[1], image.shape[2]) != self.image_shape:
             image = torch.nn.functional.interpolate(
                 image.unsqueeze(0), 
                 size=self.image_shape, 
                 mode="bilinear", 
                 align_corners=False
             ).squeeze(0)

        # Extrinsics
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        
        # Convert to W2C (extrinsics)
        w2c = c2w.inverse()

        # Intrinsics
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = self.fl_x
        intrinsics[1, 1] = self.fl_y
        intrinsics[0, 2] = self.cx
        intrinsics[1, 2] = self.cy
        
        # Scale intrinsics if resized
        scale_w = self.image_shape[1] / self.w
        scale_h = self.image_shape[0] / self.h
        intrinsics[0] *= scale_w
        intrinsics[1] *= scale_h

        return {
            "extrinsics": w2c,
            "intrinsics": intrinsics,
            "image": image,
            "near": torch.tensor(0.1), # Default near
            "far": torch.tensor(100.0), # Default far
            "index": torch.tensor(index),
        }

    def collate_views(self, views_list: list[UnbatchedViews]) -> UnbatchedViews:
        return {
            "extrinsics": torch.stack([v["extrinsics"] for v in views_list]),
            "intrinsics": torch.stack([v["intrinsics"] for v in views_list]),
            "image": torch.stack([v["image"] for v in views_list]),
            "near": torch.stack([v["near"] for v in views_list]),
            "far": torch.stack([v["far"] for v in views_list]),
            "index": torch.stack([v["index"] for v in views_list]),
        }
