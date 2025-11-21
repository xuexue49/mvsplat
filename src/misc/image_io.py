import io
from pathlib import Path
from typing import Union


import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


from plyfile import PlyData, PlyElement

def save_ply(
    means: Float[Tensor, "gaussian 3"],
    colors: Float[Tensor, "gaussian 3"] | None,
    path: Union[Path, str],
) -> None:
    """Save a point cloud to a PLY file."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    means = means.detach().cpu().numpy()
    if colors is not None:
        colors = colors.detach().cpu().numpy()
        # Ensure colors are 0-255 uint8
        if colors.max() <= 1.0:
            colors = (colors * 255).clip(0, 255).astype(np.uint8)
        else:
            colors = colors.clip(0, 255).astype(np.uint8)
        
        vertices = np.empty(len(means), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = means[:, 0]
        vertices['y'] = means[:, 1]
        vertices['z'] = means[:, 2]
        vertices['red'] = colors[:, 0]
        vertices['green'] = colors[:, 1]
        vertices['blue'] = colors[:, 2]
    else:
        vertices = np.array([tuple(v) for v in means], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(str(path))
