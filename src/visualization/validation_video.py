import moviepy.editor as mpy
import torch
import wandb
from einops import pack, repeat
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat


def render_video_wobble(
    encoder,
    decoder,
    batch: BatchedExample,
    global_step: int,
    device: torch.device,
    logger,
) -> None:
    # Two views are needed to get the wobble radius.
    _, v, _, _ = batch["context"]["extrinsics"].shape
    if v != 2:
        return

    def trajectory_fn(t):
        origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
        origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        extrinsics = generate_wobble(
            batch["context"]["extrinsics"][:, 0],
            delta * 0.25,
            t,
        )
        intrinsics = repeat(
            batch["context"]["intrinsics"][:, 0],
            "b i j -> b v i j",
            v=t.shape[0],
        )
        return extrinsics, intrinsics

    return render_video_generic(
        encoder, decoder, batch, trajectory_fn, "wobble", global_step, device, logger, num_frames=60
    )


def render_video_interpolation(
    encoder,
    decoder,
    batch: BatchedExample,
    global_step: int,
    device: torch.device,
    logger,
) -> None:
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, 0],
            (
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0]
            ),
            t,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            (
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0]
            ),
            t,
        )
        return extrinsics[None], intrinsics[None]

    return render_video_generic(encoder, decoder, batch, trajectory_fn, "rgb", global_step, device, logger)


def render_video_interpolation_exaggerated(
    encoder,
    decoder,
    batch: BatchedExample,
    global_step: int,
    device: torch.device,
    logger,
) -> None:
    # Two views are needed to get the wobble radius.
    _, v, _, _ = batch["context"]["extrinsics"].shape
    if v != 2:
        return

    def trajectory_fn(t):
        origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
        origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        tf = generate_wobble_transformation(
            delta * 0.5,
            t,
            5,
            scale_radius_with_t=False,
        )
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, 0],
            (
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0]
            ),
            t * 5 - 2,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            (
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0]
            ),
            t * 5 - 2,
        )
        return extrinsics @ tf, intrinsics[None]

    return render_video_generic(
        encoder,
        decoder,
        batch,
        trajectory_fn,
        "interpolation_exagerrated",
        global_step,
        device,
        logger,
        num_frames=300,
        smooth=False,
        loop_reverse=False,
    )


def render_video_generic(
    encoder,
    decoder,
    batch: BatchedExample,
    trajectory_fn,
    name: str,
    global_step: int,
    device: torch.device,
    logger,
    num_frames: int = 30,
    smooth: bool = True,
    loop_reverse: bool = True,
) -> None:
    # Render probabilistic estimate of scene.
    gaussians_prob = encoder(batch["context"], global_step, False)
    # gaussians_det = encoder(batch["context"], global_step, True)

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    extrinsics, intrinsics = trajectory_fn(t)

    _, _, _, h, w = batch["context"]["image"].shape

    # Color-map the result.
    def depth_map(result):
        near = result[result > 0][:16_000_000].quantile(0.01).log()
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        result = result.log()
        result = 1 - (result - near) / (far - near)
        return apply_color_map_to_image(result, "turbo")

    # TODO: Interpolate near and far planes?
    near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
    output_prob = decoder.forward(
        gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
    )
    images_prob = [
        vcat(rgb, depth)
        for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
    ]
    
    images = [
        add_border(
            hcat(
                add_label(image_prob, "Softmax"),
                # add_label(image_det, "Deterministic"),
            )
        )
        for image_prob, _ in zip(images_prob, images_prob)
    ]

    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    visualizations = {
        f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
    }

    # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
    try:
        wandb.log(visualizations)
    except Exception:
        assert isinstance(logger, LocalLogger)
        for key, value in visualizations.items():
            tensor = value._prepare_video(value.data)
            clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
            dir = LOG_PATH / key
            dir.mkdir(exist_ok=True, parents=True)
            clip.write_videofile(
                str(dir / f"{global_step:0>6}.mp4"), logger=None
            )


def add_label(image, label):
    from ..visualization.annotation import add_label as _add_label
    return _add_label(image, label)
