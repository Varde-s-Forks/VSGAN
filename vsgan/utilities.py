from __future__ import annotations

import gc

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.constants import IS_VS_API_4, MAX_DTYPE_VALUES


def get_frame_plane(f: vs.VideoFrame, n: int) -> memoryview:
    """
    Get a VideoFrame's Plane data as a MemoryView or a numpy array.
    Supports VS API 3 and 4.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
        n: Plane number.
    """
    if IS_VS_API_4:
        return f[n]
    return f.get_read_array(n)  # type: ignore


def frame_to_tensor(f: vs.VideoFrame, as_f32=True, half: bool = False) -> torch.Tensor:
    """
    Convert a VapourSynth VideoFrame into a PyTorch Tensor.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
        as_f32: Convert to float32 in 0,1 range.
        half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
    """
    tensor = torch.stack(tuple(
        torch.frombuffer(
            buffer=memoryview(mv.tobytes()).cast("b"),  # plane as contiguous signed data
            dtype=torch.float32
        ).reshape(mv.shape)
        for plane in range(f.format.num_planes)
        for mv in [get_frame_plane(f, plane)]
    ))

    if as_f32:
        max_val = MAX_DTYPE_VALUES.get(tensor.dtype, 1.0)
        tensor = tensor.to(torch.float32) / max_val

    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_frame(f: vs.VideoFrame, t: torch.Tensor) -> vs.VideoFrame:
    """
    Copies each channel from a Tensor into a VapourSynth VideoFrame.
    Supports any depth and format, and will return in the same format.

    It expects the tensor array to have the dimension count (C) first
    in the shape, e.g., CHW or CWH.

    Parameters:
        f: VapourSynth frame to store retrieved planes.
        t: PyTorch Tensor array to retrieve planes from.
    """
    # TODO: - Is the return needed? Looks like in-place modification
    #       - What if the frame is read-only?

    array = t.squeeze(0).detach().clamp(0, 1).cpu().numpy()

    d_type = np.asarray(f[0]).dtype
    array = MAX_DTYPE_VALUES.get(d_type, 1.0) * array
    array = array.astype(d_type)

    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])

    return f


def tensor_to_clip(clip: vs.VideoNode, image: torch.Tensor) -> vs.VideoNode:
    """
    Convert a PyTorch Tensor into a VapourSynth VideoNode (clip).

    Expecting Torch shape to be in CHW order.

    Parameters:
        clip: Used to inherit expected return properties only.
        image: PyTorch Tensor.
    """
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )


def tile_tensor(t: torch.Tensor, overlap: int = 16) -> tuple[torch.Tensor, ...]:
    """
    Tile PyTorch Tensor into 4 quadrants with an overlap between tiles.
    Expects input PyTorch Tensor's shape to end in HW order.
    """
    b, c, h, w = t.shape

    top_left_lr = t[..., : h // 2 + overlap, : w // 2 + overlap]
    top_right_lr = t[..., : h // 2 + overlap, w // 2 - overlap:]
    bottom_left_lr = t[..., h // 2 - overlap:, : w // 2 + overlap]
    bottom_right_lr = t[..., h // 2 - overlap:, w // 2 - overlap:]

    return top_left_lr, top_right_lr, bottom_left_lr, bottom_right_lr


def recursive_tile_tensor(
    t: torch.Tensor,
    model: torch.nn.Module,
    overlap: int = 16,
    max_depth: int = None,
    current_depth: int = 1
) -> tuple[torch.Tensor, int]:
    """
    Recursively Tile PyTorch Tensor until the device has enough VRAM.
    It will try to tile as little as possible, and wont tile unless needed.
    Expects input PyTorch Tensor's shape to end in HW order.
    """
    if current_depth > 10:
        torch.cuda.empty_cache()
        gc.collect()
        raise RecursionError(f"Exceeded maximum tiling recursion of 10...")

    if max_depth is None or max_depth == current_depth:
        # attempt non-tiled super-resolution if no known depth, or at depth
        try:
            t_sr = model(t).data
            # del t  # TODO: Truly beneficial?
            return t_sr, current_depth
        except RuntimeError as e:
            if "allocate" in str(e) or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()  # TODO: Truly beneficial?
            else:
                raise

    # Not at known depth, and non-tiled super-resolution failed, try tiled

    # reduce overlap in half every recursion
    overlap //= current_depth

    tiles_lr = tile_tensor(t, overlap)
    # take depth from top_left result as the size would be same for all quadrants
    # by re-using the depth, we can know exactly how much tiling is needed immediately
    tiles_lr_top_left, depth = recursive_tile_tensor(tiles_lr[0], model, overlap, current_depth=current_depth + 1)
    tiles_lr_top_right, _ = recursive_tile_tensor(tiles_lr[1], model, overlap, depth, current_depth=current_depth + 1)
    tiles_lr_bottom_left, _ = recursive_tile_tensor(tiles_lr[2], model, overlap, depth, current_depth=current_depth + 1)
    tiles_lr_bottom_right, _ = recursive_tile_tensor(tiles_lr[3], model, overlap, depth, current_depth=current_depth + 1)

    output_img = join_tiles(
        (tiles_lr_top_left, tiles_lr_top_right, tiles_lr_bottom_left, tiles_lr_bottom_right),
        overlap * model.scale
    )

    return output_img, depth


def join_tiles(tiles: tuple[torch.Tensor, ...], overlap: int) -> torch.Tensor:
    """
    Join Tiled PyTorch Tensor quadrants into one large PyTorch Tensor.
    Expects input PyTorch Tensor's shapes to end in HW order.

    Ensure the overlap value is what it currently is, possibly after
    super-resolution, not before!

    Parameters:
        tiles: The PyTorch Tensor tiles you wish to rejoin.
        overlap: The amount of overlap currently between tiles.
    """
    b, c, h, w = tiles[0].shape

    h = (h - overlap) * 2
    w = (w - overlap) * 2

    joined_tile = torch.empty((b, c, h, w), dtype=tiles[0].dtype, device=tiles[0].device)
    joined_tile[..., : h // 2, : w // 2] = tiles[0][..., : h // 2, : w // 2]
    joined_tile[..., : h // 2, -w // 2:] = tiles[1][..., : h // 2, -w // 2:]
    joined_tile[..., -h // 2:, : w // 2] = tiles[2][..., -h // 2:, : w // 2]
    joined_tile[..., -h // 2:, -w // 2:] = tiles[3][..., -h // 2:, -w // 2:]

    return joined_tile
