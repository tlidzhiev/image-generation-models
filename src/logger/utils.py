import numpy as np
import torch
import torchvision
from einops import rearrange


def make_grid_block(images: torch.Tensor, nrow: int) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images with shape (N, C, H, W).
    nrow : int
        Number of images per row in the grid.

    Returns
    -------
    np.ndarray
        Grid of images as numpy array with shape (H, W, C), normalized to [0, 1].
    """
    grid = torchvision.utils.make_grid(
        images,
        nrow=nrow,
        normalize=True,
        pad_value=1.0,
    ).numpy()

    grid = rearrange(grid, 'c h w -> h w c')
    return grid
