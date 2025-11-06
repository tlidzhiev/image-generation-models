from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils.torch import get_norm_layer

from ..utils import get_num_groups


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with residual connection.
    """

    def __init__(
        self,
        channels: int,
        norm_type: Literal['batch', 'group'],
        num_heads: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of input channels.
        norm_type : {'batch', 'group'}
            Type of normalization.
        num_heads : int, optional
            Number of attention heads, by default 8.
        """
        super().__init__()
        self.channels: int = channels
        self.num_heads: int = num_heads
        self.head_dim: int = channels // num_heads

        assert channels % num_heads == 0, 'channels must be divisible by num_heads'

        self.norm = get_norm_layer(
            norm_type,
            channels,
            get_num_groups(channels) if norm_type == 'group' else None,
        )
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection, same shape as input.
        """
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = rearrange(
            qkv,
            'b (qkv heads c) h w -> qkv b heads (h w) c',
            qkv=3,
            heads=self.num_heads,
            c=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', h=h, w=w)
        out = self.proj(out)
        return x + out
