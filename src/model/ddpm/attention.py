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

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of input channels.
        num_heads : int, optional
            Number of attention heads, by default 8.
        """
        super().__init__()
        self.channels: int = channels
        self.num_heads: int = num_heads
        self.head_dim: int = channels // num_heads

        assert channels % num_heads == 0, 'channels must be divisible by num_heads'

        self.norm = get_norm_layer('group', channels, get_num_groups(channels))
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).
        t : torch.Tensor or None, optional
            Time embedding (unused, for interface compatibility), by default None.

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection, same shape as input.
        """
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm, 'b c h w -> b (h w) c')
        qkv = self.qkv(x_norm)
        qkv = rearrange(
            qkv,
            'b n (qkv heads d) -> qkv b heads n d',
            qkv=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b heads n d -> b n (heads d)')
        out = self.proj(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return x + out
