import torch
import torch.nn as nn
from einops import rearrange

from src.utils.torch import get_activation, get_norm_layer

from ..utils import get_num_groups


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding for DDPM UNet.

    Applies two convolution blocks with time embedding injection and residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        time_dim : int
            Dimension of time embedding.
        dropout : float, optional
            Dropout probability, by default 0.1.
        """
        super().__init__()
        self.conv_block1 = nn.Sequential(
            get_norm_layer('group', in_channels, get_num_groups(in_channels)),
            get_activation('silu'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv_block2 = nn.Sequential(
            get_norm_layer('group', out_channels, get_num_groups(out_channels)),
            get_activation('silu'),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time embedding injection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).
        t : torch.Tensor
            Time embedding of shape (batch, time_dim).

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection, shape (batch, out_channels, height, width).
        """
        h = self.conv_block1(x)
        time_emb = self.time_proj(t)
        time_emb = rearrange(time_emb, 'b d -> b d 1 1')
        h = h + time_emb
        h = self.conv_block2(h)
        return h + self.shortcut(x)
