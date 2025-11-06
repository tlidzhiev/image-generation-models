from typing import Literal

import torch
import torch.nn as nn

from src.utils.torch import get_activation, get_norm_layer

from ..utils import get_num_groups


class ResidualBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: str,
        norm_type: Literal['batch', 'group'],
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int
            Stride for first convolution.
        activation : str, optional
            Activation function name.
        norm_type : {'batch', 'group'}
            Type of normalization.
        """
        super().__init__()

        self.main = nn.Sequential(
            get_norm_layer(
                norm_type,
                in_channels,
                get_num_groups(in_channels) if norm_type == 'group' else None,
            ),
            get_activation(activation),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            get_norm_layer(
                norm_type,
                out_channels,
                get_num_groups(out_channels) if norm_type == 'group' else None,
            ),
            get_activation(activation),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height//stride, width//stride).
        """
        out = self.main(x)
        skip = self.skip(x)
        out = out + skip
        return out
