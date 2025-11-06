import math
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.torch import get_activation, get_norm_layer, initialize_weights

from .attention import SelfAttentionBlock
from .residual import ResidualBlock


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for DDPM.

    Converts timestep indices to continuous embeddings using sinusoidal positional encoding.
    """

    def __init__(self, dim: int) -> None:
        """
        Parameters
        ----------
        dim : int
            Embedding dimension (must be even).
        """
        assert dim % 2 == 0, 'dim must be even'
        super().__init__()

        hidden_dim = dim * 4
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation('silu'),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            'freq_bands',
            torch.exp(-math.log(10000) * torch.arange(0, dim // 2) / (dim // 2)),
        )

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """
        Compute time embeddings.

        Parameters
        ----------
        t : torch.LongTensor
            Timestep indices of shape (batch,).

        Returns
        -------
        torch.Tensor
            Time embeddings of shape (batch, dim*4).
        """
        freqs = rearrange(t, 'b -> b 1').float() * self.freq_bands
        embeddings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.proj(embeddings)


class DownBlock(nn.Module):
    """
    Downsampling block for UNet encoder.

    Applies residual block and optional self-attention.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, has_attn: bool) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        time_dim : int
            Dimension of time embedding.
        has_attn : bool
            Whether to apply self-attention.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        if has_attn:
            self.attn = SelfAttentionBlock(out_channels, num_heads=4)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through down block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        t : torch.Tensor
            Time embedding.

        Returns
        -------
        torch.Tensor
            Output tensor after residual and attention blocks.
        """
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle block for UNet bottleneck.

    Applies two residual blocks with self-attention in between.
    """

    def __init__(self, channels: int, time_dim: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of channels.
        time_dim : int
            Dimension of time embedding.
        """
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_dim)
        self.attn = SelfAttentionBlock(channels, num_heads=4)
        self.res2 = ResidualBlock(channels, channels, time_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through middle block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        t : torch.Tensor
            Time embedding.

        Returns
        -------
        torch.Tensor
            Output tensor after two residual blocks and attention.
        """
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block for UNet decoder.

    Applies residual block with skip connection and optional self-attention.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, has_attn: bool) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (will be concatenated with skip connection).
        out_channels : int
            Number of output channels.
        time_dim : int
            Dimension of time embedding.
        has_attn : bool
            Whether to apply self-attention.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_dim)
        if has_attn:
            self.attn = SelfAttentionBlock(out_channels, num_heads=4)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through up block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (after concatenation with skip connection).
        t : torch.Tensor
            Time embedding.

        Returns
        -------
        torch.Tensor
            Output tensor after residual and attention blocks.
        """
        x = self.res(x, t)
        x = self.attn(x)
        return x


class Downsample(nn.Module):
    """
    Spatial downsampling layer.

    Reduces spatial dimensions by factor of 2 using strided convolution.
    """

    def __init__(self, channels: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Downsample spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).
        t : torch.Tensor or None, optional
            Time embedding (unused, for interface compatibility), by default None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, height//2, width//2).
        """
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling layer.

    Increases spatial dimensions by factor of 2 using interpolation and convolution.
    """

    def __init__(self, channels: int) -> None:
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Upsample spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).
        t : torch.Tensor
            Time embedding (unused, for interface compatibility).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, height*2, width*2).
        """
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for DDPM.

    Encoder-decoder architecture with skip connections, time embeddings,
    and optional self-attention layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        ch_mults: list[int],
        use_attn: list[bool],
        num_blocks: int,
        init_mode: Literal['normal', 'uniform'] = 'normal',
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input image channels.
        num_channels : int
            Base number of channels.
        ch_mults : list[int]
            Channel multipliers for each resolution level.
        use_attn : list[bool]
            Whether to use attention at each resolution level.
        num_blocks : int
            Number of residual blocks per resolution level.
        init_mode : {'normal', 'uniform'}, optional
            Weight initialization mode, by default 'normal'.
        """
        super().__init__()
        image_channels = in_channels
        num_resolutions = len(ch_mults)
        self.input_proj = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(num_channels)

        down = []
        out_channels = in_channels = num_channels
        for i in range(num_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(num_blocks):
                down.append(DownBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
                in_channels = out_channels
            if i < num_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, num_channels * 4)

        up = []
        in_channels = out_channels
        for i in reversed(range(num_resolutions)):
            out_channels = in_channels
            for _ in range(num_blocks):
                up.append(UpBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.output_proj = nn.Sequential(
            get_norm_layer('group', num_channels, 8),
            get_activation('silu'),
            nn.Conv2d(num_channels, image_channels, kernel_size=3, padding=1),
        )
        self._initialize_weights(mode=init_mode)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass through UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, in_channels, height, width).
        t : torch.LongTensor
            Timestep indices of shape (batch,).

        Returns
        -------
        torch.Tensor
            Predicted noise of same shape as input.
        """
        x, t = self.input_proj(x), self.time_emb(t)
        skips = [x]
        for block in self.down:
            x = block(x, t)
            skips.append(x)

        x = self.middle(x, t)

        for block in self.up:
            if isinstance(block, Upsample):
                x = block(x, t)
            else:
                s = skips.pop()
                x = torch.cat((x, s), dim=1)
                x = block(x, t)
        return self.output_proj(x)

    def _initialize_weights(self, mode: Literal['normal', 'uniform']) -> None:
        """
        Initialize network weights.

        Parameters
        ----------
        mode : {'normal', 'uniform'}
            Initialization mode.
        """
        initialize_weights(self, activation='silu', mode=mode)
