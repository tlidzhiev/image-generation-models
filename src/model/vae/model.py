from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.torch import get_activation, get_norm_layer, initialize_weights

from ..base import BaseModel
from ..utils import get_num_groups
from .attention import SelfAttentionBlock
from .residual import ResidualBlock


class Encoder(nn.Module):
    """
    VAE encoder that maps images to latent distributions.

    Parameters
    ----------
    image_size : int
        Input image size (assumes square images).
    hidden_dims : list[int]
        Channel dimensions for each layer.
    latent_dim : int
        Dimension of latent space.
    activation : {'relu', 'leaky_relu', 'silu', 'gelu'}
        Activation function.
    norm_type : {'batch', 'group'}
        Normalization type.
    use_res : bool
        Use residual blocks.
    attn_at : list[int] or None
        Resolutions at which to apply attention (e.g., [16, 8]).
        If None, no attention is used.
    """

    def __init__(
        self,
        image_size: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batch', 'group'],
        use_res: bool,
        attn_at: list[int] | None,
    ) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim

        if len(hidden_dims) < 2:
            raise ValueError(
                '`hidden_dims` must contain at least two entries: base channels and '
                'at least one downsampling stage.'
            )

        if attn_at is None:
            attn_at = []

        downsample_dims = hidden_dims[1:]

        self.features = self._build_encoder_layers(
            hidden_dims[0],
            downsample_dims,
            activation,
            norm_type,
            use_res,
            attn_at,
            image_size,
        )

        self.feature_size: int = self._calculate_feature_size(image_size, len(downsample_dims))
        self.flatten_dim: int = downsample_dims[-1] * self.feature_size * self.feature_size

        self.to_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.to_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def _build_encoder_layers(
        self,
        in_channels: int,
        hidden_dims: list[int],
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batch', 'group'],
        use_res: bool,
        attn_at: list[int],
        image_size: int,
    ) -> nn.Sequential:
        layers = []
        current_resolution = image_size

        for out_channels in hidden_dims:
            if use_res:
                block = ResidualBlock(
                    in_channels,
                    out_channels,
                    stride=2,
                    activation=activation,
                    norm_type=norm_type,
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    get_norm_layer(
                        norm_type,
                        out_channels,
                        get_num_groups(out_channels) if norm_type == 'group' else None,
                    ),
                    get_activation(activation),
                )
            layers.append(block)
            current_resolution = current_resolution // 2
            if current_resolution in attn_at:
                layers.append(SelfAttentionBlock(out_channels, norm_type))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _calculate_feature_size(image_size: int, num_layers: int) -> int:
        downsample_factor = 2**num_layers
        if image_size % downsample_factor != 0:
            raise ValueError(
                f'Encoder configuration expects image size divisible by {downsample_factor}, '
                f'got {image_size}. Adjust `hidden_dims` or `image_size`.'
            )
        return image_size // downsample_factor

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input images.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Latent sample, mean, and log variance.
        """
        features = self.features(x)
        features_flat = rearrange(features, 'b c h w -> b (c h w)')

        mu = self.to_mu(features_flat)
        logvar = self.to_logvar(features_flat)

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class UpsampleBlock(nn.Module):
    """
    Upsampling block with interpolation and convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mode : {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}
        Upsampling interpolation mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'],
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample and convolve.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height*2, width*2).
        """
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """
    VAE decoder that maps latent vectors to images.

    Parameters
    ----------
    image_size : int
        Output image size (assumes square images).
    hidden_dims : list[int]
        Channel dimensions for each layer.
    latent_dim : int
        Dimension of latent space.
    activation : {'relu', 'leaky_relu', 'silu', 'gelu'}
        Activation function.
    norm_type : {'batch', 'group'}
        Normalization type.
    use_res : bool
        Use residual blocks.
    attn_at : list[int] or None
        Resolutions at which to apply attention (e.g., [8, 16]).
        If None, no attention is used.
    upsample_mode : {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}
        Upsampling mode.
    """

    def __init__(
        self,
        image_size: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batch', 'group'],
        use_res: bool,
        attn_at: list[int] | None,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'],
    ) -> None:
        super().__init__()

        if len(hidden_dims) < 2:
            raise ValueError(
                '`hidden_dims` must contain at least two entries: bottleneck channels and '
                'at least one upsampling stage.'
            )

        if attn_at is None:
            attn_at = []

        upsample_dims = hidden_dims[1:]

        self.initial_size: int = self._calculate_initial_size(image_size, len(upsample_dims))
        self.initial_channels: int = hidden_dims[0]
        self.initial_dim: int = self.initial_channels * self.initial_size * self.initial_size

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.initial_dim),
            get_activation(activation),
        )

        self.features = self._build_decoder_layers(
            self.initial_channels,
            upsample_dims,
            activation,
            norm_type,
            use_res,
            attn_at,
            self.initial_size,
            upsample_mode,
        )

    @staticmethod
    def _calculate_initial_size(image_size: int, num_layers: int) -> int:
        downsample_factor = 2**num_layers
        if image_size % downsample_factor != 0:
            raise ValueError(
                f'Decoder configuration expects image size divisible by {downsample_factor}, '
                f'got {image_size}. Adjust `hidden_dims` or `image_size`.'
            )
        return image_size // downsample_factor

    def _build_decoder_layers(
        self,
        in_channels: int,
        upsample_dims: list[int],
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batch', 'group'],
        use_res: bool,
        attn_at: list[int],
        initial_size: int,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'],
    ) -> nn.Sequential:
        layers = []
        current_resolution = initial_size

        if current_resolution in attn_at:
            layers.append(SelfAttentionBlock(in_channels, norm_type))

        for out_dim in upsample_dims:
            layers.append(UpsampleBlock(in_channels, out_dim, mode=upsample_mode))
            current_resolution *= 2

            if use_res:
                layers.append(
                    ResidualBlock(
                        out_dim,
                        out_dim,
                        stride=1,
                        activation=activation,
                        norm_type=norm_type,
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        get_norm_layer(
                            norm_type,
                            out_dim,
                            get_num_groups(in_channels) if norm_type == 'group' else None,
                        ),
                        get_activation(activation),
                    )
                )
            if current_resolution in attn_at:
                layers.append(SelfAttentionBlock(out_dim, norm_type))
            in_channels = out_dim
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors.

        Returns
        -------
        torch.Tensor
            Decoded images.
        """
        hidden = self.from_latent(z)
        hidden = rearrange(
            hidden,
            'b (c h w) -> b c h w',
            c=self.initial_channels,
            h=self.initial_size,
            w=self.initial_size,
        )
        output = self.features(hidden)
        return output


class VAE(BaseModel):
    """
    Variational Autoencoder for image generation.

    Parameters
    ----------
    image_size : int
        Size of square input images.
    in_channels : int
        Number of input image channels.
    hidden_dims : tuple[int, ...] or list[int]
        Channel dimensions for each layer.
    latent_dim : int
        Dimension of latent space.
    activation : {'relu', 'leaky_relu', 'silu', 'gelu'}
        Activation function.
    norm_type : {'batch', 'group'}
        Normalization type.
    use_res : bool
        Use residual blocks.
    attn_enc_at : list[int] or None, optional
        Encoder attention at specified resolutions (e.g., [16, 8]).
        If None, no attention is used in encoder, by default None.
    attn_dec_at : list[int] or None, optional
        Decoder attention at specified resolutions (e.g., [8, 16]).
        If None, no attention is used in decoder, by default None.
    upsample_mode : {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}, optional
        Upsampling mode, by default 'nearest'.
    output_range : {'sigmoid', 'tanh'}, optional
        Output activation, by default 'sigmoid'.
    init_mode : {'normal', 'uniform'}, optional
        Weight initialization mode, by default: 'normal'.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batch', 'group'],
        use_res: bool,
        attn_enc_at: list[int] | None = None,
        attn_dec_at: list[int] | None = None,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'nearest',
        output_range: Literal['sigmoid', 'tanh'] = 'sigmoid',
        init_mode: Literal['normal', 'uniform'] = 'normal',
    ) -> None:
        super().__init__()

        self.input_proj = nn.Conv2d(
            in_channels,
            hidden_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.encoder = Encoder(
            image_size=image_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
            use_res=use_res,
            attn_at=attn_enc_at,
        )
        self.decoder = Decoder(
            image_size=image_size,
            hidden_dims=hidden_dims[::-1],
            latent_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
            use_res=use_res,
            attn_at=attn_dec_at,
            upsample_mode=upsample_mode,
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(
                hidden_dims[0],
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid() if output_range == 'sigmoid' else nn.Tanh(),
        )

        self.latent_dim: int = latent_dim
        self._initialize_weights(activation, init_mode)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """
        Forward pass through VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input images.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'x_hat' (reconstruction), 'mu', and 'logvar'.
        """
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return {'x_hat': x_hat, 'mu': mu, 'logvar': logvar}

    def encode(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode images to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input images.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Latent sample, mean, and log variance.
        """
        x = self.input_proj(x)
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors.

        Returns
        -------
        torch.Tensor
            Reconstructed images.
        """
        output = self.decoder(z)
        output = self.output_proj(output)
        return output

    @torch.inference_mode()
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from random latent vectors.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : str, optional
            Device to generate on (default: 'cpu').

        Returns
        -------
        torch.Tensor
            Generated images.
        """
        was_training = self.training
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        output = self.decode(z)
        self.train(was_training)
        return output

    def _initialize_weights(self, activation: str, mode: Literal['normal', 'uniform']) -> None:
        """
        Initialize network weights.

        Parameters
        ----------
        activation : str
            Activation function type.
        mode : str
            Initialization mode ('normal' or 'uniform').
        """
        initialize_weights(self, activation=activation, mode=mode)
