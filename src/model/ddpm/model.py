import torch
from tqdm.auto import tqdm

from ..base import BaseModel
from .noise_scheduler import NoiseScheduler
from .unet import UNet


class DDPM(BaseModel):
    """
    Denoising Diffusion Probabilistic Model (DDPM).

    Implements the DDPM architecture with UNet backbone and noise scheduler
    for image generation.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        num_channels: int,
        ch_mults: list[int],
        use_attn: list[bool],
        num_blocks: int = 1,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        """
        Parameters
        ----------
        image_shape : tuple[int, int, int]
            Shape of input images (channels, height, width).
        num_channels : int, optional
            Base number of channels in UNet, by default 64.
        ch_mults : list[int]
            Channel multipliers for each resolution level.
        use_attn : list[bool]
            Whether to use attention at each resolution level.
        num_blocks : int, optional
            Number of residual blocks per resolution level, by default 1.
        num_steps : int, optional
            Number of diffusion steps, by default 1000.
        beta_start : float, optional
            Starting beta value for noise schedule, by default 0.0001.
        beta_end : float, optional
            Ending beta value for noise schedule, by default 0.02.
        """
        super().__init__()
        self.image_shape: tuple[int, int, int] = image_shape
        self.backbone = UNet(
            in_channels=self.image_shape[0],
            num_channels=num_channels,
            ch_mults=ch_mults,
            use_attn=use_attn,
            num_blocks=num_blocks,
        )
        self.noise_scheduler = NoiseScheduler(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    def forward(
        self,
        x: torch.Tensor,
        eps: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Parameters
        ----------
        x : torch.Tensor
            Clean input images.
        eps : torch.Tensor or None, optional
            Noise tensor. If None, random noise is generated, by default None.
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'xt' (noised images), 'eps_theta' (predicted noise),
            and 'eps' (true noise).
        """
        x0 = x
        if eps is None:
            eps = torch.randn_like(x0)

        batch_size = x0.shape[0]
        t = torch.randint(
            low=0,
            high=self.noise_scheduler.num_steps,
            size=(batch_size,),
            device=x0.device,
            dtype=torch.long,
        )
        xt = self.noise(x0=x0, t=t, eps=eps)
        eps_theta = self.backbone(xt, t)
        return {'xt': xt, 'eps_theta': eps_theta, 'eps': eps}

    def noise(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add noise to clean images at timestep t.

        Parameters
        ----------
        x0 : torch.Tensor
            Clean input images.
        eps : torch.Tensor
            Noise tensor.
        t : torch.Tensor or None, optional
            Timestep indices. If None, random timesteps are sampled, by default None.

        Returns
        -------
        torch.Tensor
            Noised images at timestep t.
        """
        if t is None:
            batch_size = x0.shape[0]
            t = torch.randint(
                low=0,
                high=self.noise_scheduler.num_steps,
                size=(batch_size,),
                device=x0.device,
                dtype=torch.long,
            )
        return self.noise_scheduler(x0=x0, t=t, eps=eps)

    def denoise(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Denoise images from noisy input.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input images.

        Returns
        -------
        torch.Tensor
            Denoised images.
        """
        return self._denoise(xt, 'Denoising...')

    @torch.inference_mode()
    def sample(self, num_samples: int, device: str) -> torch.Tensor:
        """
        Generate samples from random noise.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : str
            Device to generate on.

        Returns
        -------
        torch.Tensor
            Generated images.
        """
        was_training = self.training
        self.eval()
        xt = torch.randn(num_samples, *self.image_shape, device=device)
        x0 = self._denoise(xt, 'Sampling...')
        self.train(was_training)
        return x0

    @torch.inference_mode()
    def _denoise(self, xt: torch.Tensor, desc: str | None = None) -> torch.Tensor:
        """
        Internal denoising loop.

        Parameters
        ----------
        xt : torch.Tensor
            Noisy input images.
        desc : str or None, optional
            Progress bar description, by default None.

        Returns
        -------
        torch.Tensor
            Denoised images clamped to [-1, 1].
        """
        self.eval()
        batch_size = xt.shape[0]
        T = self.noise_scheduler.num_steps
        for step in tqdm(reversed(range(T)), desc=desc, total=self.noise_scheduler.num_steps):
            t = torch.full((batch_size,), step, device=xt.device, dtype=torch.long)
            eps_theta = self.backbone(xt, t)
            xt = self.noise_scheduler.inverse(
                xt=xt,
                t=t,
                eps=eps_theta,
                add_noise=(step > 0),
            )
        return xt.clamp_(-1.0, 1.0)
