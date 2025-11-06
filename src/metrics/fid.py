from typing import Literal

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.utils.torch import get_dtype

from .base import BaseMetric


class FIDMetric(BaseMetric):
    def __init__(
        self,
        feature: int,
        device: str,
        name: str | None = None,
        dtype: Literal['float32', 'float64'] = 'float32',
    ) -> None:
        """
        Initialize FID metric.

        Parameters
        ----------
        feature : int
            Feature dimension for Inception network (64, 192, 768, or 2048).
        device : str
            Device to run metric computation on ('cuda', 'cpu', or 'auto').
        name : str, optional
            Name of the metric, by default None.
        dtype : torch.dtype, optional
            Data type for computations, by default torch.float32.
        """
        super().__init__(name=name)
        self.feature = feature
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fid = (
            FrechetInceptionDistance(
                feature=feature,
                normalize=True,
            )
            .to(device)
            .set_dtype(get_dtype(dtype))
        )

    def update(self, fake: torch.Tensor, real: torch.Tensor, **kwargs) -> None:
        """
        Update FID metric with generated and real images.

        Parameters
        ----------
        fake : torch.Tensor
            Generated (fake) images tensor.
        real : torch.Tensor
            Real images tensor.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        Images are normalized to [0, 1] range before updating the metric.
        """
        with torch.no_grad():
            fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
            real = (real - real.min()) / (real.max() - real.min() + 1e-8)

        self.fid.update(fake, real=False)  # ty: ignore[ invalid-argument-type]
        self.fid.update(real, real=True)  # ty: ignore[ invalid-argument-type]

    def __call__(
        self,
        fake: torch.Tensor | None = None,
        real: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        """
        Compute FID score and reset internal state.

        Parameters
        ----------
        fake : torch.Tensor or None, optional
            Generated (fake) images tensor, by default None.
        real : torch.Tensor or None, optional
            Real images tensor, by default None.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Computed FID score.

        Notes
        -----
        This method computes the final FID score and automatically resets
        the metric state for the next epoch.
        """
        if fake is not None and real is not None:
            self.update(fake=fake, real=real)
        value = self.fid.compute().item()  # ty: ignore[missing-argument]
        self.fid.reset()
        return value

    def __str__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation with feature dimension.
        """
        return f'{type(self).__name__}(feature={self.feature})'
