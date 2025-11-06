from typing import Literal

import torch
import torch.nn.functional as F

from .base import BaseLoss


class VAELoss(BaseLoss):
    """
    Loss function for Variational Autoencoder (VAE).

    Computes the combination of reconstruction loss and KL divergence.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reconstruction_loss: Literal['mse', 'bce'] = 'mse',
        per_element_mean: bool = False,  # False: sum over dims, then mean over batch
    ) -> None:
        """
        Parameters
        ----------
        beta : float, optional
            Weight for KL divergence term, by default 1.0.
        reconstruction_loss : {'mse', 'bce'}, optional
            Type of reconstruction loss, by default 'mse'.
        per_element_mean : bool, optional
            If True, use mean over dimensions; if False, use sum over dimensions,
            by default False.
        """
        super().__init__()
        self.beta: float = beta
        self.reconstruction_loss: Literal['mse', 'bce'] = reconstruction_loss
        self.per_element_mean: bool = per_element_mean
        self.loss_names: list[str] = ['loss', 'recon_loss', 'kl_loss']

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Parameters
        ----------
        x : torch.Tensor
            Original input data.
        x_hat : torch.Tensor
            Reconstructed data from decoder.
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log variance of latent distribution.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'loss', 'recon_loss', and 'kl_loss' keys.
        """
        if self.reconstruction_loss == 'mse':
            per_sample = F.mse_loss(x_hat, x, reduction='none').flatten(1)
            recon = per_sample.mean(1) if self.per_element_mean else per_sample.sum(1)
        elif self.reconstruction_loss == 'bce':
            per_sample = F.binary_cross_entropy(x_hat, x, reduction='none').flatten(1)
            recon = per_sample.mean(1) if self.per_element_mean else per_sample.sum(1)
        else:
            raise ValueError(
                f"reconstruction_loss must be 'mse' or 'bce', got {self.reconstruction_loss}"
            )
        recon = recon.mean()  # mean over batch

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.flatten(1).sum(1).mean()  # sum over latent dim, mean over batch

        loss = recon + self.beta * kl
        return {'loss': loss, 'recon_loss': recon, 'kl_loss': kl}

    def __repr__(self) -> str:
        """
        Return string representation of the loss.

        Returns
        -------
        str
            String with loss configuration.
        """
        return (
            f'{type(self).__name__}(beta={self.beta}, '
            f'reconstruction_loss="{self.reconstruction_loss}", '
            f'per_element_mean={self.per_element_mean})'
        )
