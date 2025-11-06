import torch
import torch.nn.functional as F

from .base import BaseLoss


class DDPMLoss(BaseLoss):
    """
    Loss function for Denoising Diffusion Probabilistic Models (DDPM).

    Computes MSE loss between true noise and predicted noise.
    """

    loss_names: list[str] = ['loss']

    def forward(
        self,
        eps: torch.Tensor,
        eps_theta: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute DDPM loss.

        Parameters
        ----------
        eps : torch.Tensor
            True noise added to the input.
        eps_theta : torch.Tensor
            Predicted noise from the model.
        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'loss' key containing the computed loss.
        """
        recon = F.mse_loss(eps, eps_theta, reduction='none').flatten(1)
        recon = recon.sum(dim=1).mean()
        return {'loss': recon}
