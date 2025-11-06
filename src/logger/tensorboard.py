from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.logger.base import BaseWriter
from src.utils.io import get_root


class TensorboardWriter(BaseWriter):
    """
    Tensorboard experiment tracker writer.

    Logs metrics, images, and hyperparameters to TensorBoard.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        log_dir: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        project_config : dict[str, Any]
            Configuration dictionary for the project.
        project_name : str
            Name of the project.
        log_dir : str or None, optional
            Directory for TensorBoard logs, by default None.
            If None, uses 'runs/{project_name}'.
        run_id : str or None, optional
            Unique identifier for the run, by default None.
        run_name : str or None, optional
            Human-readable name for the run, by default None.
        **kwargs
            Additional keyword arguments passed to SummaryWriter.
        """
        super().__init__(project_config, project_name, run_id, run_name, **kwargs)

        if log_dir is None:
            log_dir = f'runs/{project_name}'
            if run_name is not None:
                log_dir = f'{log_dir}/{run_name}'
            if run_id is not None:
                log_dir = f'{log_dir}_{run_id}'

        self.writer = SummaryWriter(log_dir=str(get_root() / log_dir), **kwargs)

        # Log hyperparameters
        self._log_config(project_config)

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """
        Log checkpoints to the experiment tracker (not supported by TensorBoard).

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        save_dir : str
            Base path for the checkpoint.

        Notes
        -----
        TensorBoard doesn't support checkpoint storage.
        Checkpoints should be saved separately.
        """
        # TensorBoard doesn't support checkpoint storage
        # Checkpoints should be saved separately
        pass

    def add_scalar(self, name: str, value: float) -> None:
        """
        Log a scalar to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the scalar.
        value : float
            Value of the scalar.
        """
        self.writer.add_scalar(self._object_name(name), value, global_step=self.step)

    def add_scalars(self, values: dict[str, float]) -> None:
        """
        Log several scalars to the experiment tracker.

        Parameters
        ----------
        values : dict[str, float]
            Dict containing scalar names and values.
        """
        for name, value in values.items():
            self.writer.add_scalar(self._object_name(name), value, global_step=self.step)

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        """
        Log an image to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the image.
        image : np.ndarray or Path or str
            Image as numpy array or path to image file.
        """
        if isinstance(image, (str, Path)):
            from PIL import Image

            image = np.array(Image.open(image))

        # Convert HWC to CHW if needed
        if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))

        self.writer.add_image(self._object_name(name), image, global_step=self.step)

    def finish(self) -> None:
        """
        Finalize and close the experiment tracker.
        """
        self.writer.flush()
        self.writer.close()

    def _log_config(self, config: dict[str, Any]) -> None:
        """
        Log configuration to TensorBoard as text.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary.
        """
        # Flatten nested dict for TensorBoard
        flattened = self._flatten_dict(config)
        # Convert to text
        config_text = '\n'.join([f'{k}: {v}' for k, v in flattened.items()])
        self.writer.add_text('config', config_text, global_step=0)

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '/') -> dict:
        """
        Flatten nested dictionary.

        Parameters
        ----------
        d : dict
            Dictionary to flatten.
        parent_key : str, optional
            Parent key prefix, by default ''.
        sep : str, optional
            Separator for nested keys, by default '/'.

        Returns
        -------
        dict
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
