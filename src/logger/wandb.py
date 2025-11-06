from pathlib import Path
from typing import Any, Literal

import numpy as np
import wandb

from src.logger.base import BaseWriter


class WandBWriter(BaseWriter):
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        entity: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        mode: Literal['online', 'offline', 'disabled', 'shared'] | None = 'offline',
        save_code: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        project_config : dict[str, Any]
            Configuration dictionary for the project.
        project_name : str
            Name of the project.
        entity : str or None, optional
            Wandb entity (username or team name), by default None.
        run_id : str or None, optional
            Unique identifier for the run, by default None.
        run_name : str or None, optional
            Human-readable name for the run, by default None.
        mode : {'online', 'offline', 'disabled', 'shared'} or None, optional
            Wandb run mode, by default 'offline'.
        save_code : bool, optional
            Whether to save code to wandb, by default False.
        **kwargs
            Additional keyword arguments passed to wandb.init.
        """
        super().__init__(project_config, project_name, run_id, run_name, **kwargs)

        wandb.login()

        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            resume='allow',
            id=run_id,
            mode=mode,
            save_code=save_code,
        )

        self.wandb = wandb

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """
        Log checkpoints to the experiment tracker.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        save_dir : str
            Base path for the checkpoint.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

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
        self.wandb.log(
            {self._object_name(name): value},
            step=self.step,
        )

    def add_scalars(self, values: dict[str, float]) -> None:
        """
        Log several scalars to the experiment tracker.

        Parameters
        ----------
        values : dict[str, float]
            Dict containing scalar names and values.
        """
        self.wandb.log(
            {self._object_name(k): v for k, v in values.items()},
            step=self.step,
        )

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
        self.wandb.log(
            {self._object_name(name): self.wandb.Image(image)},
            step=self.step,
        )

    def finish(self) -> None:
        """
        Finalize and close the experiment tracker.
        """
        self.wandb.finish()
