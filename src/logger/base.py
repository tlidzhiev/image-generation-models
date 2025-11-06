from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np


class BaseWriter:
    """
    Base class for experiment tracking writers.

    Provides common interface for logging metrics, images, and checkpoints.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
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
        run_id : str or None, optional
            Unique identifier for the run, by default None.
        run_name : str or None, optional
            Human-readable name for the run, by default None.
        **kwargs
            Additional keyword arguments.
        """
        self.project_config = project_config
        self.project_name = project_name
        self.run_id = run_id
        self.run_name = run_name
        self.mode = ''
        self.step = 0
        self.timer = datetime.now()

    def set_step(self, step: int, mode: Literal['train', 'val', 'test'] = 'train') -> None:
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Parameters
        ----------
        step : int
            Current step.
        mode : {'train', 'val', 'test'}, optional
            Current mode (partition name), by default 'train'.
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', (self.step - previous_step) / duration.total_seconds())
            self.timer = datetime.now()

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """
        Log checkpoints to the experiment tracker.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        save_dir : str
            Path to the directory where checkpoint is saved.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_checkpoint method')

    def add_scalar(self, name: str, value: float) -> None:
        """
        Log a scalar to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the scalar to use in the tracker.
        value : float
            Value of the scalar.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_scalar method')

    def add_scalars(self, values: dict[str, float]) -> None:
        """
        Log several scalars to the experiment tracker.

        Parameters
        ----------
        values : dict[str, float]
            Dict containing scalar names and values.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_scalars method')

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        """
        Log an image to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the image to use in the tracker.
        image : np.ndarray or Path or str
            Image as numpy array or path to image file.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_image method')

    def finish(self) -> None:
        """
        Finalize and close the experiment tracker.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement finish method')

    def _object_name(self, name: str) -> str:
        """
        Update object name (scalar, image, etc.) with the current mode.

        Used to separate metrics from different partitions.

        Parameters
        ----------
        name : str
            Current object name.

        Returns
        -------
        str
            Updated object name with mode prefix.
        """
        return f'{self.mode}_{name}'
