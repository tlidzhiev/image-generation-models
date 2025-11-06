import os
import random

import numpy as np
import torch


def set_worker_seed(worker_id: int) -> None:
    """
    Set seed for each dataloader worker.

    Parameters
    ----------
    worker_id : int
        ID of the worker.

    Notes
    -----
    For more info, see https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for model training or inference.

    Parameters
    ----------
    seed : int
        Defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed % (2**32 - 1))
    os.environ['PYTHONHASHSEED'] = str(seed % (2**32 - 1))
