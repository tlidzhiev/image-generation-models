from typing import Any

import torch


class collate_images_fn:
    """
    Collate function for image datasets.

    Converts individual dataset items into batches.
    """

    def __init__(self, use_condition: bool) -> None:
        """
        Parameters
        ----------
        use_condition : bool
            Whether to include conditional labels in the batch.
        """
        self.use_condition = use_condition

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate and pad fields in the dataset items.

        Converts individual items into a batch.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of objects from dataset.__getitem__.

        Returns
        -------
        dict[str, torch.Tensor]
            Dict containing batch-version of the tensors.
        """
        images: list[torch.Tensor]
        conditions: list[int]

        images, conditions = [], []
        for item in batch:
            images.append(item['x'])

            if self.use_condition:
                conditions.append(item['c'])

        x = torch.stack(images)
        c = torch.tensor(conditions, dtype=torch.long) if self.use_condition else None

        result_batch = {'x': x}
        if self.use_condition:
            result_batch.update({'c': c})
        return result_batch
