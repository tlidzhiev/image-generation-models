class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, name: str | None = None, *args, **kwargs):
        """
        Parameters
        ----------
        name : str or None, optional
            Metric name to use in logger and writer, by default None.
            If None, uses class name.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """

        self.name = name if name is not None else type(self).__name__

    def update(self, **batch):
        """
        Update metric state with batch data.

        Parameters
        ----------
        **batch
            Batch data containing inputs and targets.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement update method')

    def __call__(self, **batch) -> float:
        """
        Defines metric calculation logic for a given batch.

        Can use external functions (like TorchMetrics) or custom ones.

        Parameters
        ----------
        **batch
            Batch data containing inputs and targets.

        Returns
        -------
        float
            Computed metric value.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement __call__ method')

    def __str__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation.
        """
        return f'{type(self).__name__}()'
