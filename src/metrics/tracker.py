import pandas as pd


class MetricTracker:
    """
    Class to aggregate metrics from many batches.

    Uses pandas DataFrame internally to track running totals,
    counts, and averages for each metric.
    """

    def __init__(self, *keys: str) -> None:
        """
        Parameters
        ----------
        *keys : str
            Metric names (may include the names of losses).
        """
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self) -> None:
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float) -> None:
        """
        Update metrics DataFrame with new value.

        Parameters
        ----------
        key : str
            Metric name.
        value : float
            Metric value on the batch.
        """
        self._data.loc[key, 'total'] += value
        self._data.loc[key, 'counts'] += 1
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]

    def __getitem__(self, key: str) -> float:
        """
        Return average value for a given metric.

        Parameters
        ----------
        key : str
            Metric name.

        Returns
        -------
        float
            Average value for the metric.
        """
        return self._data.average[key]

    def result(self) -> dict[str, float]:
        """
        Return average value of each metric.

        Returns
        -------
        dict[str, float]
            Dict containing average metrics for each metric name.
        """
        return dict(self._data.average)

    def keys(self) -> list[str]:
        """
        Return all metric names defined in the MetricTracker.

        Returns
        -------
        list[str]
            All metric names in the table.
        """
        return list(self._data.total.keys())
