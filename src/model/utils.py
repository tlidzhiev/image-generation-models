def get_num_groups(channels: int, max_groups: int = 32) -> int:
    """
    Get the number of groups for GroupNorm that divides channels evenly.

    Parameters
    ----------
    channels : int
        Number of channels.
    max_groups : int, optional
        Maximum number of groups to consider, by default 32.

    Returns
    -------
    int
        Number of groups that evenly divides channels.
    """
    for num_groups in [max_groups, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1
