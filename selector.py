import numpy as np

from typing import Iterable, List, Callable
from itertools import pairwise


def _get_limit_masks(array : np.array, pairs : Iterable) -> List[np.array]:
    """Generates a list of boolean masks for the given array based on the provided pairs of limits.

    Args:
        array (np.array): array to be masked
        pairs (Iterable): list of pairs of limits

            e.g. [(0, 1), (1, 2), (2, 3)]

    Returns:
        List[np.array]: list of boolean masks for the given array based on the provided pairs of limits
    """

    return [ (vmin <= array) & (array < vmax) for vmin, vmax in pairs ]

def interval_choice(array : np.ndarray, size : int, intervals : Iterable, replace = True) -> np.ndarray:
    """Generates a random sample from the given array based on the provided intervals.

    Args:
        array (np.ndarray): array to be sampled
        size (int): size for each interval
        intervals (Iterable): intervals to be sampled from
            
                e.g. [(0, 1), (1, 2), (2, 3)]
        replace (bool, optional): np.choice sample argument. Defaults to True.

    Returns:
        np.ndarray: Random sample from the given array based on the provided intervals.
    """

    limit_masks = _get_limit_masks(array, pairwise(intervals))
    return np.array([ np.random.choice(array[in_limits], size, replace = replace) for in_limits in limit_masks ]).ravel()

def arginterval_choice(array : np.ndarray, size : int, intervals : Iterable, replace = True) -> np.ndarray:
    """Generates the indexes of a random sample from the given array based on the provided intervals.

    Args:
        array (np.ndarray): array to be sampled
        size (int): size for each interval
        intervals (Iterable): intervals to be sampled from
            
                e.g. [(0, 1), (1, 2), (2, 3)]
        replace (bool, optional): np.choice sample argument. Defaults to True.

    Returns:
        np.ndarray: The indexes of a random sample from the given array based on the provided intervals.
    """

    indexes = np.arange(array.size)
    limit_masks = _get_limit_masks(array, pairwise(intervals))
    return np.array([ np.random.choice(indexes[in_limits], size, replace = replace) for in_limits in limit_masks ]).ravel()

def composite(arrays : np.ndarray, method : Callable | np.ndarray = np.nanmax) -> np.ndarray:
    """Generates a synthetic array based on the provided method.

    Args:
        arrays (np.ndarray): a list of arrays to be composed
        method (Callable | np.ndarray, optional): numpy funcion or list of indexes to compose the final array. Defaults to np.nanmax.

    Returns:
        np.ndarray: a synthetic array based on the provided method.
    """

    if isinstance(method, np.ndarray):
        m,n = method.shape
        i, j = np.ogrid[:m,:n]
        return arrays[method, i, j]
    
    else:
        return method(arrays, axis = 0)
    
def argcomposite(arrays : np.ndarray, method : Callable = np.argmax) -> np.ndarray:
    """Generates the indexes of a synthetic array based on the provided method.

    Args:
        arrays (np.ndarray): the array to be composed
        method (Callable, optional): a numpy method such as argmax. Defaults to np.argmax.

    Returns:
        np.ndarray: The indexes of a synthetic array
    """

    nans = np.isnan(arrays).all(axis = 0)
    arrays[:, nans] = np.inf
    indexes = method(arrays, axis = 0)
    arrays[:, nans] = np.nan
    return indexes