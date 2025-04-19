import numpy as np

from typing import Iterable, List, Callable
from itertools import pairwise


def _get_limit_masks(array : np.array, pairs : Iterable) -> List[np.array]:
    return [ (vmin <= array) & (array < vmax) for vmin, vmax in pairs ]

def interval_choice(array : np.ndarray, size : int, intervals : Iterable, replace = True) -> np.ndarray:
    limit_masks = _get_limit_masks(array, pairwise(intervals))
    return np.array([ np.random.choice(array[in_limits], size, replace = replace) for in_limits in limit_masks ]).ravel()

def arginterval_choice(array : np.ndarray, size : int, intervals : Iterable, replace = True) -> np.ndarray:
    indexes = np.arange(array.size)
    limit_masks = _get_limit_masks(array, pairwise(intervals))
    return np.array([ np.random.choice(indexes[in_limits], size, replace = replace) for in_limits in limit_masks ]).ravel()

def composite(arrays : np.ndarray, method : Callable | np.ndarray = np.nanmax) -> np.ndarray:
    if isinstance(method, np.ndarray):
        m,n = method.shape
        i, j = np.ogrid[:m,:n]
        return arrays[method, i, j]
    
    else:
        return method(arrays, axis = 0)