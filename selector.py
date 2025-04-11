import numpy as np

from typing import Iterable
from itertools import pairwise


def interval_choice(array : np.ndarray, size : int, intervals : Iterable) -> np.ndarray:
    limit_masks = [ (vmin <= array) & (array <= vmax) for vmin, vmax in pairwise(intervals) ]
    return np.array([ np.random.choice(array[in_limits], size) for in_limits in limit_masks ]).ravel()

def arginterval_choice(array : np.ndarray, size : int, intervals : Iterable) -> np.ndarray:
    indexes = np.arange(array.size)
    limit_masks = [ (vmin <= array) & (array <= vmax) for vmin, vmax in pairwise(intervals) ]
    return np.array([ np.random.choice(indexes[in_limits], size) for in_limits in limit_masks ]).ravel()

def naninterval_choice(array : np.ndarray, size : int, intervals : Iterable) -> np.ndarray:
    no_nan_mask = ~np.isnan(array)
    limit_masks = [ (vmin <= array) & (array <= vmax) for vmin, vmax in pairwise(intervals) ]
    return np.array([ np.random.choice(array[no_nan_mask & in_limits], size) for in_limits in limit_masks ]).ravel()

def nanarginterval_choice(array : np.ndarray, size : int, intervals : Iterable) -> np.ndarray:
    indexes = np.arange(array.size)
    no_nan_mask = ~np.isnan(array)
    limit_masks = [ (vmin <= array) & (array <= vmax) for vmin, vmax in pairwise(intervals) ]
    return np.array([ np.random.choice(indexes[no_nan_mask & in_limits], size) for in_limits in limit_masks ]).ravel()