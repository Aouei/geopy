import numpy as np


def is_valid(array : np.ndarray) -> np.ndarray:
    return ~np.isnan(array)

def is_lt(array : np.ndarray, value : float) -> np.ndarray:
    return array < value

def is_eq(array : np.ndarray, value : float) -> np.ndarray:
    return array == value

def is_gt(array : np.ndarray, value : float) -> np.ndarray:
    return array > value

def is_lte(array : np.ndarray, value : float) -> np.ndarray:
    return is_lt(array, value) | is_eq(array, value)

def is_gte(array : np.ndarray, value : float) -> np.ndarray:
    return is_gt(array, value) | is_eq(array, value)

def is_in_range(array : np.ndarray, vmin : float, vmax : float) -> np.ndarray:
    return is_gte(array, vmin) & is_lte(array, vmax)