import numpy as np


def is_valid(array : np.ndarray) -> np.ndarray:
    """Returns a mask of valid values in the array, excluding NaNs."""
    return ~np.isnan(array)

def is_lt(array : np.ndarray, value : float) -> np.ndarray:
    """Returns a mask of values in the array that are less than the given value."""
    return array < value

def is_eq(array : np.ndarray, value : float) -> np.ndarray:
    """Returns a mask of values in the array that are equal to the given value."""
    return array == value

def is_gt(array : np.ndarray, value : float) -> np.ndarray:
    """Returns a mask of values in the array that are greater than the given value."""
    return array > value

def is_lte(array : np.ndarray, value : float) -> np.ndarray:
    """Returns a mask of values in the array that are less than or equal to the given value."""
    return is_lt(array, value) | is_eq(array, value)

def is_gte(array : np.ndarray, value : float) -> np.ndarray:
    """Returns a mask of values in the array that are greater than or equal to the given value."""
    return is_gt(array, value) | is_eq(array, value)

def is_in_range(array : np.ndarray, vmin : float, vmax : float) -> np.ndarray:
    """"Returns a mask of values in the array that are within the given range [vmin, vmax]."""
    return is_gte(array, vmin) & is_lte(array, vmax)