import numpy as np

from scipy.stats import zscore


def IQR(data : np.ndarray, distance : float = 1.5) -> np.ndarray:
    Q1 = np.nanpercentile(data.reshape(-1), 25)
    Q3 = np.nanpercentile(data.reshape(-1), 75)
    outlier_distance = distance * (Q3 - Q1)

    return np.where( (data <= (Q1 - outlier_distance)) | (data >= (Q3 + outlier_distance)), np.nan, data)

def Z_Score(data : np.ndarray) -> np.ndarray:
    z_scores = np.abs(zscore(data))

    return np.where(z_scores >= 3, np.nan, data)

def upper_percentile(data : np.ndarray, percentile : float) -> np.ndarray:
    limit = np.nanpercentile(data.reshape(-1), percentile)
    return np.where( data >= limit, np.nan, data)

def lower_percentile(data : np.ndarray, percentile : float) -> np.ndarray:
    limit = np.nanpercentile(data.reshape(-1), percentile)

    return np.where( data <= limit, np.nan, data)