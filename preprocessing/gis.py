import numpy as np
import rasterio
import shapely

from rasterio.features import geometry_mask
from rasterio.enums import Resampling
from fractions import Fraction
from typing import List, Tuple
from affine import Affine


def get_clip_mask(geometries : List[shapely.geometry.Polygon], transform : Affine, shape = tuple) -> np.ndarray:
    return geometry_mask(
        geometries = geometries,
        out_shape = shape,
        transform = transform,
        invert = True
    )


def remove_empty_columns_rows(data : np.ndarray) -> np.ndarray:
    if data.ndim not in {2, 3}:
        raise ValueError('data must have 2 o 3 dimensions')
    
    if data.ndim == 3:
        rows = np.where(~np.isnan(data[0]).all(axis = 1))[0]
        cols = np.where(~np.isnan(data[0]).all(axis = 0))[0]
        return data[:, rows, :][:, :, cols]
    elif data.ndim == 2:
        rows = np.where(~np.isnan(data).all(axis = 1))[0]
        cols = np.where(~np.isnan(data).all(axis = 0))[0]
        return data[rows, :][:, cols]
    

def resample(in_file : str, resampling : Resampling, scale_x : Fraction, scale_y : Fraction) -> Tuple[np.ndarray, dict]:
    with rasterio.open(in_file, 'r') as src:
        profile = src.profile.copy()
        transform = profile.get('transform')
        count, height, width = profile.get('count'), profile.get('height'), profile.get('width')

        out_height, out_width = int(height * scale_y), int(width * scale_x)
        out_shape = count, out_height, out_width
        dst_transform = transform * transform.scale( (width / out_width), (height / out_height) )

        profile.update(
            {
                "transform": dst_transform,
                "width": out_width,
                "height": out_height,
            }
        )
        
        data = src.read(out_shape = out_shape, resampling = resampling)
        
    return data, profile