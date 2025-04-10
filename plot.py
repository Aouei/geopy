import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from typing import List, Optional
import matplotlib.axes as mplaxes

import contextily as cx
from contextily import providers
import os
import reader


def get_projection(image):
    if hasattr(image.crs, 'to_dict') and 'proj' in image.crs.to_dict():
        proj_info = image.crs.to_dict()
        if proj_info['proj'] == 'utm':
            projection = ccrs.UTM(proj_info.get('zone', 30))
        else:
            projection = ccrs.Projection(image.crs)
    else:
        projection = ccrs.Mercator()

    return projection


def get_geofigure(image, nrows, ncols, figsize = (12, 6), **kwargs) -> tuple:
    return plt.subplots(ncols = ncols, nrows = nrows, figsize=figsize, 
                        subplot_kw={'projection': get_projection(image)}, **kwargs)


def plot_band(image, band : str, ax: Optional[mplaxes.Axes] = None, figsize: tuple = (12, 6), 
              cmap: str = 'viridis', pcolormesh_kwargs = None) -> tuple:
    
    pcolormesh_kwargs = {} if pcolormesh_kwargs is None else pcolormesh_kwargs
    
    if ax is None:
        fig, ax = get_geofigure(image, 1, 1, figsize)
    else:
        fig = ax.figure

    data = image.select(band)
    mappable = ax.pcolormesh(*image.xs_ys, data, cmap=cmap, **pcolormesh_kwargs)
    
    return fig, ax, mappable


def add_basemap(ax, west, south, east, north, crs, source = providers.OpenStreetMap.Mapnik):
    temp_file = '_temp.tif'

    try:
        cx.bounds2raster(west, south, east, north, path = temp_file, ll = True, source = source)
        image = reader.open(temp_file)            
        image.reproject(crs)

        rgb = np.moveaxis(image.values, 0, -1)
        rgb = np.clip(rgb, 0, 255).astype(np.float32) / 255
        
        ax.pcolormesh(*image.xs_ys, rgb)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return ax