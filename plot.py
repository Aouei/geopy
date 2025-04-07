import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from typing import List, Optional
import matplotlib.axes as mplaxes

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
              cmap: str = 'viridis', **pcolormesh_kwargs) -> tuple:
        
    if ax is None:
        fig, ax = get_geofigure(image, 1, 1, figsize)
    else:
        fig = ax.figure

    data = image.select(band)
    mappable = ax.pcolormesh(*image.coords, data, cmap=cmap, **pcolormesh_kwargs)
    
    return fig, ax, mappable