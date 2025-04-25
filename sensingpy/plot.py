import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import contextily as cx
import numpy as np
import pyproj
import os

from matplotlib.pyplot import Figure
from typing import Tuple, List, Any
from matplotlib.axes import Axes
from contextily import providers


import sensingpy.reader as reader
from sensingpy.image import Image


def get_projection(crs : pyproj.CRS) -> ccrs.Projection:
    """Obtains the projection of the crs for plotting.

    Args:
        crs (pyproj.CRS): custom object containing crs data.

    Returns:
        ccrs.Projection: cartopy projection for plotting.

    Raises:
        ValueError: If the CRS object is not valid or not supported.
    """
    
    projection = ccrs.Mercator()

    if hasattr(crs, 'to_dict') and 'proj' in crs.to_dict():
        proj_info = crs.to_dict()
        if proj_info['proj'] == 'utm':
            projection = ccrs.UTM(proj_info['zone'])
    else:
        raise ValueError("Invalid CRS object. Expected a pyproj CRS object or similar.")
        
    return projection


def get_geofigure(crs : pyproj.CRS, nrows : int, ncols : int, figsize : tuple = (12, 6), **kwargs) -> Tuple[Figure, Axes | List[Axes]]:
    """Generates a subplots with georeferenced axes.

    Args:
        crs (pyproj): crs
        nrows (int): number of rows for the subplots.
        ncols (int): number of columns for the subplots.
        figsize (tuple, optional): dimensions in inches of the figure. Defaults to (12, 6).

    Returns:
        Tuple[Figure, Axes | List[Axes]]: Figure and axes objects for plotting.
    """

    return plt.subplots(ncols = ncols, nrows = nrows, figsize=figsize, 
                        subplot_kw={'projection': get_projection(crs)}, **kwargs)


def plot_band(image : Image, band : str, ax : Axes, cmap : str = 'viridis', **kwargs) -> Tuple[Axes, Any]:
    """Plots a band of the image object

    Args:
        image (Image): object with crs data and bands data.
        band (str): band name to be plotted.
        ax (Axes): axes where show the data.
        cmap (str, optional): colomap to use. Defaults to 'viridis'.

    Returns:
        Tuple[Axes, Any]: axes with data plotted and mappable object for colorbar.
    """

    data = image.select(band)
    mappable = ax.pcolormesh(*image.xs_ys, data, cmap=cmap, **kwargs)
    return ax, mappable


def plot_rgb(image : Image, red : str, green : str, blue : str, ax : Axes, brightness : float = 1, **kwargs) -> Axes:
    """Plots a RGB image on the given axes based on the selected bands of the image object.

    Args:
        image (Image): object with crs data and bands data.
        red (str): band name for red channel.
        green (str): band name for green channel.
        blue (str): band name for blue channel.
        ax (Axes): axes where to plot.
        brightness (float, optional): value to multiply the RGB. Defaults to 1.

    Returns:
        Axes: axes with the RGB image plotted.
    """

    rgb = np.dstack(image.select([red, green, blue]))
    limit = 1 if rgb.dtype != np.uint8 else 255

    rgb = np.clip(rgb * brightness, 0, limit)

    ax.pcolormesh(*image.xs_ys, rgb, **kwargs)    
    return ax

def add_basemap(ax : Axes, west : float, south : float, east : float, north : float, 
                crs : pyproj.CRS, source : Any = providers.OpenStreetMap.Mapnik) -> Axes:
    """Plots a RGB image on the given axes using contextily.

    Args:
        ax (Axes): axes where the image will be plotted.
        west (float): min longitude of the image.
        south (float): min latitude of the image.
        east (float): max longitude of the image.
        north (float): max latitude of the image.
        crs (pyproj.CRS): CRS of the image.
        source (Any, optional): basemap to use. Defaults to providers.OpenStreetMap.Mapnik.

    Returns:
        Axes: axes with the basemap plotted.
    """

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

def add_gridlines(ax : Axes, **kwargs) -> Tuple[Axes, Any]:
    """Add geographic gridlines to a cartopy axes.

    Adds latitude/longitude gridlines to a cartopy axes with labels on bottom and left edges.
    Labels on top and right edges are disabled by default.

    Args:
        ax (Axes): The cartopy axes to add gridlines to
        **kwargs: Additional keyword arguments passed to ax.gridlines()

    Returns:
        Tuple[Axes, Any]: Tuple containing:
            - The axes with added gridlines
            - The gridlines object for further customization

    Example:
        >>> fig, ax = get_geofigure(image, 1, 1)
        >>> ax, gl = add_gridlines(ax, linestyle='--')
        >>> # Customize gridlines further if needed
        >>> gl.xlabel_style = {'size': 15}
    """
    
    gl = ax.gridlines(draw_labels = True, **kwargs)
    gl.top_labels = gl.right_labels = False

    return ax, gl
