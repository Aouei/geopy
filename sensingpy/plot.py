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


def get_projection(crs: pyproj.CRS) -> ccrs.Projection:
    """
    Obtain cartopy projection from a CRS object for plotting.
    
    Parameters
    ----------
    crs : pyproj.CRS
        Custom object containing coordinate reference system data
        
    Returns
    -------
    ccrs.Projection
        Cartopy projection object suitable for plotting
        
    Raises
    ------
    ValueError
        If the CRS object is not valid or not supported
        
    Notes
    -----
    Currently supports UTM projections explicitly. Defaults to Mercator
    for other projection types.
    """
    projection = ccrs.Mercator()

    if hasattr(crs, 'to_dict') and 'proj' in crs.to_dict():
        proj_info = crs.to_dict()
        if proj_info['proj'] == 'utm':
            projection = ccrs.UTM(proj_info['zone'])
    else:
        raise ValueError("Invalid CRS object. Expected a pyproj CRS object or similar.")
        
    return projection


def get_geofigure(crs: pyproj.CRS, nrows: int, ncols: int, 
                 figsize: tuple = (12, 6), **kwargs) -> Tuple[Figure, Axes | List[Axes]]:
    """
    Generate matplotlib figure and axes with georeferenced projections.
    
    Parameters
    ----------
    crs : pyproj.CRS
        Coordinate reference system for the plot
    nrows : int
        Number of rows for the subplots
    ncols : int
        Number of columns for the subplots
    figsize : tuple, optional
        Dimensions in inches of the figure, by default (12, 6)
    **kwargs
        Additional keyword arguments passed to plt.subplots()
        
    Returns
    -------
    Tuple[Figure, Axes | List[Axes]]
        Figure object and either a single Axes object (if nrows=ncols=1)
        or a list of Axes objects
        
    Notes
    -----
    Creates a figure with axes that use the appropriate cartopy projection
    based on the provided CRS.
    """
    return plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, 
                        subplot_kw={'projection': get_projection(crs)}, **kwargs)


def plot_band(image: Image, band: str, ax: Axes, 
             cmap: str = 'viridis', **kwargs) -> Tuple[Axes, Any]:
    """
    Plot a single band of an Image object.
    
    Parameters
    ----------
    image : Image
        Image object containing bands data and coordinate reference system
    band : str
        Band name to be plotted
    ax : Axes
        Matplotlib axes on which to plot the data
    cmap : str, optional
        Colormap to use for visualization, by default 'viridis'
    **kwargs
        Additional keyword arguments passed to ax.pcolormesh()
        
    Returns
    -------
    Tuple[Axes, Any]
        Axes with the plotted data and the mappable object for creating a colorbar
        
    Examples
    --------
    >>> fig, ax = get_geofigure(image.crs, 1, 1)
    >>> ax, mappable = plot_band(image, 'nir', ax, cmap='inferno')
    >>> plt.colorbar(mappable, ax=ax, label='NIR Reflectance')
    """
    data = image.select(band)
    mappable = ax.pcolormesh(*image.xs_ys, data, cmap=cmap, **kwargs)
    return ax, mappable


def plot_rgb(image: Image, red: str, green: str, blue: str, ax: Axes, 
            brightness: float = 1, **kwargs) -> Axes:
    """
    Create an RGB visualization from three bands of an Image object.
    
    Parameters
    ----------
    image : Image
        Image object containing bands data and coordinate reference system
    red : str
        Band name to use for the red channel
    green : str
        Band name to use for the green channel
    blue : str
        Band name to use for the blue channel
    ax : Axes
        Matplotlib axes on which to plot the RGB image
    brightness : float, optional
        Value to multiply the RGB values to adjust brightness, by default 1
    **kwargs
        Additional keyword arguments passed to ax.pcolormesh()
        
    Returns
    -------
    Axes
        Axes with the RGB image plotted
        
    Notes
    -----
    This function handles both float (0-1) and uint8 (0-255) data types.
    Values are clipped to valid range after brightness adjustment.
    
    Examples
    --------
    >>> fig, ax = get_geofigure(image.crs, 1, 1)
    >>> ax = plot_rgb(image, 'red', 'green', 'blue', ax, brightness=1.5)
    >>> plt.title('True Color Composite')
    """
    rgb = np.dstack(image.select([red, green, blue]))
    limit = 1 if rgb.dtype != np.uint8 else 255

    rgb = np.clip(rgb * brightness, 0, limit)

    ax.pcolormesh(*image.xs_ys, rgb, **kwargs)    
    return ax


def add_basemap(ax: Axes, west: float, south: float, east: float, north: float, 
               crs: pyproj.CRS, source: Any = providers.OpenStreetMap.Mapnik) -> Axes:
    """
    Add a basemap to a cartopy axes using contextily.
    
    Parameters
    ----------
    ax : Axes
        Cartopy axes on which to plot the basemap
    west : float
        Western longitude boundary
    south : float
        Southern latitude boundary
    east : float
        Eastern longitude boundary
    north : float
        Northern latitude boundary
    crs : pyproj.CRS
        Coordinate reference system for the plot
    source : Any, optional
        Contextily basemap provider, by default providers.OpenStreetMap.Mapnik
        
    Returns
    -------
    Axes
        Axes with the basemap added
        
    Notes
    -----
    This function downloads tile data from the specified provider and displays it
    on the map. It creates a temporary GeoTIFF file that is removed after plotting.
    
    Examples
    --------
    >>> fig, ax = get_geofigure(image.crs, 1, 1)
    >>> ax = add_basemap(ax, image.left, image.bottom, image.right, image.top, 
    ...                 image.crs, providers.Stamen.Terrain)
    >>> ax = plot_rgb(image, 'red', 'green', 'blue', ax, alpha=0.7)
    """
    temp_file = '_temp.tif'

    try:
        cx.bounds2raster(west, south, east, north, path=temp_file, ll=True, source=source)
        image = reader.open(temp_file)            
        image.reproject(crs)

        rgb = np.moveaxis(image.values, 0, -1)
        rgb = np.clip(rgb, 0, 255).astype(np.float32) / 255
        
        ax.pcolormesh(*image.xs_ys, rgb)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return ax


def add_gridlines(ax: Axes, **kwargs) -> Tuple[Axes, Any]:
    """
    Add geographic gridlines to a cartopy axes.
    
    Parameters
    ----------
    ax : Axes
        Cartopy axes to which gridlines will be added
    **kwargs
        Additional keyword arguments passed to ax.gridlines()
        
    Returns
    -------
    Tuple[Axes, Any]
        Axes with added gridlines and the gridlines object for further customization
        
    Notes
    -----
    Labels on top and right edges are disabled by default.
    The returned gridlines object can be used for additional customization.
    
    Examples
    --------
    >>> fig, ax = get_geofigure(image.crs, 1, 1)
    >>> ax, gl = add_gridlines(ax, linestyle='--')
    >>> # Customize gridlines further if needed
    >>> gl.xlabel_style = {'size': 15}
    >>> gl.ylabel_style = {'color': 'gray'}
    """
    gl = ax.gridlines(draw_labels=True, **kwargs)
    gl.top_labels = gl.right_labels = False

    return ax, gl
