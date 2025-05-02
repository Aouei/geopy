# from __future__ import annotations

import xarray as xr
import numpy as np
import rasterio
import pyproj
import sensingpy.enums as enums

from typing import Dict
from sensingpy.image import Image


class ImageReader:
    """
    Base class for reading geospatial images from different file formats.
    
    This is an abstract base class that defines the interface for all image readers.
    Concrete subclasses must implement the `read` method.
    
    Notes
    -----
    This class follows the Strategy pattern to provide different algorithms
    for reading various geospatial file formats while maintaining a consistent
    interface.
    """

    def read(self, filename: str) -> Image:
        """
        Read an image file and convert it to an Image object.
        
        Parameters
        ----------
        filename : str
            Path to the image file to be read
            
        Returns
        -------
        Image
            A processed image with spatial reference information
            
        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")


class NetCDFReader(ImageReader):
    """
    Reader for NetCDF image files with geospatial metadata.
    
    This class reads NetCDF (.nc, .nc4, .netcdf) files and extracts image data,
    coordinate systems, and spatial metadata to create an Image object. It handles
    various CF-convention metadata structures and fallback options for CRS information.
    
    Notes
    -----
    NetCDF files can store coordinate reference system (CRS) information in several
    locations according to the CF conventions. This reader searches for CRS information
    in coordinates, variables, and global attributes, in that order.
    """

    def read(self, filename: str) -> Image:
        """
        Read a NetCDF file and convert it to an Image object.
        
        Parameters
        ----------
        filename : str
            Path to the NetCDF file
            
        Returns
        -------
        Image
            An Image object containing the data, coordinates, and CRS information
            
        Notes
        -----
        The reader searches for CRS information in the following locations:
        1. DataArrays with a 'crs_wkt' attribute in coordinates
        2. Variables with a 'crs_wkt' attribute
        3. Global attributes ('crs_wkt' or 'proj4_string')
        
        If no CRS information is found, a default grid mapping variable is created.
        """
        grid_mapping = 'projection'
        
        with xr.open_dataset(filename) as src:
            crs = None
            crs_var_name = None
            
            # Search for a DataArray with crs_wkt attribute in coordinates
            for coord_name, coord in src.coords.items():
                if 'crs_wkt' in coord.attrs:
                    crs = pyproj.CRS.from_wkt(coord.attrs['crs_wkt'])
                    crs_var_name = coord_name
                    break
            
            # If not found in coordinates, search in variables
            if crs is None:
                for var_name, var in src.data_vars.items():
                    if 'crs_wkt' in var.attrs:
                        crs = pyproj.CRS.from_wkt(var.attrs['crs_wkt'])
                        crs_var_name = var_name
                        src.coords[var_name] = var
                        break
            
            # If still not found, look in global attributes
            if crs is None:
                if 'crs_wkt' in src.attrs:
                    crs = pyproj.CRS.from_wkt(src.attrs['crs_wkt'])
                elif 'proj4_string' in src.attrs:
                    crs = pyproj.CRS.from_proj4(src.attrs['proj4_string'])
            
            # If no variable name was found for the projection, use the default
            if crs_var_name is None:
                crs_var_name = grid_mapping
                src.coords[grid_mapping] = xr.DataArray(0, attrs=crs.to_cf())
            
            # Ensure all variables have the grid_mapping attribute
            for var in src.data_vars:
                src[var].attrs['grid_mapping'] = crs_var_name

            src.attrs['grid_mapping'] = crs_var_name
            
            return Image(data=src, crs=crs)


class GeoTIFFReader(ImageReader):
    """
    Reader for GeoTIFF image files with geospatial metadata.
    
    This class reads GeoTIFF (.tif, .tiff, .geotiff) files and extracts image data,
    coordinate systems, and spatial metadata to create an Image object. It handles
    conversion from GDAL/rasterio representation to xarray format.
    
    Notes
    -----
    The class preserves band-specific metadata, nodata values, and coordinates while
    transforming raster data into xarray's data model with explicit dimensions and
    coordinates.
    """

    def read(self, filename: str) -> Image:
        """
        Read a GeoTIFF file and convert it to an Image object.
        
        Parameters
        ----------
        filename : str
            Path to the GeoTIFF file
            
        Returns
        -------
        Image
            An Image object containing the data, coordinates, and CRS information
            
        Notes
        -----
        This method extracts GeoTIFF metadata including:
        - Spatial reference information (CRS)
        - Band data and descriptions
        - Nodata values
        - TIFF tags and band-specific metadata
        """
        grid_mapping = 'projection'
        
        with rasterio.open(filename) as src:
            crs = pyproj.CRS.from_proj4(src.crs.to_proj4())
            coords = self._prepare_coords(src, crs, grid_mapping)
            variables = self._prepare_vars(src, coords, grid_mapping)
            
            # Create global dataset attributes
            attrs = {}
            
            # Add nodata values
            for i in range(1, src.count + 1):
                nodata = src.nodatavals[i-1]
                if nodata is not None:
                    attrs[f'_FillValue_band_{i}'] = nodata
            
            # Add other relevant metadata from GeoTIFF file
            for key, value in src.tags().items():
                # Filter some tags that might cause problems or aren't relevant
                if key not in ['TIFFTAG_DATETIME', 'TIFFTAG_SOFTWARE']:
                    attrs[f'tiff_{key}'] = value
            
            # Add band metadata summary
            for i in range(1, src.count + 1):
                band_tags = src.tags(i)
                for tag_key, tag_value in band_tags.items():
                    attrs[f'band_{i}_{tag_key}'] = tag_value
            
            attrs['grid_mapping'] = grid_mapping
            
            # Create dataset with all attributes
            dataset = xr.Dataset(data_vars=variables, coords=coords, attrs=attrs)
            
            return Image(data=dataset, crs=crs)
    
    def _prepare_coords(self, src: rasterio.DatasetReader, crs: pyproj.CRS, grid_mapping: str) -> Dict[str, xr.DataArray]:
        """
        Generate coordinates for the dataset based on the CRS and source data.
        
        Parameters
        ----------
        src : rasterio.DatasetReader
            Rasterio object representing the source data
        crs : pyproj.CRS
            CRS object representing the coordinate reference system
        grid_mapping : str
            Name of the projection variable
            
        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary of coordinate arrays including x, y and the grid mapping
            
        Notes
        -----
        This method creates coordinate arrays with proper CF Convention attributes
        derived from the CRS.
        """
        x_meta, y_meta = crs.cs_to_cf()
        wkt_meta = crs.to_cf()
            
        x = np.array([src.xy(row, col)[0] for row, col in zip(np.zeros(src.width), np.arange(src.width))])
        y = np.array([src.xy(row, col)[-1] for row, col in zip(np.arange(src.height), np.zeros(src.height))])
        
        coords = {
            'x': xr.DataArray(
                data=x,
                coords={'x': x},
                attrs=x_meta
            ),
            'y': xr.DataArray(
                data=y,
                coords={'y': y},
                attrs=y_meta
            ),
            grid_mapping: xr.DataArray(
                data=0,
                attrs=wkt_meta
            )
        }
        
        return coords
    
    def _prepare_vars(self, src: rasterio.DatasetReader, coords: Dict[str, xr.DataArray], grid_mapping: str) -> Dict[str, xr.DataArray]:
        """
        Generate data variables (bands) for the dataset from the source data.
        
        Parameters
        ----------
        src : rasterio.DatasetReader
            Rasterio object representing the source data
        coords : Dict[str, xr.DataArray]
            Dictionary of coordinate arrays
        grid_mapping : str
            Name of the projection variable
            
        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary of data variables (bands) with metadata
            
        Notes
        -----
        This method processes each raster band, preserving band descriptions,
        nodata values, and any band-specific metadata from the GeoTIFF.
        """
        band_names = src.descriptions if not None in src.descriptions else [f'Band {i}' for i in range(1, src.count + 1)]
        
        variables = {}
        
        for idx, band_name in enumerate(band_names, start=1):
            band_data = src.read(idx)
            nodata = src.nodatavals[idx-1]
            
            # Create band-specific attributes
            attrs = {
                'grid_mapping': grid_mapping,
                'long_name': band_name
            }
            
            # Add nodata value if it exists
            if nodata is not None:
                attrs['_FillValue'] = nodata
            
            # Add band-specific metadata
            for key, value in src.tags(idx).items():
                attrs[f'tiff_{key}'] = value
            
            variables[band_name] = xr.DataArray(
                data=band_data,
                dims=('y', 'x'),
                coords={'y': coords['y'], 'x': coords['x']},
                attrs=attrs
            )
        
        return variables


def open(filename: str) -> Image:
    """
    Open an image file using the appropriate reader based on file extension.
    
    This factory function determines the correct reader to use based on the file
    extension and delegates to the appropriate reader class.
    
    Parameters
    ----------
    filename : str
        Path to the image file to be opened
        
    Returns
    -------
    Image
        An Image object containing the data, coordinates, and CRS information
        
    Raises
    ------
    ValueError
        If the file format is not supported based on its extension
        
    Examples
    --------
    >>> from sensingpy import reader
    >>> img = reader.open('example.tif')
    >>> img.plot()
    
    >>> # Open a NetCDF file
    >>> img = reader.open('example.nc')
    >>> print(img.band_names)
    """
    extension = filename.split('.')[-1].lower()
    
    if extension in enums.FILE_EXTENTIONS.TIF.value:
        return GeoTIFFReader().read(filename)
    elif extension in enums.FILE_EXTENTIONS.NETCDF.value:
        return NetCDFReader().read(filename)
    else:
        raise ValueError(f"Unsupported file format: {extension}")