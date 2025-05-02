from __future__ import annotations

import rasterio.features
import xarray as xr
import numpy as np
import rasterio
import sensingpy.selector as selector
import pyproj
import sensingpy.enums as enums


from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry.base import BaseGeometry
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from typing import Tuple, List, Iterable, Self, Callable
from affine import Affine
from copy import deepcopy



class Image(object):
    """
    A geospatial image processing class for remote sensing operations.
    
    This class provides tools for working with geospatial image data in Python.
    It wraps xarray Datasets with geospatial metadata and provides methods for
    common remote sensing operations including reprojection, band manipulation,
    spatial analysis, and more.
    
    Parameters
    ----------
    data : xr.Dataset
        xarray Dataset containing image bands and coordinates
    crs : pyproj.CRS
        Coordinate reference system of the image
        
    Attributes
    ----------
    data : xr.Dataset
        xarray Dataset containing image bands and coordinates
    crs : pyproj.CRS
        Coordinate reference system of the image
    name : str
        Optional name identifier for the image
    grid_mapping : str
        Name of the grid mapping variable in the Dataset
    
    Notes
    -----
    The Image class is designed to maintain spatial reference information throughout
    operations. Most methods modify the image in-place and return self for method chaining.
    """

    grid_mapping : str = 'projection'

    def __init__(self, data: xr.Dataset, crs: pyproj.CRS) -> None:
        """
        Initialize an Image object with geospatial data and coordinate reference system.
        
        Parameters
        ----------
        data : xr.Dataset
            The xarray Dataset containing the image data with dimensions
            and variables representing different bands/channels
        crs : pyproj.CRS
            The coordinate reference system defining the spatial reference
            of the image data
        """

        self.crs: pyproj.CRS = crs
        self.data: xr.Dataset = data
        self.name: str = ''

    def replace(self, old : str, new : str) -> Self:
        """
        Replace occurrences of a substring in all band names with a new substring.

        Parameters
        ----------
        old : str
            The substring to be replaced in band names
        new : str
            The substring to replace with

        Returns
        -------
        Self
            Returns the Image object for method chaining
            
        Examples
        --------
        >>> image.replace('B01', 'blue')  # Renames band 'B01' to 'blue'
        """
        
        new_names = {
            var: var.replace(old, new) for var in self.data.data_vars if old in var
        }

        self.data = self.data.rename(new_names)
        return self

    def rename(self, new_names) -> Self:
        """
        Rename band names using a dictionary mapping.

        Parameters
        ----------
        new_names : dict
            Dictionary mapping old band names to new band names

        Returns
        -------
        Self
            Returns the Image object for method chaining
        """
        
        self.data = self.data.rename(new_names)
        return self
    
    def rename_by_enum(self, enum : enums.Enum) -> Self:
        """
        Rename bands using an enumeration mapping.

        Renames image bands using a mapping defined in an enumeration class.

        Parameters
        ----------
        enum : enums.Enum
            Enumeration class containing band name mappings. Each enum value
            should be a List[str] of wavelength strings that map to the enum name.

        Returns
        -------
        Self
            Returns the Image object for method chaining
            
        Examples
        --------
        >>> # Using SENTINEL2_BANDS enum to rename bands
        >>> image.rename_by_enum(SENTINEL2_BANDS)
        >>> # Renames bands like '443' to 'B1', '493' to 'B2', etc.
            
        See Also
        --------
        enums.SENTINEL2_BANDS : Enum for Sentinel-2 band mappings
        enums.MICASENSE_BANDS : Enum for MicaSense RedEdge band mappings
        """

        for band in enum:
            for wavelenght in band.value:
                self.replace(wavelenght, band.name)

        return self

    @property
    def band_names(self) -> List[str]:
        """
        Get list of band names in the image.

        Returns
        -------
        List[str]
            List of band names
        """

        return list(self.data.data_vars.keys())
    
    @property
    def width(self) -> int:
        """
        Get width of the image in pixels.

        Returns
        -------
        int
            Image width
        """

        return len(self.data.x)
    
    @property
    def height(self) -> int:
        """
        Get height of the image in pixels.

        Returns
        -------
        int
            Image height 
        """
        
        return len(self.data.y)
    
    @property
    def count(self) -> int:
        """
        Get number of bands in the image.

        Returns
        -------
        int
            Number of bands
        """
        
        return len(self.data.data_vars)

    @property
    def x_res(self) -> float | int:
        """
        Get pixel resolution in x direction.

        Returns
        -------
        float or int
            X resolution
        """

        return float(abs(self.data.x[0] - self.data.x[1]))

    @property
    def y_res(self) -> float | int:
        """
        Get pixel resolution in y direction.

        Returns
        -------
        float or int
            Y resolution
        """
        
        return float(abs(self.data.y[0] - self.data.y[1]))
    
    @property
    def transform(self) -> Affine:
        """
        Get affine transform for the image.
        
        Returns
        -------
        Affine
            Affine transform object representing the spatial relationship
            between pixel coordinates and CRS coordinates
        """
         
        return from_origin(self.left, self.top, self.x_res, self.y_res)

    @property
    def xs_ys(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get meshgrid of x and y coordinates.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X and Y coordinate arrays
        """

        return np.meshgrid(self.data.x, self.data.y)

    @property
    def left(self) -> float:
        """
        Get min longitude coordinate of the image.

        Returns
        -------
        float
            Left coordinate
        """
        
        return float(self.data.x.min()) - abs(self.x_res / 2)

    @property
    def right(self) -> float:
        """
        Get max longitude coordinate of the image.

        Returns
        -------
        float
            Right coordinate
        """

        return float(self.data.x.max()) + abs(self.x_res / 2)
    
    @property
    def top(self) -> float:
        """
        Get max latitude coordinate of the image.

        Returns
        -------
        float
            Top coordinate
        """
        
        return float(self.data.y.max()) + abs(self.y_res / 2)

    @property
    def bottom(self) -> float:
        """
        Get min latitude coordinate of the image.

        Returns
        -------
        float
            Bottom coordinate
        """

        return float(self.data.y.min()) - abs(self.y_res / 2)

    @property
    def bbox(self) -> Polygon:
        """
        Get bounding box polygon of the image.

        Returns
        -------
        Polygon
            Shapely polygon representing image bounds
        """
        
        return box(self.left, self.bottom, self.right, self.top)

    @property
    def values(self) -> np.ndarray:
        """
        Get array of all band values.

        Returns
        -------
        np.ndarray
            Array containing band values
        """
        
        return np.array( [self.data[band].values.copy() for band in self.band_names] )


    def reproject(self, new_crs: pyproj.CRS, interpolation: Resampling = Resampling.nearest) -> Self:        
        """
        Reproject image to new coordinate reference system.

        Parameters
        ----------
        new_crs : pyproj.CRS
            Target coordinate reference system
        interpolation : Resampling, optional
            Resampling method to use during reprojection, by default Resampling.nearest.
            Available options from rasterio.warp.Resampling include:
            - nearest: Nearest neighbor (default, preserves exact values)
            - bilinear: Bilinear interpolation (smooth, better for continuous data)
            - cubic: Cubic interpolation (smoother than bilinear)
            - cubic_spline: Cubic spline interpolation (smoothest)
            - lanczos: Lanczos windowed sinc interpolation (sharp edges)
            - average: Average of all contributing pixels
            - mode: Mode of all contributing pixels
            - max: Maximum value of all contributing pixels
            - min: Minimum value of all contributing pixels
            - med: Median of all contributing pixels
            - q1: First quartile of all contributing pixels
            - q3: Third quartile of all contributing pixels

        Returns
        -------
        Self
            Returns the Image object for method chaining
            
        Examples
        --------
        >>> # Reproject to UTM Zone 10N
        >>> utm_crs = pyproj.CRS.from_epsg(32610)
        >>> image.reproject(utm_crs, interpolation=Resampling.bilinear)
        >>> 
        >>> # Reproject to Web Mercator for web mapping
        >>> webmerc_crs = pyproj.CRS.from_epsg(3857)
        >>> image.reproject(webmerc_crs)
        """
        
        src_crs = self.crs
        dst_crs = new_crs
        
        src_height, src_width = self.height, self.width
        
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            left=float(self.data.x.min()), bottom=float(self.data.y.min()),
            right=float(self.data.x.max()), top=float(self.data.y.max())
        )
        
        self.data = self.__update_data(interpolation, dst_transform, dst_width, dst_height, self.crs, dst_crs)
        self.crs = dst_crs
        
        return self
    
    def align(self, reference: Image, interpolation: Resampling = Resampling.nearest) -> Self:
        """
        Align image to match reference image's CRS, resolution and extent.
        
        Transforms this image to match the coordinate reference system (CRS), spatial resolution,
        and geographic extent of a reference image.
        
        Parameters
        ----------
        reference : Image
            Reference image to align to. This image will be used as the
            template for CRS, resolution, and extent.
        interpolation : Resampling, optional
            Resampling method from rasterio.warp.Resampling to use
            during transformation, by default Resampling.nearest. Options include:
            - nearest: Nearest neighbor (preserves original values, best for categorical data)
            - bilinear: Bilinear interpolation (smooth, better for continuous data)
            - cubic: Cubic interpolation (smoother than bilinear)
            - lanczos: Lanczos windowed sinc interpolation (sharp edges)
                
        Returns
        -------
        Self
            Returns the modified Image object for method chaining
            
        Examples
        --------
        >>> # Align a Landsat image to match a Sentinel-2 reference image
        >>> landsat_img.align(sentinel2_img, interpolation=Resampling.bilinear)
        >>> 
        >>> # Check that the images now have the same dimensions
        >>> assert landsat_img.width == sentinel2_img.width
        >>> assert landsat_img.height == sentinel2_img.height
            
        Notes
        -----
        This operation modifies the original image in-place. Use the copy() method first
        if you want to preserve the original image.
        """
        
        if self.crs != reference.crs:
            self.reproject(reference.crs, interpolation)
        
        dst_transform = reference.transform
        dst_width, dst_height = reference.width, reference.height
        
        new_data_vars = {}
        
        for var_name, var_data in self.data.data_vars.items():
            dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
            dst_array[:] = np.nan
            
            dst_array, _ = reproject(
                source=var_data.values,
                destination=dst_array,
                src_transform=self.transform,
                src_crs=self.crs,
                dst_transform=dst_transform,
                dst_crs=reference.crs,
                dst_nodata=np.nan,
                resampling=interpolation
            )
            
            new_data_vars[var_name] = xr.DataArray(
                data=dst_array,
                dims=('y', 'x'),
                coords={'y': reference.data.y, 'x': reference.data.x},
                attrs={'grid_mapping': self.grid_mapping}
            )
        
        self.data = xr.Dataset(
            data_vars=new_data_vars,
            coords={
                'x': reference.data.x.copy(),
                'y': reference.data.y.copy(),
                self.grid_mapping: xr.DataArray(
                    data=0,
                    attrs=reference.crs.to_cf()
                )
            },
            attrs=self.data.attrs
        )
        
        self.crs = reference.crs

        return self
    
    def resample(self, scale: int, downscale: bool = True, interpolation: Resampling = Resampling.nearest) -> Self:        
        """
        Resample image by scaling factor to change spatial resolution.
        
        Changes the spatial resolution of the image by either increasing or decreasing the number
        of pixels while maintaining the same geographic extent.
        
        Parameters
        ----------
        scale : int
            Scale factor to apply. For example, a scale factor of 2 with downscale=True
            will reduce the image dimensions by half, while with downscale=False it will 
            double the image dimensions.
        downscale : bool, optional
            Direction of scaling operation, by default True:
            - True: Reduce resolution by dividing dimensions by scale factor
            - False: Increase resolution by multiplying dimensions by scale factor
        interpolation : Resampling, optional
            Resampling method from rasterio.warp.Resampling to use, by default Resampling.nearest:
            - nearest: Nearest neighbor (preserves exact values, best for categorical data)
            - bilinear: Bilinear interpolation (smooth, better for continuous data)
            - cubic: Cubic interpolation (smoother than bilinear)
            - lanczos: Lanczos windowed sinc interpolation (preserves sharp edges)
            - average: Averages all pixels that contribute to the output pixel
                
        Returns
        -------
        Self
            Returns the Image object for method chaining
            
        Examples
        --------
        >>> # Reduce image resolution by half (4x smaller area)
        >>> image.resample(scale=2, downscale=True)
        >>> print(f"New dimensions: {image.width}x{image.height}")
        >>> 
        >>> # Double the image resolution (4x larger area)
        >>> image.resample(scale=2, downscale=False, interpolation=Resampling.bilinear)
        >>> print(f"New dimensions: {image.width}x{image.height}")
            
        Notes
        -----
        When downscaling, higher values of scale mean coarser resolution.
        When upscaling, higher values of scale mean finer resolution.
        This operation modifies the image in-place.
        """
        
        if downscale:
            scale = 1 / scale

        dst_transform = self.transform * Affine.scale(1 / scale, 1 / scale)
        dst_width = int(len(self.data.x) * scale)
        dst_height = int(len(self.data.y) * scale)

        self.data = self.__update_data(interpolation, dst_transform, dst_width, dst_height, self.crs, self.crs)
        return self

    def __update_data(self, interpolation : Resampling, new_transform : Affine, dst_width : int, dst_height : int, src_crs : pyproj.CRS, dst_crs : pyproj.CRS) -> xr.Dataset:
        """
        Update image data using new spatial parameters and coordinate reference system.

        Parameters
        ----------
        interpolation : Resampling
            Resampling method to use when transforming data
        new_transform : Affine
            New affine transform matrix
        dst_width : int
            Width of destination image in pixels
        dst_height : int
            Height of destination image in pixels
        src_crs : pyproj.CRS
            Source coordinate reference system
        dst_crs : pyproj.CRS
            Destination coordinate reference system
        
        Returns
        -------
        xr.Dataset
            New dataset with updated spatial parameters

        Notes
        -----
        Internal method used by reproject() and resample() to update the image data with new
        spatial parameters. Performs resampling and coordinate transformation for all bands.

        Examples
        --------
        >>> # Used internally by reproject():
        >>> self.__update_data(
        ...     Resampling.nearest,
        ...     dst_transform,
        ...     dst_width,
        ...     dst_height, 
        ...     self.crs,
        ...     new_crs
        ... )
        """
            
        dst_x, _ = rasterio.transform.xy(new_transform, np.zeros(dst_width), np.arange(dst_width))
        _, dst_y = rasterio.transform.xy(new_transform, np.arange(dst_height), np.zeros(dst_height))
        x_meta, y_meta = self.crs.cs_to_cf()
        
        if x_meta['standard_name'] == 'latitude':
            x_meta, y_meta = y_meta, x_meta
        
        wkt_meta = dst_crs.to_cf()
        
        coords = {
            'x': xr.DataArray(
                data=dst_x,
                coords={'x': dst_x},
                attrs=x_meta
            ),
            'y': xr.DataArray(
                data=dst_y,
                coords={'y': dst_y},
                attrs=y_meta
            ),
            self.grid_mapping: xr.DataArray(
                data=0,
                attrs=wkt_meta
            )
        }

        new_data_vars = {}        
        for band in self.band_names:
            data = self.data[band].values
            dst_shape = (len(dst_y), len(dst_x))
            new_data = np.empty(dst_shape, dtype = data.dtype)

            new_data, _ = reproject(
                source = data,
                destination = new_data,
                src_transform = self.transform,
                src_crs = src_crs,
                dst_transform = new_transform,
                dst_crs = dst_crs,
                dst_nodata = 0 if data.dtype == np.uint8 else np.nan,
                resampling = interpolation
            )

            new_data_vars[band] = xr.DataArray(
                data=new_data,
                dims=('y', 'x'),
                coords={'y': coords['y'], 'x': coords['x']},
                attrs={'grid_mapping': self.grid_mapping}
            )
        
        return xr.Dataset(
            data_vars=new_data_vars,
            coords=coords,
            attrs = self.data.attrs
        )
    

    def clip(self, geometries : List[BaseGeometry]) -> Self:
        """
        Clip image to given geometries.

        Creates a mask from the input geometries and trims the image extent to the minimum 
        bounding box that contains all non-zero values.

        Parameters
        ----------
        geometries : List[BaseGeometry]
            List of geometries to clip to. The image will be
            clipped to the combined extent of all geometries.

        Returns
        -------
        Self
            Returns the Image object for method chaining

        Notes
        -----
        The new extent is calculated by:
        1. Finding the first and last rows that contain any values
        2. Finding the first and last columns that contain any values
        3. Keeping only the data within these bounds

        Examples
        --------
        >>> # If an image has this pattern (where 0=outside geometry, 1=inside):
        >>> # 0 0 0 0 0
        >>> # 0 1 1 0 0  <- First row with values
        >>> # 0 1 1 1 0
        >>> # 0 0 0 0 0  <- Last row with values
        >>> # The result will be trimmed to:
        >>> # 1 1 0  <- Columns 1-3 only
        >>> # 1 1 1
        """
        
        inshape = rasterio.features.geometry_mask(geometries = geometries, out_shape = (self.height, self.width), 
                                                  transform = self.transform, invert = True)
            
        rows, cols = self.__find_empty_borders(inshape)
        self.data = self.data.isel({'y' : rows, 'x' : cols})
        return self
    
    def mask(self, condition : np.ndarray, bands : str | List[str] = None) -> Self:     
        """
        Mask image bands using condition array.

        Parameters
        ----------
        condition : np.ndarray
            Boolean mask array
        bands : str or List[str], optional
            Band(s) to apply mask to, by default None which applies to all bands

        Returns
        -------
        Self
            Returns the Image object for method chaining
        """
        
        if bands is not None:
            self.data[bands] = self.data[bands].where( xr.DataArray(data = condition, dims = ('y', 'x')) )
        else:
            self.data = self.data.where( xr.DataArray(data = condition, dims = ('y', 'x')) )
        return self
    
    def geometry_mask(self, geometries : List[BaseGeometry], mask_out : bool = True,  bands : str | List[str] = None) -> Self:
        """
        Mask image using geometries.

        Creates a binary mask from the input geometries and sets values to NaN either inside
        or outside the geometries depending on the mask_out parameter.

        Parameters
        ----------
        geometries : List[BaseGeometry]
            List of geometries for masking
        mask_out : bool, optional
            If True mask outside geometries, if False mask inside, by default True
        bands : str or List[str], optional
            Band(s) to apply mask to, by default None which applies to all bands

        Returns
        -------
        Self
            Returns the Image object for method chaining

        Examples
        --------
        >>> # If mask_out=True with this pattern (where 1=inside geometry, 0=outside):
        >>> # 0 0 1 0 0
        >>> # 0 1 1 1 0
        >>> # 0 0 1 0 0
        >>> # The result will be:
        >>> # N N 5 N N  (where N=NaN, 5=original value)
        >>> # N 3 4 2 N
        >>> # N N 1 N N
        """

        condition = rasterio.features.geometry_mask(geometries = geometries, out_shape = (self.height, self.width), 
                                                    transform = self.transform, invert = mask_out)
            
        self.mask(condition, bands)
        return self

    def dropna(self) -> Self:
        """
        Remove rows and columns that contain all NaN values only when adjacent rows/columns also contain all NaN values.
    
        Returns
        -------
        Self
            Returns the Image object for method chaining

        Notes
        -----
        The method preserves rows/columns with all NaN values if they are between rows/columns containing valid values.
        For example, if row 1 has values, row 2 is all NaN, and row 3 has values, row 2 will be preserved.

        Examples
        --------
        >>> # If an image has this pattern (where 0=value, N=NaN):
        >>> # 0 0 N N N
        >>> # 0 0 N N N 
        >>> # N N N N N  <- This row will be dropped
        >>> # N N N N N  <- This row will be dropped
        >>> # The rightmost 3 columns will also be dropped

        >>> # If an image has this pattern (where 0=value, N=NaN):
        >>> # 0 0 N 0 N
        >>> # 0 0 N 0 N 
        >>> # 0 0 N 0 N
        >>> # 0 0 N 0 N
        >>> # Only the last column will be dropped
        """
        
        mask = np.zeros((self.height, self.width))
        for data in self.data.data_vars.values():
            mask = np.logical_or(mask, ~np.isnan(data.values))
            
        rows, cols = self.__find_empty_borders(mask)
        self.data = self.data.isel({'y' : rows, 'x' : cols})
        return self
    
    def __find_empty_borders(self, array : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find non-empty row and column ranges in a binary array.

        Parameters
        ----------
        array : np.ndarray
            Binary array where True/non-zero values indicate data to keep

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two arrays containing:
            - Row indices spanning first to last non-empty rows
            - Column indices spanning first to last non-empty columns

        Notes
        -----
        Finds the minimum spanning range of rows and columns that contain non-zero values.
        Used internally by clip() and dropna() methods to trim image extents.

        Examples
        --------
        >>> # For input array:
        >>> # 0 0 0 0 0
        >>> # 0 1 1 0 0
        >>> # 0 1 1 0 0
        >>> # 0 0 0 0 0
        >>> rows, cols = __find_empty_borders(array)
        >>> # rows = [1, 2]  # Indices of rows with data
        >>> # cols = [1, 2]  # Indices of columns with data
        """
        
        rows = np.where(array.any(axis=1))[0]
        rows = np.arange(rows.min(), rows.max() + 1)

        cols = np.where(array.any(axis=0))[0]
        cols = np.arange(cols.min(), cols.max() + 1)

        return rows, cols
    

    def select(self, bands : str | List[str], only_values : bool = True) -> np.ndarray | xr.DataArray:
        """
        Select specific bands from the image.

        Parameters
        ----------
        bands : str or List[str]
            Band(s) to select
        only_values : bool, optional
            If True return array of values, if False return DataArray, by default True

        Returns
        -------
        np.ndarray or xr.DataArray
            Selected band data
        """
        
        result = None

        if only_values:
            if isinstance(bands, list):
                result = np.array([self.data[band].values.copy() for band in bands])
            else:
                result = self.data[bands].values.copy()
        else:
            result = deepcopy(self.data[bands])
    
        return result
    
    def add_band(self, band_name : str, data : np.ndarray | xr.DataArray) -> Self:
        """
        Add a new band to the image or update an existing band.

        Parameters
        ----------
        band_name : str
            Name of the band to add or update
        data : np.ndarray or xr.DataArray
            Band data to add. Must match the spatial dimensions of existing bands

        Returns
        -------
        Self
            Returns the Image object for method chaining

        Examples
        --------
        >>> # Add new band
        >>> image.add_band('ndvi', ndvi_data)
        >>> # Update existing band
        >>> image.add_band('blue', new_blue_data)
        """
        
        if isinstance(data, np.ndarray):
            if not band_name in self.band_names:
                self.data[band_name] = (('y', 'x'), data)
            else:
                self.data[band_name].values = data
        else:
            self.data[band_name] = data
        return self
    
    def drop_bands(self, bands) -> Self:
        """
        Remove specified bands from the image.

        Parameters
        ----------
        bands : str or List[str]
            Band(s) to remove

        Returns
        -------
        Self
            Returns the Image object for method chaining
        """
        
        self.data = self.data.drop_vars(bands)
        return self


    def normalized_diference(self, band1: str, band2: str) -> np.ndarray:
        """
        Calculate normalized difference between two bands.
        
        Computes the normalized difference index between two bands using the formula:
        (band1 - band2) / (band1 + band2)
        
        Parameters
        ----------
        band1 : str
            Name of the first band in the calculation (numerator)
        band2 : str
            Name of the second band in the calculation (denominator)
        
        Returns
        -------
        np.ndarray
            2D array containing the normalized difference values ranging from -1 to 1.
            Areas where both bands have zero values will result in NaN values.
        
        Notes
        -----
        This is a common operation in remote sensing used to create various spectral indices
        such as NDVI (Normalized Difference Vegetation Index), NDWI (Normalized Difference
        Water Index), etc.
        
        Values outside the -1 to 1 range can occur if negative values are present
        in the input bands.
        
        Examples
        --------
        >>> # Calculate NDVI using NIR and Red bands
        >>> ndvi = image.normalized_diference('nir', 'red')
        >>> image.add_band('ndvi', ndvi)
        >>> 
        >>> # Calculate NDWI using Green and NIR bands
        >>> ndwi = image.normalized_diference('green', 'nir')
        >>> image.add_band('ndwi', ndwi)
        """
        
        b1 = self.data[band1].values.copy()
        b2 = self.data[band2].values.copy()
        
        return (b1 - b2) / (b1 + b2)


    def extract_values(self, xs: np.ndarray, ys: np.ndarray, bands: List[str] = None, is_1D: bool = False) -> np.ndarray:
        """
        Extract values at specified coordinates from the image.
        
        Parameters
        ----------
        xs : np.ndarray
            X coordinates (longitude/easting) in the image's CRS
        ys : np.ndarray
            Y coordinates (latitude/northing) in the image's CRS
        bands : List[str], optional
            List of band names to extract values from.
            If None, extracts from all bands, by default None
        is_1D : bool, optional
            If True, treats xs and ys as paired 1D arrays of points.
            If False, treats xs and ys as meshgrid arrays, by default False
        
        Returns
        -------
        np.ndarray
            Array of extracted values with shape:
            - If is_1D=True: (n_bands, n_points)
            - If is_1D=False: (n_bands, xs.shape[0], xs.shape[1])
        
        Notes
        -----
        This method uses xarray's 'nearest' method for coordinate selection, which
        finds the closest pixel to each requested coordinate.
        
        Examples
        --------
        >>> # Extract values at specific points
        >>> import numpy as np
        >>> from rasterio.transform import xy
        >>> 
        >>> # Create sample points (e.g., along a transect)
        >>> row_indices = np.array([100, 120, 140, 160, 180])
        >>> col_indices = np.array([200, 210, 220, 230, 240])
        >>> 
        >>> # Convert pixel coordinates to CRS coordinates
        >>> xs, ys = xy(image.transform, row_indices, col_indices)
        >>> xs = np.array(xs)
        >>> ys = np.array(ys)
        >>> 
        >>> # Extract values from 'red' and 'nir' bands
        >>> values = image.extract_values(xs, ys, bands=['red', 'nir'], is_1D=True)
        >>> print(f"Shape of extracted values: {values.shape}")  # (2, 5)
        >>> print(f"Red values: {values[0]}")
        >>> print(f"NIR values: {values[1]}")
        """
        
        bands = self.band_names if bands is None else bands
        filtered = self.data.sel({'x' : xs, 'y' : ys}, method = 'nearest')
        values = np.array( [ filtered[band].values.copy() for band in bands ] )

        if xs.ndim == ys.ndim and is_1D:
            values = values[:, np.arange(len(xs)), np.arange(len(xs))]
            
        return values

    def interval_choice(self, band: str, size: int, intervals: Iterable, replace: bool = True) -> np.ndarray:
        """
        Choose random values from intervals in specified band.

        Parameters
        ----------
        band : str
            Band to sample from
        size : int
            Number of samples
        intervals : Iterable
            Value intervals to sample from
        replace : bool, optional
            Sample with replacement if True, by default True

        Returns
        -------
        np.ndarray
            Selected values
        """
        
        if not isinstance(band, str):
            raise ValueError('band argument must a string')


        array = self.select(band).ravel()        
        return selector.interval_choice(array, size, intervals, replace)

    def arginterval_choice(self, band: str, size: int, intervals: Iterable, replace: bool = True) -> np.ndarray:
        """
        Choose random indices from intervals in specified band.

        Parameters
        ----------
        band : str
            Band to sample from
        size : int
            Number of samples
        intervals : Iterable
            Value intervals to sample from
        replace : bool, optional
            Sample with replacement if True, by default True

        Returns
        -------
        np.ndarray
            Selected indices
        """
        
        if not isinstance(band, str):
            raise ValueError('band argument must a string')


        array = self.select(band).ravel()        
        return selector.arginterval_choice(array, size, intervals, replace)


    def empty_like(self) -> Image:
        """
        Create empty image with same metadata and coordinates.

        Returns
        -------
        Image
            New empty image
        """
        
        result = Image(deepcopy(self.data), deepcopy(self.crs))
        result.drop_bands(result.band_names)
        return result
    
    def copy(self) -> Image:
        """
        Create a deep copy of the image.
        
        Returns
        -------
        Image
            Deep copy of the image
        """

        return deepcopy(self)
    

    def to_netcdf(self, filename):
        """
        Save image to NetCDF file.

        Parameters
        ----------
        filename : str
            Output filename
        """
        
        self.data.attrs['proj4_string'] = self.crs.to_proj4()
        self.data.attrs['crs_wkt'] = self.crs.to_wkt()
        
        return self.data.to_netcdf(filename)
    
    def to_tif(self, filename):
        """
        Save image to GeoTIFF file.

        Parameters
        ----------
        filename : str
            Output filename
        """
        
        height, width = self.height, self.width
        count = self.count
        
        # Prepare the metadata for rasterio
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': next(iter(self.data.data_vars.values())).dtype,
            'crs': self.crs,
            'transform': self.transform
        }
        
        with rasterio.open(filename, 'w', **meta) as dst:
            # Write each band
            for idx, (band_name, band_data) in enumerate(self.data.data_vars.items(), start=1):
                dst.write(band_data.values, idx)
                dst.set_band_description(idx, band_name)


    def __str__(self) -> str:
        return f'Bands: {self.band_names} | Height: {self.height} | Width: {self.width}'
    
    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        return self.data._repr_html_()
    

def compose(images: List[Image], method: Callable | np.ndarray, bands: List[str] = None) -> Image:
    """
    Compose multiple images into one using a composition method.
    
    This function combines multiple images by applying a composition method to 
    corresponding pixels across all input images. Common uses include creating 
    cloud-free composites, calculating statistics across time series, or 
    selecting optimal pixels based on quality metrics.
    
    Parameters
    ----------
    images : List[Image]
        List of Image objects to compose. All images should have compatible
        dimensions and coordinate reference systems.
    method : Callable or np.ndarray
        Method to use for composition. Can be either:
        - A callable function that takes an array of values across images and
          returns a single value (e.g., np.nanmean, np.nanmax)
        - An array of indices specifying which image to select for each pixel
    bands : List[str], optional
        List of band names to include in the composition, by default None which
        uses all bands from the first image
        
    Returns
    -------
    Image
        New Image object containing the composition result
    
    Notes
    -----
    The output image retains the spatial metadata (CRS, transform) from the first
    image in the list, but contains new pixel values based on the composition method.
    
    Examples
    --------
    >>> # Create a mean composite from multiple images
    >>> mean_composite = compose(image_list, np.nanmean)
    >>> 
    >>> # Create a maximum NDVI composite
    >>> import numpy as np
    >>> ndvi_values = np.array([img.normalized_diference('nir', 'red') for img in image_list])
    >>> best_indices = np.nanargmax(ndvi_values, axis=0)
    >>> max_ndvi_composite = compose(image_list, best_indices)
    """
    if bands is None:
        bands = images[0].band_names

    
    result = images[0].empty_like()
    for band in bands:
        result.add_band(band, selector.composite(np.array([image.data[band].values for image in images]), method))
    
    return result