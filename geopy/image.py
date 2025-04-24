from __future__ import annotations

import rasterio.features
import xarray as xr
import numpy as np
import rasterio
import selector
import pyproj
import enums


from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry.base import BaseGeometry
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from typing import Tuple, List, Iterable, Self
from affine import Affine
from copy import deepcopy



class Image(object):
    grid_mapping : str = 'projection'

    def __init__(self, data: xr.Dataset, crs: pyproj.CRS) -> None:
        """Initialize an Image object with geospatial data and coordinate reference system.
        
        Args:
            data (xr.Dataset): The xarray Dataset containing the image data with dimensions
                and variables representing different bands/channels
            crs (pyproj.CRS): The coordinate reference system defining the spatial reference
                of the image data
        """

        self.crs: pyproj.CRS = crs
        self.data: xr.Dataset = data
        self.name: str = ''

    def replace(self, old : str, new : str) -> Self:
        """Replace occurrences of a substring in all band names with a new substring.

        Args:
            old (str): The substring to be replaced in band names
            new (str): The substring to replace with

        Returns:
            Self: Returns the Image object for method chaining
            
        Example:
            >>> image.replace('B01', 'blue')  # Renames band 'B01' to 'blue'
        """
        
        new_names = {
            var: var.replace(old, new) for var in self.data.data_vars if old in var
        }

        self.data = self.data.rename(new_names)
        return self

    def rename(self, new_names) -> Self:
        """Rename band names using a dictionary mapping.

        Args:
            new_names (dict): Dictionary mapping old band names to new band names

        Returns:
            Self: Returns the Image object for method chaining
        """
        
        self.data = self.data.rename(new_names)
        return self
    
    def rename_by_enum(self, enum : enums.Enum) -> Self:
        """Rename bands using an enumeration mapping.

        Renames image bands using a mapping defined in an enumeration class. The enumeration
        should contain band name mappings where each enum value is a list of possible wavelength
        strings and the enum name is the new band name.

        Args:
            enum (enums.Enum): Enumeration class containing band name mappings. Each enum value
                should be a List[str] of wavelength strings that map to the enum name.

        Returns:
            Self: Returns the Image object for method chaining
            
        Example:
            >>> # Using SENTINEL2_BANDS enum to rename bands
            >>> image.rename_by_enum(SENTINEL2_BANDS)
            >>> # Renames bands like '443' to 'B1', '493' to 'B2', etc.
            
        See Also:
            enums.SENTINEL2_BANDS: Enum for Sentinel-2 band mappings
            enums.MICASENSE_BANDS: Enum for MicaSense RedEdge band mappings
        """

        for band in enum:
            for wavelenght in band.value:
                self.replace(wavelenght, band.name)

        return self

    @property
    def band_names(self) -> List[str]:
        """Get list of band names in the image.

        Returns:
            List[str]: List of band names
        """

        return list(self.data.data_vars.keys())
    
    @property
    def width(self) -> int:
        """Get width of the image in pixels.

        Returns:
            int: Image width
        """

        return len(self.data.x)
    
    @property
    def height(self) -> int:
        """Get height of the image in pixels.

        Returns:
            int: Image height 
        """
        
        return len(self.data.y)
    
    @property
    def count(self) -> int:
        """Get number of bands in the image.

        Returns:
            int: Number of bands
        """
        
        return len(self.data.data_vars)

    @property
    def x_res(self) -> float | int:
        """Get pixel resolution in x direction.

        Returns:
            float|int: X resolution
        """

        return float(abs(self.data.x[0] - self.data.x[1]))

    @property
    def y_res(self) -> float | int:
        """Get pixel resolution in y direction.

        Returns:
            float|int: Y resolution
        """
        
        return float(abs(self.data.y[0] - self.data.y[1]))
    
    @property
    def transform(self) -> Affine:
        """Get affine transform for the image.

        Returns:
            Affine: Affine transform object
        """
         
        return from_origin(self.left, self.top, self.x_res, self.y_res)

    @property
    def xs_ys(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get meshgrid of x and y coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X and Y coordinate arrays
        """

        return np.meshgrid(self.data.x, self.data.y)

    @property
    def left(self) -> float:
        """Get min longitude coordinate of the image.

        Returns:
            float: Left coordinate
        """
        
        return float(self.data.x.min()) - abs(self.x_res / 2
)
    @property
    def right(self) -> float:
        """Get max longitude coordinate of the image.

        Returns:
            float: Right coordinate
        """

        return float(self.data.x.max()) + abs(self.x_res / 2
)
    @property
    def top(self) -> float:
        """Get max latitude coordinate of the image.

        Returns:
            float: Top coordinate
        """
        
        return float(self.data.y.max()) + abs(self.y_res / 2)

    @property
    def bottom(self) -> float:
        """Get min latitude coordinate of the image.

        Returns:
            float: Bottom coordinate
        """

        return float(self.data.y.min()) - abs(self.y_res / 2)

    @property
    def bbox(self) -> Polygon:
        """Get bounding box polygon of the image.

        Returns:
            Polygon: Shapely polygon representing image bounds
        """
        
        return box(self.left, self.bottom, self.right, self.top)

    @property
    def values(self) -> np.ndarray:
        """Get array of all band values.

        Returns:
            np.ndarray: Array containing band values
        """
        
        return np.array( [self.data[band].values.copy() for band in self.band_names] )


    def reproject(self, new_crs: pyproj.CRS, interpolation : Resampling = Resampling.nearest) -> Self:        
        """Reproject image to new coordinate reference system.

        Args:
            new_crs (pyproj.CRS): Target coordinate reference system
            interpolation (Resampling): Resampling method to use

        Returns:
            Self: Returns the Image object for method chaining
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
        """Align image to match reference image's CRS, resolution and extent.

        Args:
            reference (Image): Reference image to align to
            interpolation (Resampling): Resampling method to use

        Returns:
            Self: Returns the Image object for method chaining
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
    
    def resample(self, scale : int, downscale : bool = True, interpolation : Resampling = Resampling.nearest) -> Self:        
        """Resample image by scaling factor.

        Args:
            scale (int): Scale factor
            downscale (bool): If True, downscale by factor, if False upscale
            interpolation (Resampling): Resampling method to use

        Returns:
            Self: Returns the Image object for method chaining
        """
        
        if downscale:
            scale = 1 / scale

        dst_transform = self.transform * Affine.scale(1 / scale, 1 / scale)
        dst_width = int(len(self.data.x) * scale)
        dst_height = int(len(self.data.y) * scale)

        self.data = self.__update_data(interpolation, dst_transform, dst_width, dst_height, self.crs, self.crs)
        return self

    def __update_data(self, interpolation : Resampling, new_transform : Affine, dst_width : int, dst_height : int, src_crs : pyproj.CRS, dst_crs : pyproj.CRS) -> xr.Dataset:
        """Update image data using new spatial parameters and coordinate reference system.

        Internal method used by reproject() and resample() to update the image data with new
        spatial parameters. Performs resampling and coordinate transformation for all bands.

        Args:
            interpolation (Resampling): Resampling method to use when transforming data
            new_transform (Affine): New affine transform matrix
            dst_width (int): Width of destination image in pixels
            dst_height (int): Height of destination image in pixels
            src_crs (pyproj.CRS): Source coordinate reference system
            dst_crs (pyproj.CRS): Destination coordinate reference system
        
        Returns:
            Dataset: New dataset.

        Example:
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
        """Clip image to given geometries.

        Creates a mask from the input geometries and trims the image extent to the minimum 
        bounding box that contains all non-zero values. The new extent is calculated by:
        1. Finding the first and last rows that contain any values
        2. Finding the first and last columns that contain any values
        3. Keeping only the data within these bounds

        Args:
            geometries (List[BaseGeometry]): List of geometries to clip to. The image will be
                clipped to the combined extent of all geometries.

        Returns:
            Self: Returns the Image object for method chaining

        Example:
            >>> # If an image has this pattern (where 0=outside geometry, 1=inside):
            >>> # 0 0 0 0 0
            >>> # 0 1 1 0 0  <- First row with values
            >>> # 0 1 1 0 0
            >>> # 0 0 0 0 0  <- Last row with values
            >>> # The result will be trimmed to:
            >>> # 1 1 0  <- Columns 1-3 only
            >>> # 1 1 0
        """
        
        inshape = rasterio.features.geometry_mask(geometries = geometries, out_shape = (self.height, self.width), 
                                                  transform = self.transform, invert = True)
            
        rows, cols = self.__find_empty_borders(inshape)
        self.data = self.data.isel({'y' : rows, 'x' : cols})
        return self
    
    def mask(self, condition : np.ndarray, bands : str | List[str] = None) -> Self:     
        """Mask image bands using condition array.

        Args:
            condition (np.ndarray): Boolean mask array
            bands (str|List[str]): Band(s) to apply mask to

        Returns:
            Self: Returns the Image object for method chaining
        """
        
        if bands is not None:
            self.data[bands] = self.data[bands].where( xr.DataArray(data = condition, dims = ('y', 'x')) )
        else:
            self.data = self.data.where( xr.DataArray(data = condition, dims = ('y', 'x')) )
        return self
    
    def geometry_mask(self, geometries : List[BaseGeometry], mask_out : bool = True,  bands : str | List[str] = None) -> Self:
        """Mask image using geometries.

        Creates a binary mask from the input geometries and sets values to NaN either inside
        or outside the geometries depending on the mask_out parameter.

        Args:
            geometries (List[BaseGeometry]): List of geometries for masking. Values outside
                these geometries will be set to NaN if mask_out is True.
            mask_out (bool): If True mask outside geometries, if False mask inside. Defaults
                to True.
            bands (str|List[str]): Band(s) to apply mask to. If None, applies to all bands.

        Returns:
            Self: Returns the Image object for method chaining

        Example:
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
        """Remove rows and columns that contain all NaN values only when adjacent rows/columns also contain all NaN values.
    
        The method preserves rows/columns with all NaN values if they are between rows/columns containing valid values.
        For example, if row 1 has values, row 2 is all NaN, and row 3 has values, row 2 will be preserved.

        Returns:
            Self: Returns the Image object for method chaining

        Example:
            >>> # If an image has this pattern (where 0=value, N=NaN):
            >>> # 0 0 N N N
            >>> # 0 0 N N N 
            >>> # N N N N N  <- This row will be dropped
            >>> # N N N N N  <- This row will be dropped
            >>> # The rightmost 3 columns will also be dropped

        Example:
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
        """Find non-empty row and column ranges in a binary array.

        Finds the minimum spanning range of rows and columns that contain non-zero values.
        Used internally by clip() and dropna() methods to trim image extents.

        Args:
            array (np.ndarray): Binary array where True/non-zero values indicate data to keep

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing:
                - Row indices spanning first to last non-empty rows
                - Column indices spanning first to last non-empty columns

        Example:
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
        """Select specific bands from the image.

        Args:
            bands (str|List[str]): Band(s) to select
            only_values (bool): If True return array of values, if False return DataArray

        Returns:
            np.ndarray|xr.DataArray: Selected band data
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
        """Add a new band to the image or update an existing band.

        Adds a new band with the specified name and data to the image. If a band with the
        given name already exists, its data will be updated with the new values.

        Args:
            band_name (str): Name of the band to add or update
            data (np.ndarray|xr.DataArray): Band data to add. Must match the spatial 
                dimensions of existing bands

        Returns:
            Self: Returns the Image object for method chaining

        Example:
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
        """Remove specified bands from the image.

        Args:
            bands (str|List[str]): Band(s) to remove

        Returns:
            Self: Returns the Image object for method chaining
        """
        
        self.data = self.data.drop_vars(bands)
        return self


    def extract_values(self, xs : np.ndarray, ys : np.ndarray, bands : List[str] = None, is_1D : bool = False) -> np.ndarray:
        """Extract values at specified coordinates.

        Args:
            xs (np.ndarray): X coordinates
            ys (np.ndarray): Y coordinates
            bands (List[str]): Bands to extract from
            is_1D (bool): If True treat coordinates as 1D arrays

        Returns:
            np.ndarray: Extracted values
        """
        
        bands = self.band_names if bands is None else bands
        filtered = self.data.sel({'x' : xs, 'y' : ys}, method = 'nearest')
        values = np.array( [ filtered[band].values.copy() for band in bands ] )

        if xs.ndim == ys.ndim and is_1D:
            values = values[:, np.arange(len(xs)), np.arange(len(xs))]
            
        return values

    def interval_choice(self, band : str, size : int, intervals : Iterable, replace : bool = True) -> np.ndarray:
        """Choose random values from intervals in specified band.

        Args:
            band (str): Band to sample from
            size (int): Number of samples
            intervals (Iterable): Value intervals to sample from
            replace (bool): Sample with replacement if True

        Returns:
            np.ndarray: Selected values
        """
        
        if not isinstance(band, str):
            raise ValueError('band argument must a string')


        array = self.select(band).ravel()        
        return selector.interval_choice(array, size, intervals, replace)

    def arginterval_choice(self, band : str, size : int, intervals : Iterable, replace : bool = True) -> np.ndarray:
        """Choose random indices from intervals in specified band.

        Args:
            band (str): Band to sample from
            size (int): Number of samples
            intervals (Iterable): Value intervals to sample from
            replace (bool): Sample with replacement if True

        Returns:
            np.ndarray: Selected indices
        """
        
        if not isinstance(band, str):
            raise ValueError('band argument must a string')


        array = self.select(band).ravel()        
        return selector.arginterval_choice(array, size, intervals, replace)


    def empty_like(self) -> Image:
        """Create empty image with same metadata and coordinates.

        Returns:
            Image: New empty image
        """
        
        result = Image(deepcopy(self.data), deepcopy(self.crs))
        result.drop_bands(result.band_names)
        return result
    

    def to_netcdf(self, filename):
        """Save image to NetCDF file.

        Args:
            filename (str): Output filename
        """
        
        self.data.attrs['proj4_string'] = self.crs.to_proj4()
        self.data.attrs['crs_wkt'] = self.crs.to_wkt()
        
        return self.data.to_netcdf(filename)
    
    def to_tif(self, filename):
        """Save image to GeoTIFF file.

        Args:
            filename (str): Output filename
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