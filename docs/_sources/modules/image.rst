Image Module
===========

The Image module provides functionality for handling geospatial image data through the Image class.

.. currentmodule:: sensingpy.image

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   Image

Image
-----

.. autoclass:: Image
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: grid_mapping, data, crs, name
   
   .. rubric:: Spatial Properties
   
   .. autosummary::
      :nosignatures:
      
      ~Image.width
      ~Image.height
      ~Image.transform
      ~Image.x_res
      ~Image.y_res
      ~Image.left
      ~Image.right
      ~Image.top
      ~Image.bottom
      ~Image.bbox
   
   .. rubric:: Data Properties
   
   .. autosummary::
      :nosignatures:
      
      ~Image.band_names
      ~Image.count
      ~Image.values
      ~Image.xs_ys
   
   .. rubric:: Data Manipulation Methods
   
   .. autosummary::
      :nosignatures:
      
      ~Image.add_band
      ~Image.drop_bands
      ~Image.select
      ~Image.rename
      ~Image.replace
      ~Image.rename_by_enum
   
   .. rubric:: Spatial Operations
   
   .. autosummary::
      :nosignatures:
      
      ~Image.reproject
      ~Image.align
      ~Image.resample
      ~Image.clip
      ~Image.mask
      ~Image.geometry_mask
      ~Image.dropna
   
   .. rubric:: Analysis Methods
   
   .. autosummary::
      :nosignatures:
      
      ~Image.normalized_diference
      ~Image.extract_values
      ~Image.interval_choice
      ~Image.arginterval_choice
   
   .. rubric:: Utility Methods
   
   .. autosummary::
      :nosignatures:
      
      ~Image.empty_like
      ~Image.copy
      ~Image.to_netcdf
      ~Image.to_tif