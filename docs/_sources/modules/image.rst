Image Module
==============

The Image module provides functionality for handling geospatial image data through the Image class.

.. currentmodule:: sensingpy.image

Class Overview
-------------

.. autoclass:: Image
   :no-members:

Properties
---------

.. rubric:: Spatial Properties

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.width
   Image.height
   Image.transform
   Image.x_res
   Image.y_res
   Image.left
   Image.right
   Image.top
   Image.bottom
   Image.bbox

.. rubric:: Data Properties

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.band_names
   Image.count
   Image.values
   Image.xs_ys

Core Methods
-----------

.. rubric:: Data Manipulation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.add_band
   Image.drop_bands
   Image.select
   Image.rename
   Image.replace
   Image.rename_by_enum

.. rubric:: Spatial Operations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.reproject
   Image.align
   Image.resample
   Image.clip
   Image.mask
   Image.geometry_mask
   Image.dropna

.. rubric:: Analysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.normalized_diference
   Image.extract_values
   Image.interval_choice
   Image.arginterval_choice

.. rubric:: Utility Methods

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Image.empty_like
   Image.copy
   Image.to_netcdf
   Image.to_tif

Full Class Documentation
-----------------------

.. autoclass:: Image
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: grid_mapping, data, crs, name