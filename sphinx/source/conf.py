import os
import sys
sys.path.insert(0, os.path.abspath(r'C:\Users\sergi\Documents\repos\geopy'))

project = 'geopy'
copyright = '2025, Sergio Heredia'
author = 'Sergio Heredia'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = ['.nojekyll']
html_baseurl = 'https://aouei.github.io/geopy/'