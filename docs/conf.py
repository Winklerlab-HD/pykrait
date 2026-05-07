# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import pathlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # For earlier versions

pyproject_path = pathlib.Path(__file__).parents[1] / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]
copyright = f"{__import__('datetime').datetime.now().year}, Phaleanopsis"
version = pyproject_data["project"]["version"]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx_favicon",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser"
]

# -- AutoAPI configuration --------------------------------
root_doc = "index"
autoapi_type = 'python'
autoapi_dirs = ['../src/pykrait']
autoapi_python_use_implicit_namespaces = True
autoapi_add_toctree_entry = True
autoapi_keep_files = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autoapi_ignore = ["*/__main__*", "*/_*"]

favicons = [
    "favicon.ico",
    "android-chrome-192x192.png",
    "android-chrome-512x512.png",
    "apple-touch-icon.png",
    "favicon-16x16.png",
    "favicon-32x32.png",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "_static/images/pykrait_logo_light.png",
        "image_dark": "_static/images/pykrait_logo_dark.png",
    }
}

html_title = f"{project} {version} Documentation"