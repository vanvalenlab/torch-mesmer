from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torch-mesmer'
copyright = f'{date.today().year}, Van Valen Lab'
author = 'Van Valen Lab'
release = '0.0.1-dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autodoc/autosummary conf
autosummary_generate = True

# Execution conf
nb_execution_timeout = 300  # seconds
nb_execution_show_tb = True  # print tracebacks to stderr
nb_scroll_outputs = True  # Make long output boxes scrollable by default


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "torch-mesmer"
