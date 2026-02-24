import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'IFE Surrogate'
author = 'Tobias Leitgeb, Julian Tischler'
release = '0.2.2'

# Extensions
extensions = [
    'sphinx.ext.autodoc',           # auto-document Python code
    'sphinx.ext.napoleon',          # support NumPy/Google style docstrings
    'sphinx_autodoc_typehints',     # type hints in docs
    'sphinx.ext.mathjax',           # LaTeX math rendering in HTML
    'myst_parser',                  # Markdown support (optional)
    'nbsphinx',                     # Notebook
]

# notebook
# nbsphinx_execute = 'never'

# Templates
templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = "description"
# HTML output
html_theme = 'furo'
html_static_path = ['_static']
# html_logo = "_static/ife_saurogate.svg"
html_logo = "_static/logo.svg"


## OPTIONS--------------------------------------------------
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        # 1. The main Sidebar Column (the canvas) -> Off-white
        "color-sidebar-background": "#f8f9fb",
        
        # 3. Hover -> Bright Red
        "color-sidebar-item-background--hover": "#f4225a",
        
        # 4. Active -> Dark Red
        "color-sidebar-item-background--current": "#e4154b",

        # --- TEXT COLORS ---
        
        # Links are inside the Black/Red buttons, so text must be WHITE
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-link-text--top-level": "#ffffff",

        # The Site Name (Brand) sits on the Off-White background, NOT in a button.
        # So it must be DARK to be visible.
        "color-sidebar-brand-text": "#101010",
        "color-sidebar-caption-text": "#444444",
        
        # Search box colors (optional tweaks to fit the theme)
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-text": "#cccccc",
    },

    "dark_css_variables": {
        # Example dark mode equivalent (you can adjust as preferred)
        "color-sidebar-background": "#101010",

        "color-sidebar-item-background": "#222222",
        "color-sidebar-item-background--hover": "#610820",
        "color-sidebar-item-background--current": "#790624",

        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-link-text--top-level": "#ffffff",

        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-caption-text": "#ffffff",

        "color-sidebar-search-background": "#444444",
        "color-sidebar-search-text": "#cccccc",
    },
}



autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
html_css_files = [
    'hover.css',
]


def autodoc_experimental(app, what, name, obj, options, lines):
    if getattr(obj, "__experimental__", False):
        lines.insert(0, ".. note:: ⚠️ Experimental. API may change.")


def autodoc_experimental(app, what, name, obj, options, lines):
    if getattr(obj, "__deprecated__", False):
        lines.insert(0, ".. note:: ⚠️ Deprecated. This functionality is no longer supported and will be discarded in future updates.")

def setup(app):
    app.connect("autodoc-process-docstring", autodoc_experimental)
    app.add_css_file('hover.css')
