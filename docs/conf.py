# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Import local version of kinetic.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information

project = "kinetic"
copyright = "2026, The Keras Team"
author = "The Keras Team"

release = ""
version = ""


# -- General configuration

extensions = [
  "myst_nb",
  "sphinx_click",
  "sphinx_copybutton",
  "sphinx_design",
  "sphinx.ext.intersphinx",
  "sphinx.ext.napoleon",
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.viewcode",
  "sphinx_llm.txt",
]

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

intersphinx_mapping = {
  "python": ("https://docs.python.org/3/", None),
  "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_title = "Kinetic"
html_favicon = "_static/logo.svg"
html_theme_options = {
  "analytics": {
    "google_analytics_id": "G-134NR8C6KG",
  },
  "show_toc_level": 2,
  "repository_url": "https://github.com/keras-team/kinetic",
  "use_repository_button": True,
  "use_download_button": False,
  "use_fullscreen_button": False,
  "navigation_with_keys": False,
  "show_navbar_depth": 1,
  "pygments_light_style": "github-light",
  "pygments_dark_style": "github-dark",
  "toc_title": "On this page",
  "logo": {
    "text": "⚡ Kinetic",
  },
}
# The search page renders its own search box in the body, so drop the
# sidebar's persistent one there — otherwise the page shows two.
# Mirrors the theme default minus "search-button-field.html".
html_sidebars = {
  "search": [
    "navbar-logo.html",
    "icon-links.html",
    "sbt-sidebar-nav.html",
  ],
}
html_static_path = ["_static"]
html_css_files = [
  "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
  "custom.css",
]

# -- Copy button: strip prompts and console output when copying
copybutton_exclude = ".linenos, .gp, .go"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for EPUB output
epub_show_urls = "footnote"


# -- Extension configuration

autodoc_member_order = "bysource"

autodoc_default_options = {
  "members": None,
  "undoc-members": True,
  "show-inheritance": True,
  "special-members": "__call__, __init__",
}

autosummary_generate = True
