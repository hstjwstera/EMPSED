"""Data paths shared by the tools in this folder.

Copy this file to settings.py (same folder) and fill in the two paths below
for your machine. settings.py is not committed, since these paths are
machine-specific.
"""

import os

# Directory containing the pcigale model-grid output folders: every
# subfolder matching "*out" (e.g. out/, 20260922_142526_out/ - one per model
# set, such as cb19 or bc03) that holds "<id>_best_model.fits" files.
# Usually the project's root directory.
PROJECT_DIR = "/path/to/EMPSED"

# Glob pattern matching the PHANGS HST+JWST cluster catalogs, one CSV per
# galaxy.
CATALOG_GLOB = os.path.join(
    "/path/to/PHANGSGALAXIES/hstjwstfilters",
    "Phot_v5p3_*_IR4_class12human.csv",
)
