# explore

Interactive Tkinter/matplotlib tool (`interface.py`) for exploring CCD and
Q-Q diagrams of PHANGS star clusters against pcigale model tracks, with
region selection and per-region SED inspection. See the module docstring at
the top of `interface.py` for what each panel does.

## Setup

`interface.py` reads its data paths from `settings.py`, which is
machine-specific and not committed. Create it once per machine:

1. Copy `settings_example.py` to `settings.py`.
2. Fill in the two paths:
   - `PROJECT_DIR`: the directory containing the pcigale model-grid output
     folders - every subfolder matching `*out` (e.g. `out/`,
     `20260922_142526_out/`, one per model set such as cb19 or bc03) that
     holds `<id>_best_model.fits` files. Usually the project's root
     directory.
   - `CATALOG_GLOB`: a glob pattern matching the PHANGS HST+JWST cluster
     catalogs, one CSV per galaxy (`Phot_v5p3_*_IR4_class12human.csv`).

## Running

From this folder (or anywhere, since `interface.py` falls back to a relative
import of `settings`):

```
python interface.py
```
