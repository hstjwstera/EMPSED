"""
Interactive Tkinter/matplotlib tool for exploring CCD and Q-Q diagrams with a
choice of filters on every axis, and a Region toolbar to draw/select/delete
elliptical or box-shaped selection regions on either panel with the mouse.
Ellipses (not circles) because the two axes of a panel are frequently on very
different scales, so a fixed-radius circle would look distorted relative to
the data.

Left panel (CCD): a plain 2-filter color-color diagram, x = mag[X1]-mag[X2],
y = mag[Y1]-mag[Y2] - pick any 4 filters from the dropdowns. An optional
Rayleigh-Jeans limit marker shows the color(s) a pure RJ spectrum
(F_nu ~ nu^2, temperature-independent) would have for the filters
currently on each axis - a crosshair at (x, y) in color-color mode, or a
vertical line at x in CMD mode (Y is a magnitude, not a color, so has no
RJ value of its own).

Right panel (Q-Q): the reddening-free index used throughout the other
scripts in this folder, Q = (f1-f2) - k*(f2-f3) with
k = E(f1-f2)/E(f2-f3) (Fitzpatrick99, R_V=3.1) - pick a 3-filter triplet per
axis.

Data: pcigale model SED grids (synthetic photometry via effective-wavelength
sampling), one per "*out" folder found in the project root (e.g. out/ for
cb19, 20260922_142526_out/ for bc03), each labelled from its own
pcigale.ini's sed_modules line - overlaid on the PHANGS
Phot_v5p3_*_IR4_class12human.csv catalogs (all galaxies combined). Same
FILTERS convention as HaNIR.py - short band name -> the HST/JWST filter
code(s) it can be read from (some bands, like B or Ha, come from different
filter codes depending on the galaxy). Every band's magnitudes/S-N are
loaded once at startup, so switching filters in the dropdowns is instant (no
re-reading the catalogs).

Region toolbar (applies to whichever panel you interact with next):
  - Ellipse / Box radio buttons: pick the shape "Add new" draws. An ellipse
    is dragged from its center out to its edge; a box from one corner to
    the opposite corner.
  - "Add new": arms drawing mode - drag on either panel to draw a new
    region (color auto-assigned); it becomes the selection immediately;
    stays armed for drawing more until you press "Select".
  - "Select": arms selection mode - click inside a region to select it
    (highlighted with a thicker edge).
  - "Delete": removes the currently selected region(s).
  - "Delete all": removes every region on both panels.
  - "Unfocus": clears the current selection without deleting anything.
  - "Multi-select" checkbox: when off, selecting (or drawing) a region
    replaces the selection; when on, drawing adds to it and clicking a
    region toggles it, across both panels.
Delete/Backspace on the keyboard also removes the selected regions; arrow
keys and the Grow/Shrink buttons act on all selected regions.

Bottom panel (SED): the individual SEDs (magnitude converted to flux,
normalized to F814W) of every cluster inside the selected region, plus its
median as a heavier line. With several regions selected only each region's
median is drawn (in that region's color), so the curves stay readable.
Updates whenever the selection, a region, or the filter dropdowns change.
"""

import csv
import glob
import itertools
import os
import tkinter as tk
from tkinter import filedialog, ttk

import extinction
import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

try:
    import settings
except ImportError:
    from . import settings

# Matplotlib's default keyboard shortcuts (arrow keys pan, Backspace goes
# back, etc.) collide with this app's own arrow-key/Delete bindings for
# moving and deleting regions, so turn them all off.
for _keymap in list(matplotlib.rcParams):
    if _keymap.startswith("keymap."):
        matplotlib.rcParams[_keymap] = []

MODEL_CMAP = "viridis"
MODEL_LINESTYLES = ["-", ":", "-.", ":"]
MODEL_LINEWIDTHS = {"-": 3.0, "--": 2.0, "-.": 2.0, ":": 2.0}
SN_MIN = 3.0
R_V = 3.1

REGION_COLORS = [
    "tab:red",
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "gold",
]
DRAG_THRESHOLD_PX = (
    4  # minimum press-release distance to count as a drawn region
)
RESIZE_FACTOR = 1.2  # per-click growth/shrink factor for the selected region
MOVE_STEP_FRAC = (
    0.02  # arrow-key nudge, as a fraction of the panel's current axis range
)
SED_NORM_BAND = (
    "I"  # F814W - normalize every SED to this band before averaging
)
SED_MAX_LINES = 200  # cap individual SEDs drawn per region (perf)

# KDE background mode: the background sample rendered as a smoothed 2D
# density (histogram + Gaussian smoothing, filled contours) instead of a
# gray scatter - useful when the scatter is too dense to read.
KDE_GRID_SIZE = 150  # histogram bins per axis before smoothing
KDE_SMOOTH_SIGMA = 2.5  # Gaussian smoothing, in bins
KDE_LEVELS = 12  # number of filled contour levels
KDE_MIN_LEVEL_FRAC = 0.03  # fraction of peak density below which nothing
# is drawn, so near-empty regions stay blank instead of flatly shaded
KDE_CMAP = "Greys"
KDE_MIN_POINTS = 10  # below this many points, fall back to the scatter

# How to refer to each filter: short band name -> the HST/JWST filter code(s)
# it can be read from (in priority order - some bands are reported under
# different filter codes depending on the galaxy), plus the effective (mean)
# wavelength [Angstrom] and Vega zeropoint [Jy] for model synthetic
# photometry (SVO Filter Profile Service, phys_params.py). Same as HaNIR.py.
FILTERS = {
    "NUV": {"filter": ["F275W"], "eff_wave": 2718.36, "zp_vega": 924.40},
    "U": {"filter": ["F336W"], "eff_wave": 3365.86, "zp_vega": 1241.97},
    "B": {
        "filter": ["F438W", "F435W"],
        "eff_wave": 4360.06,
        "zp_vega": 4036.38,
    },
    "V": {"filter": ["F555W"], "eff_wave": 5397.60, "zp_vega": 3662.24},
    "Ha": {
        "filter": ["F657N", "F658N"],
        "eff_wave": 6566.93,
        "zp_vega": 2688.83,
    },
    "I": {"filter": ["F814W"], "eff_wave": 8129.21, "zp_vega": 2440.74},
    "2.0um": {"filter": ["F200W"], "eff_wave": 20028.15, "zp_vega": 757.65},
    "3.0um": {"filter": ["F300M"], "eff_wave": 29940.44, "zp_vega": 369.65},
    "3.35um": {"filter": ["F335M"], "eff_wave": 33675.24, "zp_vega": 298.91},
    "3.6um": {"filter": ["F360M"], "eff_wave": 36298.10, "zp_vega": 260.31},
    "7.7um": {"filter": ["F770W"], "eff_wave": 77111.39, "zp_vega": 65.08},
    "10um": {"filter": ["F1000W"], "eff_wave": 99981.09, "zp_vega": 38.51},
    "11.3um": {"filter": ["F1130W"], "eff_wave": 113159.44, "zp_vega": 29.66},
    "21um": {"filter": ["F2100W"], "eff_wave": 209373.20, "zp_vega": 9.06},
}
BAND_NAMES = list(FILTERS.keys())
EFF_WAVE = np.array([FILTERS[b]["eff_wave"] for b in BAND_NAMES])
EXTINCTION_A = dict(
    zip(BAND_NAMES, extinction.fitzpatrick99(EFF_WAVE, 1.0, R_V))
)
AB_ZP_JY = 3631.0  # AB magnitude zeropoint [Jy], same for every band
MAG_SYSTEMS = ["ab", "veg"]  # magnitude system toggle: AB or Vega


def synthetic_magnitudes(wavelength_nm, fnu_mjy, system):
    """Magnitudes (`system`: "ab" or "veg") in every FILTERS band from a model spectrum."""
    wavelength_aa = wavelength_nm * 10.0
    fnu_jy = np.interp(EFF_WAVE, wavelength_aa, fnu_mjy) / 1000.0
    if system == "veg":
        zp = np.array([FILTERS[b]["zp_vega"] for b in BAND_NAMES])
    else:
        zp = AB_ZP_JY
    return -2.5 * np.log10(fnu_jy / zp)


def load_models(models_dir):
    """{system: {band: array}} magnitudes plus ages, for both magnitude systems."""
    mags = {s: {b: [] for b in BAND_NAMES} for s in MAG_SYSTEMS}
    ages = []
    for path in sorted(
        glob.glob(os.path.join(models_dir, "*_best_model.fits"))
    ):
        with fits.open(path) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            for system in MAG_SYSTEMS:
                m = synthetic_magnitudes(
                    data["wavelength"], data["Fnu"], system
                )
                for b, v in zip(BAND_NAMES, m):
                    mags[system][b].append(v)
            ages.append(float(header["sfh.age"]))
    mags = {
        s: {b: np.array(v) for b, v in d.items()} for s, d in mags.items()
    }
    return mags, np.array(ages)


def model_label(model_dir):
    """SSP module name from `model_dir`'s own pcigale.ini sed_modules line, e.g. "cb19"."""
    ini_path = os.path.join(model_dir, "pcigale.ini")
    if os.path.exists(ini_path):
        with open(ini_path) as f:
            for line in f:
                if line.strip().startswith("sed_modules"):
                    mods = [
                        m.strip() for m in line.split("=", 1)[1].split(",")
                    ]
                    ssp = [
                        m
                        for m in mods
                        if not m.startswith("sfh") and m != "redshifting"
                    ]
                    if ssp:
                        return ssp[0]
    return os.path.basename(model_dir)


def discover_model_sets(project_dir):
    """{label: dir} for every "*out" folder in `project_dir` with best_model FITS files."""
    sets = {}
    for path in sorted(glob.glob(os.path.join(project_dir, "*out"))):
        if os.path.isdir(path) and glob.glob(
            os.path.join(path, "*_best_model.fits")
        ):
            sets[model_label(path)] = path
    return sets


SLUG_TRACK_LABEL = "slug"
SLUG_TRACK_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "modified_models",
    "slug_deterministic_track.npz",
)


def load_slug_track(path):
    """{system: {band: array}} magnitudes plus ages [Myr] for the slug3
    deterministic SSP track, from the cache written by
    modified_models/precompute_slug_track.py (slugpy itself isn't a
    dependency here - see that script's docstring for why)."""
    with np.load(path) as data:
        ages = data["ages"]
        mags = {
            system: {
                b: data[f"{system}_{b}"]
                for b in BAND_NAMES
                if f"{system}_{b}" in data
            }
            for system in MAG_SYSTEMS
        }
    return mags, ages


def resolve_filter(df, options):
    """First of `options` (HST/JWST filter codes) present as a *_ab/*_veg column in df."""
    for opt in options:
        if f"{opt}_ab" in df.columns:
            return opt
    return None


def galaxy_name_from_path(path):
    """Galaxy name from a Phot_v5p3_<galaxy>_IR4_class12human.csv catalog filename."""
    base = os.path.basename(path)
    prefix, suffix = "Phot_v5p3_", "_IR4_class12human.csv"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix) : -len(suffix)]
    return base


def load_catalog(catalog_glob):
    """Observed magnitudes (both AB and Vega) + S/N for class 1+2 clusters,
    all galaxies combined, plus the untouched original catalog rows (tagged
    with a "galaxy" column) in the same row order, for subcatalog export."""
    frames = []
    full_frames = []
    for path in sorted(glob.glob(catalog_glob)):
        df = pd.read_csv(path)
        frame = {}
        for band in BAND_NAMES:
            hst_filter = resolve_filter(df, FILTERS[band]["filter"])
            if hst_filter is None:
                for system in MAG_SYSTEMS:
                    frame[f"{band}_{system}"] = np.full(len(df), np.nan)
                frame[f"{band}_sn"] = np.full(len(df), np.nan)
            else:
                for system in MAG_SYSTEMS:
                    frame[f"{band}_{system}"] = df[f"{hst_filter}_{system}"]
                frame[f"{band}_sn"] = df[
                    f"signal_to_noise_original_{hst_filter}"
                ]
        frames.append(pd.DataFrame(frame))
        full = df.copy()
        full.insert(0, "galaxy", galaxy_name_from_path(path))
        full_frames.append(full)

    cat = pd.concat(frames, ignore_index=True)
    mag = {
        system: {b: cat[f"{b}_{system}"].to_numpy() for b in BAND_NAMES}
        for system in MAG_SYSTEMS
    }
    sn = {b: cat[f"{b}_sn"].to_numpy() for b in BAND_NAMES}
    full_cat = pd.concat(full_frames, ignore_index=True)
    return mag, sn, full_cat


def good_mask(mag, sn, bands):
    mask = np.isfinite(mag[bands[0]])
    for b in bands:
        mask &= np.isfinite(mag[b]) & (sn[b] >= SN_MIN)
    return mask


def q_index(mag, bands, k):
    """Q = (f1-f2) - k*(f2-f3) for a (f1,f2,f3) triplet."""
    f1, f2, f3 = bands
    return (mag[f1] - mag[f2]) - k * (mag[f2] - mag[f3])


def kde_density(x, y, grid_size=KDE_GRID_SIZE, smooth_sigma=KDE_SMOOTH_SIGMA):
    """(x_centers, y_centers, density) for a fast KDE approximation of
    (x, y): a 2D histogram, Gaussian-smoothed."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    hist, xedges, yedges = np.histogram2d(x, y, bins=grid_size)
    density = gaussian_filter(hist, sigma=smooth_sigma).T
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    return x_centers, y_centers, density


def _closest_local_max_index(profile, prev_index):
    """Index of the local maximum in `profile` nearest to `prev_index` -
    falls back to the profile's global argmax if it has no interior peak."""
    peaks, _ = find_peaks(profile)
    if peaks.size == 0:
        return int(np.argmax(profile))
    return int(peaks[np.argmin(np.abs(peaks - prev_index))])


def density_ridge(x, y, grid_size=KDE_GRID_SIZE, smooth_sigma=KDE_SMOOTH_SIGMA):
    """(x, y) of the density ridge: starting from the single densest grid
    row and moving outward, the x where each y-row's density peaks closest
    to the previous row's - so a color space with two branches (e.g. two
    cluster populations) doesn't make the ridge jump between them whenever
    their relative heights flip. Rows whose raw (unsmoothed) point count is
    below KDE_MIN_LEVEL_FRAC of the fullest row's are dropped from the
    result - smoothing can make a handful of stray points in an otherwise
    near-empty row look like a locally significant peak, which a threshold
    on the smoothed density itself would miss."""
    finite = np.isfinite(x) & np.isfinite(y)
    raw_hist, _, _ = np.histogram2d(x[finite], y[finite], bins=grid_size)
    row_count = raw_hist.sum(axis=0)
    if row_count.max() <= 0:
        return np.array([]), np.array([])
    valid = row_count >= KDE_MIN_LEVEL_FRAC * row_count.max()

    x_centers, y_centers, density = kde_density(x, y, grid_size, smooth_sigma)
    row_peak = density.max(axis=1)
    n_rows = density.shape[0]
    ridge_col = np.empty(n_rows, dtype=int)
    start = int(np.argmax(row_peak))
    ridge_col[start] = int(np.argmax(density[start, :]))
    for row in range(start - 1, -1, -1):
        ridge_col[row] = _closest_local_max_index(
            density[row, :], ridge_col[row + 1]
        )
    for row in range(start + 1, n_rows):
        ridge_col[row] = _closest_local_max_index(
            density[row, :], ridge_col[row - 1]
        )

    return x_centers[ridge_col[valid]], y_centers[valid]


class Panel:
    """One axes' worth of state: its data, regions, and drag state."""

    def __init__(self, ax):
        self.ax = ax
        self.regions = []  # list of dicts: {shape, center, rx, ry, color}
        self.color_cycle = itertools.cycle(REGION_COLORS)
        self.selected = []  # subset of self.regions (same dict objects)
        self.press_xy = None
        self.press_pixel = None
        self.drag_region = None
        self.x = np.array([])
        self.y = np.array([])
        self.model_x = {}
        self.model_y = {}
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.colorbar_obj = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("CCD / Q-Q explorer")

        self.model_dirs = discover_model_sets(settings.PROJECT_DIR)
        self.model_sets = {}
        for label, model_dir in self.model_dirs.items():
            mags, age = load_models(model_dir)
            self.model_sets[label] = {"mag": mags, "log_age": np.log10(age)}
        if os.path.exists(SLUG_TRACK_CACHE):
            mags, age = load_slug_track(SLUG_TRACK_CACHE)
            self.model_sets[SLUG_TRACK_LABEL] = {
                "mag": mags,
                "log_age": np.log10(age),
            }
        self.model_labels = sorted(self.model_sets)
        self.model_linestyles = dict(
            zip(self.model_labels, itertools.cycle(MODEL_LINESTYLES))
        )
        all_log_age = (
            np.concatenate([s["log_age"] for s in self.model_sets.values()])
            if self.model_sets
            else np.array([0.0, 1.0])
        )
        self.age_vmin, self.age_vmax = (
            float(all_log_age.min()),
            float(all_log_age.max()),
        )
        self.model_visible = {
            label: tk.BooleanVar(value=True) for label in self.model_labels
        }
        self.mag_system = tk.StringVar(value="ab")
        self.cat_mag, self.cat_sn, self.full_catalog = load_catalog(
            settings.CATALOG_GLOB
        )
        self._prepare_sed_arrays()

        self.fig = Figure(figsize=(12, 8.5))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 0.7])
        ax_ccd = self.fig.add_subplot(gs[0, 0])
        ax_qq = self.fig.add_subplot(gs[0, 1])
        self.sed_ax = self.fig.add_subplot(gs[1, :])
        self.panels = {"ccd": Panel(ax_ccd), "qq": Panel(ax_qq)}

        self._build_controls()
        self._build_canvas()
        self._connect_events()
        self.redraw_all()

    def _prepare_sed_arrays(self):
        """Per-cluster flux (relative to SED_NORM_BAND) and per-band validity, for the SED panel, per magnitude system."""
        self.sed_norm_valid = {}
        self.sed_band_valid = {}
        self.sed_flux_norm = {}
        for system in MAG_SYSTEMS:
            mag = self.cat_mag[system]
            norm_mag = mag[SED_NORM_BAND]
            self.sed_norm_valid[system] = np.isfinite(norm_mag) & (
                self.cat_sn[SED_NORM_BAND] >= SN_MIN
            )
            self.sed_band_valid[system] = {
                b: np.isfinite(mag[b]) & (self.cat_sn[b] >= SN_MIN)
                for b in BAND_NAMES
            }
            self.sed_flux_norm[system] = {
                b: 10 ** (-0.4 * (mag[b] - norm_mag)) for b in BAND_NAMES
            }

    # ---- UI construction --------------------------------------------------

    def _add_combo(self, frame, label, row, default):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=default)
        combo = ttk.Combobox(
            frame,
            textvariable=var,
            values=BAND_NAMES,
            state="readonly",
            width=8,
        )
        combo.grid(row=row, column=1, sticky="w", pady=1)
        combo.bind("<<ComboboxSelected>>", lambda e: self.redraw_all())
        return var, combo

    def _build_controls(self):
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(side=tk.LEFT, fill=tk.Y)
        row = itertools.count()

        ttk.Label(frame, text="CCD / CMD panel", font=("", 10, "bold")).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        self.ccd_x1, _ = self._add_combo(frame, "X = ", next(row), "V")
        self.ccd_x2, _ = self._add_combo(frame, "    - ", next(row), "I")
        self.ccd_y1, _ = self._add_combo(frame, "Y = ", next(row), "U")
        self.ccd_y2, self.ccd_y2_combo = self._add_combo(
            frame, "    - ", next(row), "B"
        )

        self.ccd_cmd = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="CMD mode (Y = magnitude only)",
            variable=self.ccd_cmd,
            command=self._on_ccd_cmd_toggle,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        self.ccd_mirror_y = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Mirror Y axis",
            variable=self.ccd_mirror_y,
            command=self.redraw_all,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        self.ccd_show_rj = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Rayleigh-Jeans limit",
            variable=self.ccd_show_rj,
            command=self.redraw_all,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(frame, text="Q-Q / Mag-Q panel", font=("", 10, "bold")).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        self.qq_x1, _ = self._add_combo(frame, "X f1 = ", next(row), "7.7um")
        self.qq_x2, _ = self._add_combo(frame, "X f2 = ", next(row), "10um")
        self.qq_x3, _ = self._add_combo(frame, "X f3 = ", next(row), "11.3um")
        self.qq_y1, _ = self._add_combo(frame, "Y f1 = ", next(row), "NUV")
        self.qq_y2, self.qq_y2_combo = self._add_combo(
            frame, "Y f2 = ", next(row), "U"
        )
        self.qq_y3, self.qq_y3_combo = self._add_combo(
            frame, "Y f3 = ", next(row), "B"
        )

        self.qq_cmd = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="CMD mode (Y = magnitude only)",
            variable=self.qq_cmd,
            command=self._on_qq_cmd_toggle,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        self.qq_mirror_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Mirror Y axis",
            variable=self.qq_mirror_y,
            command=self.redraw_all,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(frame, text="Background sample", font=("", 10, "bold")).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        self.kde_background = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="KDE density (instead of scatter)",
            variable=self.kde_background,
            command=self.redraw_all,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        self.show_ridge = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Density ridge line",
            variable=self.show_ridge,
            command=self.redraw_all,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        ttk.Button(
            frame,
            text="Export ridge line...",
            command=self.export_ridge,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(
            frame, text="Magnitude system", font=("", 10, "bold")
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4))
        mag_system_row = ttk.Frame(frame)
        mag_system_row.grid(row=next(row), column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            mag_system_row,
            text="AB",
            variable=self.mag_system,
            value="ab",
            command=self.redraw_all,
        ).grid(row=0, column=0, padx=1)
        ttk.Radiobutton(
            mag_system_row,
            text="Vega",
            variable=self.mag_system,
            value="veg",
            command=self.redraw_all,
        ).grid(row=0, column=1, padx=1)

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(
            frame, text="Models (CCD panel)", font=("", 10, "bold")
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4))
        for label in self.model_labels:
            ttk.Checkbutton(
                frame,
                text=f"{label} ({self.model_linestyles[label]})",
                variable=self.model_visible[label],
                command=self.redraw_all,
            ).grid(row=next(row), column=0, columnspan=2, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        ttk.Label(frame, text="Region", font=("", 10, "bold")).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        self.shape = tk.StringVar(value="box")
        shape_row = ttk.Frame(frame)
        shape_row.grid(row=next(row), column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            shape_row, text="Ellipse", variable=self.shape, value="ellipse"
        ).grid(row=0, column=0, padx=1)
        ttk.Radiobutton(
            shape_row, text="Box", variable=self.shape, value="box"
        ).grid(row=0, column=1, padx=1)

        self.multi = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Multi-select", variable=self.multi).grid(
            row=next(row), column=0, columnspan=2, sticky="w"
        )

        self.mode = tk.StringVar(value="select")
        button_row = ttk.Frame(frame)
        button_row.grid(row=next(row), column=0, columnspan=2, sticky="w")
        ttk.Button(
            button_row,
            text="Add new",
            width=9,
            command=lambda: self.mode.set("add"),
        ).grid(row=0, column=0, padx=1)
        ttk.Button(
            button_row,
            text="Select",
            width=9,
            command=lambda: self.mode.set("select"),
        ).grid(row=0, column=1, padx=1)
        ttk.Button(
            button_row,
            text="Delete",
            width=9,
            command=self.delete_selected,
        ).grid(row=1, column=0, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Unfocus",
            width=9,
            command=self.unfocus_selected,
        ).grid(row=1, column=1, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Grow X",
            width=9,
            command=lambda: self.resize_selected(fx=RESIZE_FACTOR),
        ).grid(row=2, column=0, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Shrink X",
            width=9,
            command=lambda: self.resize_selected(fx=1 / RESIZE_FACTOR),
        ).grid(row=2, column=1, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Grow Y",
            width=9,
            command=lambda: self.resize_selected(fy=RESIZE_FACTOR),
        ).grid(row=3, column=0, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Shrink Y",
            width=9,
            command=lambda: self.resize_selected(fy=1 / RESIZE_FACTOR),
        ).grid(row=3, column=1, padx=1, pady=1)
        ttk.Button(
            button_row,
            text="Delete all",
            width=9,
            command=self.delete_all,
        ).grid(row=4, column=0, padx=1, pady=1)

        ttk.Button(
            frame,
            text="Export subcatalog...",
            command=self.export_subcatalog,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.mode_label = tk.StringVar()
        ttk.Label(
            frame, textvariable=self.mode_label, foreground="gray30"
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.mode.trace_add("write", lambda *a: self._update_mode_label())
        self._update_mode_label()

        ttk.Label(
            frame,
            text="Arrow keys move the selected regions.",
            foreground="gray30",
            wraplength=180,
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Separator(frame, orient="horizontal").grid(
            row=next(row), column=0, columnspan=2, sticky="ew", pady=8
        )

        self.status = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.status, wraplength=220).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self._on_ccd_cmd_toggle(redraw=False)
        self._on_qq_cmd_toggle(redraw=False)

    def _on_ccd_cmd_toggle(self, redraw=True):
        self.ccd_y2_combo.configure(
            state="disabled" if self.ccd_cmd.get() else "readonly"
        )
        if redraw:
            self.redraw_all()

    def _on_qq_cmd_toggle(self, redraw=True):
        state = "disabled" if self.qq_cmd.get() else "readonly"
        self.qq_y2_combo.configure(state=state)
        self.qq_y3_combo.configure(state=state)
        if redraw:
            self.redraw_all()

    def _build_canvas(self):
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        self.toolbar.update()
        self.canvas.draw()

    def _connect_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.root.bind("<Delete>", self.on_delete_key)
        self.root.bind("<BackSpace>", self.on_delete_key)
        self.root.bind("<Left>", lambda e: self.move_selected(-1, 0))
        self.root.bind("<Right>", lambda e: self.move_selected(1, 0))
        self.root.bind("<Up>", lambda e: self.move_selected(0, 1))
        self.root.bind("<Down>", lambda e: self.move_selected(0, -1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.root.quit()
        self.root.destroy()

    # ---- data / redraw -----------------------------------------------------

    def compute_ccd(self):
        panel = self.panels["ccd"]
        cat_mag = self.cat_mag[self.mag_system.get()]
        model_mag = {
            label: s["mag"][self.mag_system.get()]
            for label, s in self.model_sets.items()
        }
        x1, x2 = self.ccd_x1.get(), self.ccd_x2.get()
        y1, y2 = self.ccd_y1.get(), self.ccd_y2.get()
        cmd = self.ccd_cmd.get()
        needed = [x1, x2, y1] if cmd else [x1, x2, y1, y2]
        panel.mask = good_mask(
            cat_mag, self.cat_sn, list(dict.fromkeys(needed))
        )
        panel.x_all = cat_mag[x1] - cat_mag[x2]
        panel.model_x = {
            label: m[x1] - m[x2] for label, m in model_mag.items()
        }
        panel.xlabel = f"{x1}-{x2}"
        if cmd:
            panel.y_all = cat_mag[y1]
            panel.model_y = {label: m[y1] for label, m in model_mag.items()}
            panel.ylabel = y1
        else:
            panel.y_all = cat_mag[y1] - cat_mag[y2]
            panel.model_y = {
                label: m[y1] - m[y2] for label, m in model_mag.items()
            }
            panel.ylabel = f"{y1}-{y2}"
        panel.x = panel.x_all[panel.mask]
        panel.y = panel.y_all[panel.mask]
        panel.title = f"{'CMD' if cmd else 'CCD'} (N={panel.mask.sum()})"

    def compute_qq(self):
        panel = self.panels["qq"]
        cat_mag = self.cat_mag[self.mag_system.get()]
        model_mag = {
            label: s["mag"][self.mag_system.get()]
            for label, s in self.model_sets.items()
        }
        xb = [self.qq_x1.get(), self.qq_x2.get(), self.qq_x3.get()]
        yb = [self.qq_y1.get(), self.qq_y2.get(), self.qq_y3.get()]
        cmd = self.qq_cmd.get()
        k_x = (EXTINCTION_A[xb[0]] - EXTINCTION_A[xb[1]]) / (
            EXTINCTION_A[xb[1]] - EXTINCTION_A[xb[2]]
        )
        panel.x_all = q_index(cat_mag, xb, k_x)
        panel.model_x = {
            label: q_index(m, xb, k_x) for label, m in model_mag.items()
        }
        panel.xlabel = f"Q({xb[0]}-{xb[1]}-{xb[2]})"
        if cmd:
            y1 = yb[0]
            panel.y_all = cat_mag[y1]
            panel.model_y = {label: m[y1] for label, m in model_mag.items()}
            panel.ylabel = y1
            needed = xb + [y1]
        else:
            k_y = (EXTINCTION_A[yb[0]] - EXTINCTION_A[yb[1]]) / (
                EXTINCTION_A[yb[1]] - EXTINCTION_A[yb[2]]
            )
            panel.y_all = q_index(cat_mag, yb, k_y)
            panel.model_y = {
                label: q_index(m, yb, k_y) for label, m in model_mag.items()
            }
            panel.ylabel = f"Q({yb[0]}-{yb[1]}-{yb[2]})"
            needed = xb + yb
        panel.mask = good_mask(
            cat_mag, self.cat_sn, list(dict.fromkeys(needed))
        )
        panel.x = panel.x_all[panel.mask]
        panel.y = panel.y_all[panel.mask]
        panel.title = f"{'Mag-Q' if cmd else 'Q-Q'} (N={panel.mask.sum()})"

    def rj_color(self, b1, b2):
        """mag[b1]-mag[b2] a pure Rayleigh-Jeans spectrum (F_nu ~ nu^2,
        i.e. temperature-independent) would have, in the current magnitude
        system."""
        lam1, lam2 = FILTERS[b1]["eff_wave"], FILTERS[b2]["eff_wave"]
        if self.mag_system.get() == "veg":
            zp1, zp2 = FILTERS[b1]["zp_vega"], FILTERS[b2]["zp_vega"]
        else:
            zp1 = zp2 = AB_ZP_JY
        return -2.5 * np.log10((lam1**-2 / zp1) / (lam2**-2 / zp2))

    def redraw_all(self):
        try:
            self.compute_ccd()
            self.compute_qq()
        except (KeyError, ZeroDivisionError) as exc:
            self.status.set(f"Pick 3 distinct filters per axis ({exc})")
            return
        self.status.set("")
        for key in ("ccd", "qq"):
            self.redraw_panel(key)
        self.redraw_sed()
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    def redraw_panel(self, key, preserve_limits=False):
        panel = self.panels[key]
        ax = panel.ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        if self.kde_background.get() and panel.x.size >= KDE_MIN_POINTS:
            x_grid, y_grid, density = kde_density(panel.x, panel.y)
            peak = density.max()
            if peak > 0:
                levels = np.linspace(
                    KDE_MIN_LEVEL_FRAC * peak, peak, KDE_LEVELS
                )
                ax.contourf(
                    x_grid, y_grid, density, levels=levels, cmap=KDE_CMAP
                )
            else:
                ax.scatter(
                    panel.x, panel.y, s=3, c="gray", alpha=0.3, rasterized=True
                )
        else:
            ax.scatter(
                panel.x, panel.y, s=3, c="gray", alpha=0.3, rasterized=True
            )
        if self.show_ridge.get() and panel.x.size >= KDE_MIN_POINTS:
            ridge_x, ridge_y = density_ridge(panel.x, panel.y)
            if ridge_x.size > 0:
                ax.plot(
                    ridge_x,
                    ridge_y,
                    color="red",
                    linewidth=2,
                    label="density ridge",
                )
        cmd = self.ccd_cmd.get() if key == "ccd" else self.qq_cmd.get()
        if cmd or key == "qq":
            # model magnitudes aren't scaled to a real cluster mass/distance,
            # so they aren't meaningfully comparable to observed magnitudes
            if panel.colorbar_obj is not None:
                panel.colorbar_obj.ax.set_visible(False)
        else:
            norm = Normalize(vmin=self.age_vmin, vmax=self.age_vmax)
            for label in self.model_labels:
                if not self.model_visible[label].get():
                    continue
                x = panel.model_x[label]
                y = panel.model_y[label]
                age = self.model_sets[label]["log_age"]
                order = np.argsort(age)
                x, y, age = x[order], y[order], age[order]
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(
                    segments,
                    cmap=MODEL_CMAP,
                    norm=norm,
                    linestyle=self.model_linestyles[label],
                    linewidth=MODEL_LINEWIDTHS[self.model_linestyles[label]],
                )
                lc.set_array(age[:-1])
                ax.add_collection(lc)
                # dummy artist so the legend shows a line rather than the
                # LineCollection's default color-patch handle
                ax.plot(
                    [],
                    [],
                    color="black",
                    linestyle=self.model_linestyles[label],
                    linewidth=MODEL_LINEWIDTHS[self.model_linestyles[label]],
                    label=label,
                )
                if panel.colorbar_obj is None:
                    panel.colorbar_obj = self.fig.colorbar(
                        lc, ax=ax, label="log10(age / Myr)"
                    )
                else:
                    panel.colorbar_obj.ax.set_visible(True)
        if key == "ccd" and self.ccd_show_rj.get():
            x_rj = self.rj_color(self.ccd_x1.get(), self.ccd_x2.get())
            if cmd:
                ax.axvline(
                    x_rj,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Rayleigh-Jeans limit",
                )
            else:
                y_rj = self.rj_color(self.ccd_y1.get(), self.ccd_y2.get())
                ax.axvline(
                    x_rj, color="black", linestyle="--", linewidth=0.8, alpha=0.6
                )
                ax.axhline(
                    y_rj, color="black", linestyle="--", linewidth=0.8, alpha=0.6
                )
                ax.plot(
                    x_rj,
                    y_rj,
                    marker="+",
                    color="black",
                    markersize=12,
                    markeredgewidth=2,
                    linestyle="none",
                    label="Rayleigh-Jeans limit",
                )
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        mirror_y = self.ccd_mirror_y if key == "ccd" else self.qq_mirror_y
        if mirror_y.get():
            ax.invert_yaxis()
        if preserve_limits:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        for e in panel.regions:
            ax.add_patch(
                self.make_patch(
                    e["shape"],
                    e["center"],
                    e["rx"],
                    e["ry"],
                    e["color"],
                    linewidth=2.5 if self.is_selected(panel, e) else 1.4,
                )
            )

        # clusters selected by the OTHER panel's regions, shown here too
        other_key = self.other_key(key)
        other_panel = self.panels[other_key]
        for e in other_panel.regions:
            cross_mask = self.region_full_mask(other_panel, e) & panel.mask
            ax.scatter(
                panel.x_all[cross_mask],
                panel.y_all[cross_mask],
                s=25,
                facecolors="none",
                edgecolors=e["color"],
                linewidths=1.2,
                label=f"{other_key}-selected (N={cross_mask.sum()})",
            )
        if other_panel.regions:
            ax.legend(loc="best", fontsize=8)

    @staticmethod
    def make_patch(shape, center, rx, ry, color, linewidth):
        """Unfilled matplotlib patch for a region of half-widths (rx, ry) about `center`."""
        x0, y0 = center
        if shape == "box":
            return Rectangle(
                (x0 - rx, y0 - ry),
                2 * rx,
                2 * ry,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        return Ellipse(
            center,
            2 * rx,
            2 * ry,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )

    def region_full_mask(self, panel, region):
        """Boolean mask, over the full catalog, of points inside `region` and valid for `panel`."""
        x0, y0 = region["center"]
        rx, ry = max(region["rx"], 1e-12), max(region["ry"], 1e-12)
        if region["shape"] == "box":
            inside = (np.abs(panel.x_all - x0) <= rx) & (
                np.abs(panel.y_all - y0) <= ry
            )
        else:
            inside = ((panel.x_all - x0) / rx) ** 2 + (
                (panel.y_all - y0) / ry
            ) ** 2 <= 1
        return inside & panel.mask

    def count_in_region(self, panel, region):
        return int(self.region_full_mask(panel, region).sum())

    def redraw_sed(self):
        ax = self.sed_ax
        ax.clear()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel(f"Flux / Flux({SED_NORM_BAND})")

        selected = self.selected_regions()
        if not selected:
            ax.set_title("SED (select a region)")
            return
        ax.set_title("SED of selected clusters")

        system = self.mag_system.get()
        norm_valid = self.sed_norm_valid[system]
        band_valid = self.sed_band_valid[system]
        flux_norm = self.sed_flux_norm[system]

        for key, panel, region in selected:
            mask = self.region_full_mask(panel, region) & norm_valid
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue

            # with several regions selected, the individual SEDs would bury
            # each other, so only the medians are drawn
            plot_idx = idx if len(selected) == 1 else []
            if len(plot_idx) > SED_MAX_LINES:
                rng = np.random.default_rng(0)
                plot_idx = rng.choice(idx, size=SED_MAX_LINES, replace=False)
            for i in plot_idx:
                valid = [b for b in BAND_NAMES if band_valid[b][i]]
                if len(valid) >= 2:
                    ax.plot(
                        [FILTERS[b]["eff_wave"] for b in valid],
                        [flux_norm[b][i] for b in valid],
                        color=region["color"],
                        alpha=0.15,
                        linewidth=0.8,
                    )

            avg_x, avg_y = [], []
            for b in BAND_NAMES:
                vals = flux_norm[b][idx][band_valid[b][idx]]
                if vals.size > 0:
                    avg_x.append(FILTERS[b]["eff_wave"])
                    avg_y.append(np.median(vals))
            ax.plot(
                avg_x,
                avg_y,
                color=region["color"],
                linewidth=2.5,
                marker="o",
                label=f"{key} median (N={idx.size})",
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.set_title("SED (no selected cluster has valid F814W)")

    def export_ridge(self):
        """Save the current density ridge line(s) (see density_ridge()) to a
        CSV chosen via a save dialog - one row per (panel, x, y) point,
        covering whichever panel(s) currently have enough points for a
        ridge, regardless of whether "Density ridge line" is checked."""
        rows = []
        for key, panel in self.panels.items():
            if panel.x.size < KDE_MIN_POINTS:
                continue
            ridge_x, ridge_y = density_ridge(panel.x, panel.y)
            for xv, yv in zip(ridge_x, ridge_y):
                rows.append((key, panel.xlabel, panel.ylabel, xv, yv))
        if not rows:
            self.status.set("No ridge line to export (not enough points)")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Save density ridge line",
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["panel", "xlabel", "ylabel", "x", "y"])
            writer.writerows(rows)
        self.status.set(f"Ridge line saved to {os.path.basename(path)}")

    def export_subcatalog(self):
        """Save the original catalog rows for every cluster inside the
        selected box region, to a CSV chosen via a save dialog. Requires
        exactly one selected region, and it must be a box (not an ellipse)."""
        selected = self.selected_regions()
        if len(selected) != 1 or selected[0][2]["shape"] != "box":
            self.status.set(
                "Select exactly one box region to export a subcatalog"
            )
            return
        _, panel, region = selected[0]
        mask = self.region_full_mask(panel, region)
        if mask.sum() == 0:
            self.status.set("No clusters in the selected box")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Save subcatalog",
        )
        if not path:
            return
        self.full_catalog.loc[mask].to_csv(path, index=False)
        self.status.set(
            f"Subcatalog saved to {os.path.basename(path)} (N={int(mask.sum())})"
        )

    # ---- mouse interaction --------------------------------------------------

    def panel_for_axes(self, axes):
        for key, panel in self.panels.items():
            if panel.ax is axes:
                return key, panel
        return None, None

    def on_press(self, event):
        if event.inaxes is None:
            return
        _, panel = self.panel_for_axes(event.inaxes)
        if panel is None or event.xdata is None:
            return
        panel.press_xy = (event.xdata, event.ydata)
        panel.press_pixel = (event.x, event.y)
        panel.drag_region = None  # created lazily once drag exceeds threshold

    def on_motion(self, event):
        if self.mode.get() != "add":
            return
        for panel in self.panels.values():
            if panel.press_xy is None or event.inaxes is not panel.ax:
                continue
            if event.xdata is None or event.ydata is None:
                continue
            dx_px = event.x - panel.press_pixel[0]
            dy_px = event.y - panel.press_pixel[1]
            if (dx_px**2 + dy_px**2) ** 0.5 < DRAG_THRESHOLD_PX:
                continue
            x0, y0 = panel.press_xy
            if panel.drag_region is None:
                shape = self.shape.get()
                color = next(panel.color_cycle)
            else:
                shape = panel.drag_region["shape"]
                color = panel.drag_region["color"]
                panel.drag_region["patch"].remove()
            if shape == "box":
                # press point is one corner, cursor is the opposite corner
                center = ((x0 + event.xdata) / 2, (y0 + event.ydata) / 2)
                rx = abs(event.xdata - x0) / 2
                ry = abs(event.ydata - y0) / 2
            else:
                # press point is the center, cursor is on the edge
                center = (x0, y0)
                rx = abs(event.xdata - x0)
                ry = abs(event.ydata - y0)
            patch = self.make_patch(
                shape, center, rx, ry, color, linewidth=1.4
            )
            panel.ax.add_patch(patch)
            panel.drag_region = {
                "patch": patch,
                "shape": shape,
                "color": color,
                "center": center,
                "rx": rx,
                "ry": ry,
            }
            self.canvas.draw_idle()

    def on_release(self, event):
        mode = self.mode.get()
        for key, panel in self.panels.items():
            if panel.press_xy is None:
                continue
            if mode == "add" and panel.drag_region is not None:
                drag = panel.drag_region
                drag["patch"].remove()
                new_region = {
                    k: drag[k]
                    for k in ("shape", "center", "rx", "ry", "color")
                }
                panel.regions.append(new_region)
                if not self.multi.get():
                    self.clear_selection()
                panel.selected.append(new_region)
                self.refresh_regions()
            elif (
                mode == "select"
                and event.inaxes is panel.ax
                and event.xdata is not None
            ):
                self.handle_click_select(panel, event.xdata, event.ydata)
            panel.press_xy = None
            panel.press_pixel = None
            panel.drag_region = None

    def handle_click_select(self, panel, x, y):
        hit = None
        for e in reversed(panel.regions):
            x0, y0 = e["center"]
            rx, ry = max(e["rx"], 1e-12), max(e["ry"], 1e-12)
            if e["shape"] == "box":
                inside = abs(x - x0) <= rx and abs(y - y0) <= ry
            else:
                inside = ((x - x0) / rx) ** 2 + ((y - y0) / ry) ** 2 <= 1
            if inside:
                hit = e
                break
        if self.multi.get():
            if hit is None:
                return
            if self.is_selected(panel, hit):
                panel.selected = [s for s in panel.selected if s is not hit]
            else:
                panel.selected.append(hit)
        else:
            self.clear_selection()
            if hit is not None:
                panel.selected.append(hit)
        self.refresh_regions()

    # ---- Region toolbar ---------------------------------------------------

    def _update_mode_label(self):
        text = {
            "add": "Mode: Add new\ndrag on either panel to draw",
            "select": "Mode: Select\nclick inside a region to select it",
        }.get(self.mode.get(), "")
        self.mode_label.set(text)

    def other_key(self, key):
        return "qq" if key == "ccd" else "ccd"

    @staticmethod
    def is_selected(panel, region):
        return any(region is s for s in panel.selected)

    def selected_regions(self):
        """[(key, panel, region), ...] for every selected region on either panel."""
        return [
            (key, panel, region)
            for key, panel in self.panels.items()
            for region in panel.selected
        ]

    def clear_selection(self):
        for panel in self.panels.values():
            panel.selected = []

    def refresh_regions(self):
        """Redraw both panels + SED after any change to regions or selection."""
        for key in self.panels:
            self.redraw_panel(key, preserve_limits=True)
        self.redraw_sed()
        self.canvas.draw_idle()
        selected = self.selected_regions()
        if len(selected) == 1:
            key, panel, region = selected[0]
            self.status.set(
                f"Selected region: N={self.count_in_region(panel, region)}"
            )
        elif selected:
            self.status.set(f"Selected {len(selected)} regions")
        else:
            self.status.set("")

    def delete_selected(self):
        selected = self.selected_regions()
        if not selected:
            self.status.set("No region selected")
            return
        for _, panel, region in selected:
            panel.regions = [r for r in panel.regions if r is not region]
        self.clear_selection()
        self.refresh_regions()

    def delete_all(self):
        for panel in self.panels.values():
            panel.regions = []
        self.clear_selection()
        self.refresh_regions()

    def unfocus_selected(self):
        self.clear_selection()
        self.refresh_regions()

    def move_selected(self, dx_sign, dy_sign):
        selected = self.selected_regions()
        if not selected:
            return
        for _, panel, region in selected:
            xlim, ylim = panel.ax.get_xlim(), panel.ax.get_ylim()
            step_x = MOVE_STEP_FRAC * (xlim[1] - xlim[0]) * dx_sign
            step_y = MOVE_STEP_FRAC * (ylim[1] - ylim[0]) * dy_sign
            x0, y0 = region["center"]
            region["center"] = (x0 + step_x, y0 + step_y)
        self.refresh_regions()

    def resize_selected(self, fx=1.0, fy=1.0):
        selected = self.selected_regions()
        if not selected:
            self.status.set("No region selected")
            return
        for _, _, region in selected:
            region["rx"] *= fx
            region["ry"] *= fy
        self.refresh_regions()

    def on_delete_key(self, event):
        self.delete_selected()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
