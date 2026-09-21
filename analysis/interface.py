"""
Interactive Tkinter/matplotlib tool for exploring CCD and Q-Q diagrams with a
choice of filters on every axis, and a Region toolbar to draw/select/delete
elliptical or box-shaped selection regions on either panel with the mouse.
Ellipses (not circles) because the two axes of a panel are frequently on very
different scales, so a fixed-radius circle would look distorted relative to
the data.

Left panel (CCD): a plain 2-filter color-color diagram, x = mag[X1]-mag[X2],
y = mag[Y1]-mag[Y2] - pick any 4 filters from the dropdowns.

Right panel (Q-Q): the reddening-free index used throughout the other
scripts in this folder, Q = (f1-f2) - k*(f2-f3) with
k = E(f1-f2)/E(f2-f3) (Fitzpatrick99, R_V=3.1) - pick a 3-filter triplet per
axis.

Data: modelsv5 SED grid (synthetic photometry via effective-wavelength
sampling) overlaid on the PHANGS Phot_v5p3_*_IR4_class12human.csv catalogs
(all galaxies combined). Same FILTERS convention as HaNIR.py - short band
name -> the HST/JWST filter code(s) it can be read from (some bands, like B
or Ha, come from different filter codes depending on the galaxy). Every
band's magnitudes/S-N are loaded once at startup, so switching filters in the
dropdowns is instant (no re-reading the catalogs).

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

import glob
import itertools
import os
import tkinter as tk
from tkinter import ttk

import extinction
import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

# Matplotlib's default keyboard shortcuts (arrow keys pan, Backspace goes
# back, etc.) collide with this app's own arrow-key/Delete bindings for
# moving and deleting regions, so turn them all off.
for _keymap in list(matplotlib.rcParams):
    if _keymap.startswith("keymap."):
        matplotlib.rcParams[_keymap] = []

MODELS_DIR = "/home/igerasimov/Nextcloud/science/phangs/DATA/modelsv5"
CATALOG_GLOB = (
    "/home/igerasimov/Nextcloud/science/phangs/DATA/PHANGSGALAXIES/"
    "hstjwstfilters/Phot_v5p3_*_IR4_class12human.csv"
)
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
SED_NORM_BAND = "I"  # F814W - normalize every SED to this band before averaging
SED_MAX_LINES = 200  # cap individual SEDs drawn per region (perf)

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


def synthetic_magnitudes(wavelength_nm, fnu_mjy):
    """Vega magnitudes in every FILTERS band from a model spectrum."""
    wavelength_aa = wavelength_nm * 10.0
    fnu_jy = np.interp(EFF_WAVE, wavelength_aa, fnu_mjy) / 1000.0
    zp = np.array([FILTERS[b]["zp_vega"] for b in BAND_NAMES])
    return -2.5 * np.log10(fnu_jy / zp)


def load_models(models_dir):
    mags = {b: [] for b in BAND_NAMES}
    ages = []
    for path in sorted(
        glob.glob(os.path.join(models_dir, "*_best_model.fits"))
    ):
        with fits.open(path) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            m = synthetic_magnitudes(data["wavelength"], data["Fnu"])
            for b, v in zip(BAND_NAMES, m):
                mags[b].append(v)
            ages.append(float(header["sfh.age"]))
    mags = {b: np.array(v) for b, v in mags.items()}
    return mags, np.array(ages)


def resolve_filter(df, options):
    """First of `options` (HST/JWST filter codes) present as a *_veg column in df."""
    for opt in options:
        if f"{opt}_veg" in df.columns:
            return opt
    return None


def load_catalog(catalog_glob):
    """Observed Vega magnitudes + S/N for class 1+2 clusters, all galaxies combined."""
    frames = []
    for path in sorted(glob.glob(catalog_glob)):
        df = pd.read_csv(path)
        frame = {}
        for band in BAND_NAMES:
            hst_filter = resolve_filter(df, FILTERS[band]["filter"])
            if hst_filter is None:
                frame[f"{band}_veg"] = np.full(len(df), np.nan)
                frame[f"{band}_sn"] = np.full(len(df), np.nan)
            else:
                frame[f"{band}_veg"] = df[f"{hst_filter}_veg"]
                frame[f"{band}_sn"] = df[
                    f"signal_to_noise_original_{hst_filter}"
                ]
        frames.append(pd.DataFrame(frame))

    cat = pd.concat(frames, ignore_index=True)
    mag = {b: cat[f"{b}_veg"].to_numpy() for b in BAND_NAMES}
    sn = {b: cat[f"{b}_sn"].to_numpy() for b in BAND_NAMES}
    return mag, sn


def good_mask(mag, sn, bands):
    mask = np.isfinite(mag[bands[0]])
    for b in bands:
        mask &= np.isfinite(mag[b]) & (sn[b] >= SN_MIN)
    return mask


def q_index(mag, bands, k):
    """Q = (f1-f2) - k*(f2-f3) for a (f1,f2,f3) triplet."""
    f1, f2, f3 = bands
    return (mag[f1] - mag[f2]) - k * (mag[f2] - mag[f3])


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
        self.x_model = np.array([])
        self.y_model = np.array([])
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.colorbar_obj = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("CCD / Q-Q explorer")

        self.models_mag, models_age = load_models(MODELS_DIR)
        self.log_age = np.log10(models_age)
        self.cat_mag, self.cat_sn = load_catalog(CATALOG_GLOB)
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
        """Per-cluster flux (relative to SED_NORM_BAND) and per-band validity, for the SED panel."""
        norm_mag = self.cat_mag[SED_NORM_BAND]
        norm_zp = FILTERS[SED_NORM_BAND]["zp_vega"]
        self.sed_norm_valid = np.isfinite(norm_mag) & (
            self.cat_sn[SED_NORM_BAND] >= SN_MIN
        )
        self.sed_band_valid = {
            b: np.isfinite(self.cat_mag[b]) & (self.cat_sn[b] >= SN_MIN)
            for b in BAND_NAMES
        }
        self.sed_flux_norm = {
            b: (FILTERS[b]["zp_vega"] / norm_zp)
            * 10 ** (-0.4 * (self.cat_mag[b] - norm_mag))
            for b in BAND_NAMES
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

        self.ccd_mirror_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Mirror Y axis",
            variable=self.ccd_mirror_y,
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
        ttk.Checkbutton(
            frame, text="Multi-select", variable=self.multi
        ).grid(row=next(row), column=0, columnspan=2, sticky="w")

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
        x1, x2 = self.ccd_x1.get(), self.ccd_x2.get()
        y1, y2 = self.ccd_y1.get(), self.ccd_y2.get()
        cmd = self.ccd_cmd.get()
        needed = [x1, x2, y1] if cmd else [x1, x2, y1, y2]
        panel.mask = good_mask(
            self.cat_mag, self.cat_sn, list(dict.fromkeys(needed))
        )
        panel.x_all = self.cat_mag[x1] - self.cat_mag[x2]
        panel.x_model = self.models_mag[x1] - self.models_mag[x2]
        panel.xlabel = f"{x1}-{x2}"
        if cmd:
            panel.y_all = self.cat_mag[y1]
            panel.y_model = self.models_mag[y1]
            panel.ylabel = y1
        else:
            panel.y_all = self.cat_mag[y1] - self.cat_mag[y2]
            panel.y_model = self.models_mag[y1] - self.models_mag[y2]
            panel.ylabel = f"{y1}-{y2}"
        panel.x = panel.x_all[panel.mask]
        panel.y = panel.y_all[panel.mask]
        panel.title = f"{'CMD' if cmd else 'CCD'} (N={panel.mask.sum()})"

    def compute_qq(self):
        panel = self.panels["qq"]
        xb = [self.qq_x1.get(), self.qq_x2.get(), self.qq_x3.get()]
        yb = [self.qq_y1.get(), self.qq_y2.get(), self.qq_y3.get()]
        cmd = self.qq_cmd.get()
        k_x = (EXTINCTION_A[xb[0]] - EXTINCTION_A[xb[1]]) / (
            EXTINCTION_A[xb[1]] - EXTINCTION_A[xb[2]]
        )
        panel.x_all = q_index(self.cat_mag, xb, k_x)
        panel.x_model = q_index(self.models_mag, xb, k_x)
        panel.xlabel = f"Q({xb[0]}-{xb[1]}-{xb[2]})"
        if cmd:
            y1 = yb[0]
            panel.y_all = self.cat_mag[y1]
            panel.y_model = self.models_mag[y1]
            panel.ylabel = y1
            needed = xb + [y1]
        else:
            k_y = (EXTINCTION_A[yb[0]] - EXTINCTION_A[yb[1]]) / (
                EXTINCTION_A[yb[1]] - EXTINCTION_A[yb[2]]
            )
            panel.y_all = q_index(self.cat_mag, yb, k_y)
            panel.y_model = q_index(self.models_mag, yb, k_y)
            panel.ylabel = f"Q({yb[0]}-{yb[1]}-{yb[2]})"
            needed = xb + yb
        panel.mask = good_mask(
            self.cat_mag, self.cat_sn, list(dict.fromkeys(needed))
        )
        panel.x = panel.x_all[panel.mask]
        panel.y = panel.y_all[panel.mask]
        panel.title = f"{'Mag-Q' if cmd else 'Q-Q'} (N={panel.mask.sum()})"

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
        ax.scatter(panel.x, panel.y, s=3, c="gray", alpha=0.3, rasterized=True)
        cmd = self.ccd_cmd.get() if key == "ccd" else self.qq_cmd.get()
        if cmd or key == "qq":
            # model magnitudes aren't scaled to a real cluster mass/distance,
            # so they aren't meaningfully comparable to observed magnitudes
            if panel.colorbar_obj is not None:
                panel.colorbar_obj.ax.set_visible(False)
        else:
            sc = ax.scatter(
                panel.x_model,
                panel.y_model,
                c=self.log_age,
                cmap="viridis",
                s=20,
                label="modelsv5",
            )
            if panel.colorbar_obj is None:
                panel.colorbar_obj = self.fig.colorbar(
                    sc, ax=ax, label="log10(age / Myr)"
                )
            else:
                panel.colorbar_obj.ax.set_visible(True)
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

        for key, panel, region in selected:
            mask = self.region_full_mask(panel, region) & self.sed_norm_valid
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
                valid = [b for b in BAND_NAMES if self.sed_band_valid[b][i]]
                if len(valid) >= 2:
                    ax.plot(
                        [FILTERS[b]["eff_wave"] for b in valid],
                        [self.sed_flux_norm[b][i] for b in valid],
                        color=region["color"],
                        alpha=0.15,
                        linewidth=0.8,
                    )

            avg_x, avg_y = [], []
            for b in BAND_NAMES:
                vals = self.sed_flux_norm[b][idx][self.sed_band_valid[b][idx]]
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
                    k: drag[k] for k in ("shape", "center", "rx", "ry", "color")
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
