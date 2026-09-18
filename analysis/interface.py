"""
Interactive Tkinter/matplotlib tool for exploring CCD and Q-Q diagrams with a
choice of filters on every axis, and an Ellipse toolbar to draw/select/delete
elliptical selection regions on either panel with the mouse. Ellipses (not
circles) because the two axes of a panel are frequently on very different
scales, so a fixed-radius circle would look distorted relative to the data.

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

Ellipse toolbar (applies to whichever panel you interact with next):
  - "Add new": arms drawing mode - drag on either panel to draw a new
    ellipse (color auto-assigned); stays armed for drawing more until you
    press "Select".
  - "Select": arms selection mode - click inside an ellipse to select it
    (highlighted with a thicker edge).
  - "Delete": removes the currently selected ellipse.
  - "Unfocus": clears the current selection without deleting anything.
Delete/Backspace on the keyboard also removes the selected ellipse.
"""

import glob
import itertools
import os
import tkinter as tk
from tkinter import ttk

import extinction
import numpy as np
import pandas as pd
from astropy.io import fits

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

MODELS_DIR = "/home/igerasimov/Nextcloud/science/phangs/DATA/modelsv5"
CATALOG_GLOB = (
    "/home/igerasimov/Nextcloud/science/phangs/DATA/PHANGSGALAXIES/"
    "hstjwstfilters/Phot_v5p3_*_IR4_class12human.csv"
)
SN_MIN = 3.0
R_V = 3.1

ELLIPSE_COLORS = [
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
    4  # minimum press-release distance to count as a drawn ellipse
)
RESIZE_FACTOR = 1.2  # per-click growth/shrink factor for the selected ellipse
MOVE_STEP_FRAC = (
    0.02  # arrow-key nudge, as a fraction of the panel's current axis range
)

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
    """One axes' worth of state: its data, ellipses, and drag state."""

    def __init__(self, ax):
        self.ax = ax
        self.ellipses = []  # list of dicts: {center, rx, ry, color}
        self.color_cycle = itertools.cycle(ELLIPSE_COLORS)
        self.selected = None
        self.press_xy = None
        self.press_pixel = None
        self.drag_ellipse = None
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

        self.fig = Figure(figsize=(12, 5.5))
        ax_ccd = self.fig.add_subplot(1, 2, 1)
        ax_qq = self.fig.add_subplot(1, 2, 2)
        self.panels = {"ccd": Panel(ax_ccd), "qq": Panel(ax_qq)}

        self._build_controls()
        self._build_canvas()
        self._connect_events()
        self.redraw_all()

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

        ttk.Label(frame, text="Ellipse", font=("", 10, "bold")).grid(
            row=next(row), column=0, columnspan=2, sticky="w", pady=(0, 4)
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

        self.mode_label = tk.StringVar()
        ttk.Label(
            frame, textvariable=self.mode_label, foreground="gray30"
        ).grid(row=next(row), column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.mode.trace_add("write", lambda *a: self._update_mode_label())
        self._update_mode_label()

        ttk.Label(
            frame,
            text="Arrow keys move the selected ellipse.",
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
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True
        )
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
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    def redraw_panel(self, key, preserve_limits=False):
        panel = self.panels[key]
        ax = panel.ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax.scatter(panel.x, panel.y, s=3, c="gray", alpha=0.3, rasterized=True)
        cmd = self.ccd_cmd.get() if key == "ccd" else self.qq_cmd.get()
        if cmd:
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
        for e in panel.ellipses:
            patch = Ellipse(
                e["center"],
                2 * e["rx"],
                2 * e["ry"],
                fill=False,
                edgecolor=e["color"],
                linewidth=2.5 if e is panel.selected else 1.4,
            )
            ax.add_patch(patch)

        # clusters selected by the OTHER panel's ellipses, shown here too
        other_key = self.other_key(key)
        other_panel = self.panels[other_key]
        for e in other_panel.ellipses:
            cross_mask = self.ellipse_full_mask(other_panel, e) & panel.mask
            ax.scatter(
                panel.x_all[cross_mask],
                panel.y_all[cross_mask],
                s=25,
                facecolors="none",
                edgecolors=e["color"],
                linewidths=1.2,
                label=f"{other_key}-selected (N={cross_mask.sum()})",
            )
        if other_panel.ellipses:
            ax.legend(loc="best", fontsize=8)

    def ellipse_full_mask(self, panel, ellipse):
        """Boolean mask, over the full catalog, of points inside `ellipse` and valid for `panel`."""
        x0, y0 = ellipse["center"]
        rx, ry = max(ellipse["rx"], 1e-12), max(ellipse["ry"], 1e-12)
        inside = ((panel.x_all - x0) / rx) ** 2 + (
            (panel.y_all - y0) / ry
        ) ** 2 <= 1
        return inside & panel.mask

    def count_in_ellipse(self, panel, ellipse):
        return int(self.ellipse_full_mask(panel, ellipse).sum())

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
        panel.drag_ellipse = None  # created lazily once drag exceeds threshold

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
            rx = abs(event.xdata - x0)
            ry = abs(event.ydata - y0)
            if panel.drag_ellipse is None:
                color = next(panel.color_cycle)
                patch = Ellipse(
                    (x0, y0),
                    2 * rx,
                    2 * ry,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.4,
                )
                panel.ax.add_patch(patch)
                panel.drag_ellipse = {
                    "patch": patch,
                    "color": color,
                    "center": (x0, y0),
                }
            else:
                panel.drag_ellipse["patch"].set_width(2 * rx)
                panel.drag_ellipse["patch"].set_height(2 * ry)
            self.canvas.draw_idle()

    def on_release(self, event):
        mode = self.mode.get()
        for key, panel in self.panels.items():
            if panel.press_xy is None:
                continue
            if mode == "add" and panel.drag_ellipse is not None:
                center = panel.drag_ellipse["center"]
                patch = panel.drag_ellipse["patch"]
                rx, ry = patch.get_width() / 2, patch.get_height() / 2
                color = panel.drag_ellipse["color"]
                patch.remove()
                panel.ellipses.append(
                    {"center": center, "rx": rx, "ry": ry, "color": color}
                )
                panel.selected = None
                self.redraw_panel(key, preserve_limits=True)
                self.redraw_panel(self.other_key(key), preserve_limits=True)
                self.canvas.draw_idle()
            elif (
                mode == "select"
                and event.inaxes is panel.ax
                and event.xdata is not None
            ):
                self.handle_click_select(key, panel, event.xdata, event.ydata)
            panel.press_xy = None
            panel.press_pixel = None
            panel.drag_ellipse = None

    def handle_click_select(self, key, panel, x, y):
        hit = None
        for e in reversed(panel.ellipses):
            x0, y0 = e["center"]
            rx, ry = max(e["rx"], 1e-12), max(e["ry"], 1e-12)
            if ((x - x0) / rx) ** 2 + ((y - y0) / ry) ** 2 <= 1:
                hit = e
                break
        panel.selected = hit
        self.redraw_panel(key, preserve_limits=True)
        self.canvas.draw_idle()
        self.status.set(
            f"Selected ellipse: N={self.count_in_ellipse(panel, hit)}"
            if hit
            else ""
        )

    # ---- Ellipse toolbar --------------------------------------------------

    def _update_mode_label(self):
        text = {
            "add": "Mode: Add new\ndrag on either panel to draw",
            "select": "Mode: Select\nclick inside an ellipse to select it",
        }.get(self.mode.get(), "")
        self.mode_label.set(text)

    def other_key(self, key):
        return "qq" if key == "ccd" else "ccd"

    def find_selected(self):
        """(key, panel) of whichever panel currently has a selected ellipse, or (None, None)."""
        for key, panel in self.panels.items():
            if panel.selected is not None:
                return key, panel
        return None, None

    def delete_selected(self):
        key, panel = self.find_selected()
        if panel is None:
            self.status.set("No ellipse selected")
            return
        panel.ellipses.remove(panel.selected)
        panel.selected = None
        self.redraw_panel(key, preserve_limits=True)
        self.redraw_panel(self.other_key(key), preserve_limits=True)
        self.canvas.draw_idle()
        self.status.set("")

    def unfocus_selected(self):
        key, panel = self.find_selected()
        if panel is not None:
            panel.selected = None
            self.redraw_panel(key, preserve_limits=True)
            self.canvas.draw_idle()
        self.status.set("")

    def move_selected(self, dx_sign, dy_sign):
        key, panel = self.find_selected()
        if panel is None:
            return
        xlim, ylim = panel.ax.get_xlim(), panel.ax.get_ylim()
        step_x = MOVE_STEP_FRAC * (xlim[1] - xlim[0]) * dx_sign
        step_y = MOVE_STEP_FRAC * (ylim[1] - ylim[0]) * dy_sign
        x0, y0 = panel.selected["center"]
        panel.selected["center"] = (x0 + step_x, y0 + step_y)
        self.redraw_panel(key, preserve_limits=True)
        self.redraw_panel(self.other_key(key), preserve_limits=True)
        self.canvas.draw_idle()

    def resize_selected(self, fx=1.0, fy=1.0):
        key, panel = self.find_selected()
        if panel is None:
            self.status.set("No ellipse selected")
            return
        panel.selected["rx"] *= fx
        panel.selected["ry"] *= fy
        self.redraw_panel(key, preserve_limits=True)
        self.redraw_panel(self.other_key(key), preserve_limits=True)
        self.canvas.draw_idle()

    def on_delete_key(self, event):
        self.delete_selected()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
