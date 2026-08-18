"""
Δφ Projection Comparison Plot
==============================
Loads the dphi_projection CSVs written by save_projections_to_csv() from
each Pythia filter configuration, overlays them on a single panel, and
compares against CMS data.

Directory layout assumed
------------------------
plots/Correlations/
    Both 1/             dphi_projection_*.csv
    Both 2/             dphi_projection_*.csv
    Decays Restricted/  dphi_projection_*.csv
    Primary Filter 1/   dphi_projection_*.csv
    Primary Filter 2/   dphi_projection_*.csv

CMS reference data
------------------
datathief/CMS_{trig_lo}-{trig_hi}_{assoc_lo}-{assoc_hi}.csv
    columns: DeltaPhi, d2Npair

Layout
------
This version is built for a 3-row x 4-column (12 panel) grid of
kinematic windows with NO gaps between panels:
  - all panels in the same row share the y-axis (ticks/label drawn once,
    on the leftmost column only)
  - all panels in the same column share the x-axis (ticks/label drawn
    once, on the bottom row only)
  - every panel gets a small text box with its trig/assoc pT ranges
  - the panel at (row 0, col 1) also gets a Delta-eta annotation
  - the panel at (row 0, col 2) also gets a sqrt(s) annotation
"""

import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer-Modern-style math, matches LaTeX default
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN = 19.2  # GeV/c  (lower edge of trigger pT window)
TRIG_PT_MAX = 48.0  # GeV/c  (upper edge of trigger pT window)

BASE_DIR = "plots/correlations/cms/expComparison"
CMS_DIR = "datathief"
OUTPUT_DIR = "plots/correlations/cms/expComparison"

SUBDIRS = ["19.2-48.0"]

SHOW_RATIO = False  # set False to suppress the CMS/Pythia ratio panel
MAX_COLS = 4  # maximum number of kinematic windows per row

# Colour and marker style per configuration (order matches SUBDIRS)
STYLE = {
    "19.2-48.0": {
        "label": "PYTHIA Data",
        "color": "#4453FF",
        "marker": "^",
        "ls": "-",
        "lw": 2,
        "zorder": 5,
    },
}

CMS_STYLE = {
    "color": "black",
    "marker": "*",
    "ms": 10,
    "ls": "-",
    "markerfacecolor": "black",
    "markeredgewidth": 0.8,
    "zorder": 1,
    "label": "CMS Data",
}

MS = 5  # marker size for Pythia points
CAPSIZE = 2  # error bar cap size

# ──────────────────────────────────────────────────────────────────────────────
# TODO — fill these in with your actual values.
# ──────────────────────────────────────────────────────────────────────────────
# Delta-eta range annotated in the (row=0, col=1) panel. Set this to whatever
# your analysis actually uses for the trigger-associate relative
# pseudorapidity cut (or just the single-hadron |eta| acceptance, if that's
# what you meant).
DETA_LABEL = r"$|\Delta\eta| < 1$"

# sqrt(s) annotated in the (row=0, col=2) panel. Set to whatever collision
# energy your PYTHIA / CMS comparison actually uses (e.g. 13 TeV, 7 TeV, ...).
SQRT_S_LABEL = r"$\sqrt{s} = 2.76\ \mathrm{TeV}$"

# (row, col) positions (0-indexed) that get the extra annotations above.
DETA_PANEL = (0, 1)
SQRT_S_PANEL = (0, 2)

# Top y-axis limit for each row (row 0 = top row, etc.). Since each row
# shares its y-axis, this sets the tallest tick shown for every panel
# in that row.
ROW_YMAX = [1.6, 1.6, 1.6]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_pt_ranges_from_filename(fname):
    """
    Extract trigger and associate pT ranges from a filename like
        dphi_projection_trig4.0-6.0_assoc2.0-4.0_pow3.csv
    Returns (trig_lo, trig_hi, assoc_lo, assoc_hi) as floats, or None.
    """
    m = re.search(
        r"trig([\d.]+)-([\d.]+)_assoc([\d.]+)-([\d.]+)", os.path.basename(fname)
    )
    if m:
        return tuple(float(x) for x in m.groups())
    return None


def load_dphi_csvs(base_dir, subdirs):
    """
    Scan each subdir for dphi_projection_*.csv files, group them by
    kinematic window, and return a nested dict:

        data[subdir][pt_key] = {
            'phi':        1D array,
            'central':    1D array,
            'err':        1D array,
            'background': float,
            'pt_ranges':  (trig_lo, trig_hi, assoc_lo, assoc_hi),
        }

    where pt_key = 'trig{lo}-{hi}_assoc{lo}-{hi}'.
    """
    all_data = {}
    all_ptkeys = set()

    for sd in subdirs:
        path = os.path.join(base_dir, sd)
        pattern = os.path.join(path, "dphi_projection_*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"  [warn] No dphi CSVs found in: {path}")
            all_data[sd] = {}
            continue

        all_data[sd] = {}

        for fpath in files:
            pt = _parse_pt_ranges_from_filename(fpath)
            if pt is None:
                print(f"  [warn] Cannot parse pT ranges from: {fpath}")
                continue

            trig_lo, trig_hi, assoc_lo, assoc_hi = pt
            pt_key = (
                f"trig{trig_lo:.1f}-{trig_hi:.1f}_assoc{assoc_lo:.1f}-{assoc_hi:.1f}"
            )

            df = pd.read_csv(fpath, comment="#")
            bkg_col = (
                df["zyam_background"].iloc[0]
                if "zyam_background" in df.columns
                else 0.0
            )

            # read error column — zeros if missing or all-NaN
            if "stat_err_jackknife" in df.columns:
                err = df["stat_err_jackknife"].fillna(0.0).values
            else:
                err = np.zeros(len(df))

            all_data[sd][pt_key] = {
                "phi": df["delta_phi_rad"].values,
                "central": df["dNpair_dDeltaPhi"].values,
                "err": err,
                "background": bkg_col,
                "pt_ranges": pt,
            }
            all_ptkeys.add(pt_key)
            print(f"  Loaded [{sd}] {pt_key}  ({len(df)} bins)")

    def _pt_key_order(key):
        """
        Sort so that rows = associate pT bands, columns = trigger pT bands.
        Primary key: (assoc_lo, assoc_hi)  → controls which row
        Secondary:   (trig_lo,  trig_hi)   → controls which column (left→right)
        """
        m = re.search(r"trig([\d.]+)-([\d.]+)_assoc([\d.]+)-([\d.]+)", key)
        if not m:
            return (0, 0, 0, 0)
        trig_lo, trig_hi, assoc_lo, assoc_hi = (float(x) for x in m.groups())
        return (assoc_lo, assoc_hi, trig_lo, trig_hi)  # <── assoc first

    return all_data, sorted(all_ptkeys, key=_pt_key_order)


def load_cms_data(cms_dir, trig_lo, trig_hi, assoc_lo, assoc_hi):
    """
    Try to load CMS reference data for a given kinematic window.
    Returns (phi, y) arrays or (None, None).
    """
    stem = f"CMS_{trig_lo:.0f}-{trig_hi:.0f}_{assoc_lo:.0f}-{assoc_hi:.0f}"
    for ext in (".csv", ".txt"):
        fpath = os.path.join(cms_dir, stem + ext)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                phi = df["DeltaPhi"].values
                y = df["d2Npair"].values
                idx = np.argsort(phi)
                return phi[idx], y[idx]
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] Could not read CMS file {fpath}: {e}")
    return None, None


def _phi_ticks():
    ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    return ticks, labels


def _bin_edges(phi, lo=0.0, hi=np.pi):
    """
    Convert bin-center x-values into bin edges so that the first bin starts
    exactly at *lo* and the last bin stretches all the way to *hi*, instead
    of stopping at the last data point (which is what plain ax.step with
    where='mid' does).
    """
    phi = np.asarray(phi)
    if len(phi) == 1:
        return np.array([lo, hi])
    mids = (phi[:-1] + phi[1:]) / 2.0
    edges = np.concatenate(([lo], mids, [hi]))
    return edges


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell drawing helpers
# ──────────────────────────────────────────────────────────────────────────────
def _draw_top_panel(
    ax,
    all_data,
    pt_key,
    subdirs,
    style,
    cms_phi,
    cms_y,
    pt_ranges,
    show_xlabel,
    show_ylabel,
    extra_annotation=None,
    hide_zero_label=False,
    show_legend=False,
):
    """Draw the Δφ projection panel into *ax*."""
    trig_lo, trig_hi, assoc_lo, assoc_hi = pt_ranges
    phi_ticks, phi_labels = _phi_ticks()

    for sd in subdirs:
        if pt_key not in all_data.get(sd, {}):
            continue
        entry = all_data[sd][pt_key]
        phi = entry["phi"]
        y = entry["central"]
        err = entry["err"]
        st = style[sd]

        edges = _bin_edges(phi)
        ax.stairs(
            y,
            edges,
            color=st["color"],
            linestyle="-",
            linewidth=3.0,
            label=st["label"],
            baseline=None,
            zorder=st["zorder"],
        )
        if np.any(err > 0):
            ax.errorbar(
                phi,
                y,
                yerr=err,
                fmt="none",
                ms=MS,
                color=st["color"],
                capsize=CAPSIZE,
                capthick=1.6,
                elinewidth=1.6,
                linestyle=st["ls"],
                linewidth=3.0,
                zorder=st["zorder"],
            )

    if cms_phi is not None:
        cms_edges = _bin_edges(cms_phi)
        ax.stairs(
            cms_y,
            cms_edges,
            color=CMS_STYLE["color"],
            linestyle=CMS_STYLE["ls"],
            linewidth=3.5,
            zorder=CMS_STYLE["zorder"],
            label="CMS Data",
            baseline=None,
        )

    ax.axhline(0, color="k", lw=2, ls="--", alpha=0.35)

    # ── momenta-range annotation (printed in every panel) ──────────────
    lines = [
        (
            f"{trig_lo:.1f} < $p_T^{{\\rm trig}}$ < {trig_hi:.1f} GeV/c\n"
            f"{assoc_lo:.1f} < $p_T^{{\\rm assoc}}$ < {assoc_hi:.1f} GeV/c"
        )
    ]
    if extra_annotation:
        lines.append(extra_annotation)
    ax.text(
        0.04,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        zorder=10,
    )

    if show_legend:
        ax.legend(fontsize=20, frameon=False, loc="center")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_xlim(-0.05, np.pi + 0.05)

    # ticks point inward so they don't get clipped by the touching neighbor
    ax.tick_params(axis="both", direction="in", top=True, right=True)

    if show_ylabel:
        ax.set_ylabel(
            r"$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d|\Delta\phi|}$", fontsize=30
        )
        ax.tick_params(axis="y", labelsize=20, labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if show_xlabel:
        ax.set_xlabel(r"$|\Delta\phi|$ [rad]", fontsize=20)
        ax.set_xticks(phi_ticks)
        labels = list(phi_labels)
        if hide_zero_label:
            labels[0] = ""
        ax.set_xticklabels(labels, fontsize=22)
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xticks(phi_ticks)
        ax.tick_params(axis="x", labelbottom=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine — one figure for all kinematic windows
# ──────────────────────────────────────────────────────────────────────────────
def plot_all_comparisons(
    all_data, pt_keys, cms_dir, save_dir, subdirs, style, show_ratio=True, max_cols=4
):
    """
    Build a single figure containing one panel per kinematic window
    (up to *max_cols* per row), all touching with no gaps. Panels in the
    same row share the y-axis (drawn once, leftmost column); panels in
    the same column share the x-axis (drawn once, bottom row).
    """
    n = len(pt_keys)
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)

    col_w = 5.6
    row_h = 4.6
    fig_w = col_w * ncols
    fig_h = row_h * nrows

    fig = plt.figure(figsize=(fig_w, fig_h))

    # No spacing between panels — this is what makes them touch.
    outer_gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=0.0,
        wspace=0.0,
    )

    axes = np.full((nrows, ncols), None, dtype=object)

    for idx, pt_key in enumerate(pt_keys):
        grow, gcol = divmod(idx, ncols)

        # Resolve pT ranges
        pt_ranges = None
        for sd in subdirs:
            if pt_key in all_data.get(sd, {}):
                pt_ranges = all_data[sd][pt_key]["pt_ranges"]
                break
        if pt_ranges is None:
            print(f"  [skip] No data found for {pt_key}")
            continue

        trig_lo, trig_hi, assoc_lo, assoc_hi = pt_ranges
        cms_phi, cms_y = load_cms_data(cms_dir, trig_lo, trig_hi, assoc_lo, assoc_hi)

        sharex = axes[0, gcol] if grow > 0 else None
        sharey = axes[grow, 0] if gcol > 0 else None
        ax = fig.add_subplot(outer_gs[grow, gcol], sharex=sharex, sharey=sharey)
        axes[grow, gcol] = ax

        show_xlabel = grow == nrows - 1
        show_ylabel = gcol == 0

        extra_annotation = None
        if (grow, gcol) == DETA_PANEL:
            extra_annotation = DETA_LABEL
        elif (grow, gcol) == SQRT_S_PANEL:
            extra_annotation = SQRT_S_LABEL

        _draw_top_panel(
            ax,
            all_data,
            pt_key,
            subdirs,
            style,
            cms_phi,
            cms_y,
            pt_ranges,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            extra_annotation=extra_annotation,
            hide_zero_label=(gcol > 0),
            show_legend=(grow == 0 and gcol == 0),
        )

    # Apply the per-row y-axis maximum and fixed tick set. Panels in a row
    # share their y-axis, so setting this on any one panel in the row
    # applies to all of them.
    Y_TICKS = [0.0, 0.5, 1.0, 1.5]
    for grow in range(nrows):
        if grow >= len(ROW_YMAX):
            continue
        row_axes = [ax for ax in axes[grow, :] if ax is not None]
        if row_axes:
            row_axes[0].set_ylim(top=ROW_YMAX[grow])
            row_axes[0].set_yticks(Y_TICKS)

    # Hide any unused cells in the last row
    for idx in range(n, nrows * ncols):
        grow, gcol = divmod(idx, ncols)
        fig.add_subplot(outer_gs[grow, gcol]).set_visible(False)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"dphi_comparison_trig{TRIG_PT_MIN}-{TRIG_PT_MAX}.pdf"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading dphi projection CSVs …")
    all_data, pt_keys = load_dphi_csvs(BASE_DIR, SUBDIRS)

    if not pt_keys:
        raise RuntimeError(
            f"No dphi_projection_*.csv files found under {BASE_DIR}. "
            "Run the main analysis script first."
        )

    print(f"\nFound kinematic windows: {pt_keys}")
    print(f"Saving comparison plot to: {OUTPUT_DIR}\n")

    path = plot_all_comparisons(
        all_data=all_data,
        pt_keys=pt_keys,
        cms_dir=CMS_DIR,
        save_dir=OUTPUT_DIR,
        subdirs=SUBDIRS,
        style=STYLE,
        show_ratio=SHOW_RATIO,
        max_cols=MAX_COLS,
    )

    print(f"\nDone. Figure written to: {path}")


if __name__ == "__main__":
    main()
