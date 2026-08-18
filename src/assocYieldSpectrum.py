"""
Dihadron Correlation Analysis — pT_assoc Slice Yields (quenched vs. vacuum)
============================================================================

Per-pT_assoc slice correlation functions with ZYAM subtraction, compared to
CMS data, with per-trigger integrated near- and away-side yields as a function
of pT_assoc.

This version runs the full pipeline TWICE — once over the "quenched" input
files and once over the "vacuum" input files (e.g.
dihadron_pow5_pT18to600_quenched_seed100.csv vs.
dihadron_pow5_pT18to600_vacuum_seed100.csv) — and overlays the quenched and
vacuum curves for a given pT_assoc slice / bin on the same axes (Δφ panels
and integrated-yield panels).

CMS data overlay and the Pythia/CMS validation ratio panel are each gated by
their own flag — SHOW_DATA_COMPARISON and SHOW_DATA_RATIO below, or the
matching show_data_comparison / show_data_ratio parameters on
plot_dphi_slices / plot_integrated_yields.

The quenched/vacuum ratio (an I_AA-like medium-modification factor) has
moved to IAA.py, which reuses compute_slice / run_variant / integrate_yield
/ integrate_near_away from this module but keeps its own independent set of
analysis constants.

Uncertainty method
------------------
Block jackknife, delegated to compute_phi_projection_jackknife from the base
analysis script, which returns both the per-bin central errors and the full
(n_blocks x n_phi) array of per-block projections.

Integrated yield uncertainties (within one variant) are computed by
integrating each block's projection first, then applying the jackknife
variance formula to the resulting scalar yields:

    Y_k    = sum_{phi in side} max(proj_k(phi), 0) * Dphi / DpT
    Var_JK = (N-1)/N * sum_k (Y_k - mean(Y_k))^2
    sigma  = sqrt(Var_JK)

This correctly accounts for the fact that removing one jackknife block shifts
the entire Dphi projection coherently — the phi bins are NOT independent, so
propagating per-bin errors in quadrature would be wrong.

pT_assoc slices:  [2,3], [3,4], [4,6], [6,8], [8,12], [12,14]  GeV/c
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Re-use helpers from the base analysis script
# ─────────────────────────────────────────────────────────────────────────────
from assocYieldProjection import (
    compute_phi_projection_jackknife,
    read_and_combine_bins,
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis configuration
# ─────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN = 19.2
TRIG_PT_MAX = 24.0  # GeV/c

ASSOC_PT_EDGES = [0.5, 1, 2, 3, 4, 6, 8, 12, 14]  # GeV/c

NEAR_SIDE_LIMIT = 1.0  # |Dphi| < this  → near side; beyond → away side

ETA_CUT = 1
ETA_BINS = 15
PHI_BINS = 33
ETA_RANGE = (-ETA_CUT, ETA_CUT)
PHI_RANGE = (-np.pi, np.pi)

ZYAM_PHI_LO = 0.5  # ZYAM search window (rad, in folded [0,pi] coords)
ZYAM_PHI_HI = 1.5

N_JACKKNIFE_BLOCKS = 5

# Two input variants — same pTHat bins / seeds, quenching hook toggled.
VARIANTS = ("quenched", "vacuum")
VARIANT_COLORS = {"quenched": "#E2432F", "vacuum": "#4453FF"}
VARIANT_LABELS = {"quenched": "Pythia (quenched)", "vacuum": "Pythia (vacuum)"}

# Which variant is treated as the "baseline" for the Pythia/CMS validation
# ratio panel (row 1 of the integrated-yield figure).
BASELINE_VARIANT = "vacuum"

# Toggle the CMS data overlay (Δφ panels + integrated-yield panels) and the
# Pythia({BASELINE_VARIANT})/CMS validation ratio row, independently.
SHOW_DATA_COMPARISON = False
SHOW_DATA_RATIO = False


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_slice(
    data,
    assoc_pt_min,
    assoc_pt_max,
    ntrig_override,
    trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
    eta_bins=ETA_BINS,
    phi_bins=PHI_BINS,
    eta_range=ETA_RANGE,
    phi_range=PHI_RANGE,
    zyam=True,
    zyam_range=None,
    n_blocks=N_JACKKNIFE_BLOCKS,
):
    """
    Run the full jackknife pipeline for one pT_assoc slice.

    Filters the combined pair DataFrame to the requested assoc pT window, then
    delegates to compute_phi_projection_jackknife which handles the 2D
    correlatio, eta projection, phi fold, ZYAM subtraction, and jackknife
    resampling internally.

    Parameters
    ----------
    data            : pandas.DataFrame  — full combined pair data
    assoc_pt_min    : float
    assoc_pt_max    : float
    ntrig_override  : float  — metadata['NTRIG'] from read_and_combine_bins
    trig_pt_range   : (lo, hi)  — recorded into the slice metadata
    eta_bins, phi_bins, eta_range, phi_range, zyam, zyam_range, n_blocks
                    : forwarded to compute_phi_projection_jackknife.
                      zyam_range defaults to {'phi': (ZYAM_PHI_LO, ZYAM_PHI_HI)}
                      (this module's default ZYAM search window) if not given.

    Returns
    -------
    dict with keys:
        phi_centers      : 1-D array in [0, pi]
        phi_proj         : ZYAM-subtracted, JK-central folded Dphi projection
        phi_proj_err     : per-bin JK 1-sigma errors  (informational only —
                           NOT used for yield uncertainties, see integrate_yield)
        jackknife_projs  : 2-D array (n_blocks x n_phi) of per-block projections
        background       : ZYAM level
        ntrig            : effective trigger count used
        assoc_pt_range   : (min, max)
    or None if the slice is empty.
    """
    if zyam_range is None:
        zyam_range = {"phi": (ZYAM_PHI_LO, ZYAM_PHI_HI)}

    assoc_mask = (data["assoc_pT"] > assoc_pt_min) & (data["assoc_pT"] <= assoc_pt_max)
    sliced = data[assoc_mask].copy()

    if len(sliced) == 0:
        print(f"  WARNING: No pairs in slice pT_assoc [{assoc_pt_min}, {assoc_pt_max}]")
        return None

    print(
        f"\n── pT_assoc [{assoc_pt_min}, {assoc_pt_max}] GeV/c  "
        f"({len(sliced):,} pairs) ──────────────────────────────────"
    )

    metadata_slice = {
        "CUT_TRIG_PT_RANGE": list(trig_pt_range),
        "CUT_ASSOC_PT_RANGE": [assoc_pt_min, assoc_pt_max],
    }

    (phi_proj, phi_proj_err, phi_centers, ntrig, bkg, jackknife_projs) = (
        compute_phi_projection_jackknife(
            sliced,
            metadata_slice,
            eta_bins=eta_bins,
            phi_bins=phi_bins,
            eta_range=eta_range,
            phi_range=phi_range,
            zyam=zyam,
            zyam_range=zyam_range,
            ntrig_override=ntrig_override,
            n_blocks=n_blocks,
        )
    )

    return {
        "phi_centers": phi_centers,
        "phi_proj": phi_proj,
        "phi_proj_err": phi_proj_err,
        "jackknife_projs": jackknife_projs,  # shape: (n_blocks, n_phi)
        "background": bkg,
        "ntrig": ntrig,
        "assoc_pt_range": (assoc_pt_min, assoc_pt_max),
    }


def run_variant(
    variant,
    base,
    pow_,
    trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
    assoc_pt_edges=ASSOC_PT_EDGES,
    eta_cut=ETA_CUT,
    eta_bins=ETA_BINS,
    phi_bins=PHI_BINS,
    phi_range=PHI_RANGE,
    zyam=True,
    zyam_range=None,
    n_blocks=N_JACKKNIFE_BLOCKS,
):
    """
    Load one variant's ("quenched" or "vacuum") pTHat bin files, combine them,
    and run compute_slice across every pT_assoc slice.

    Every analysis knob (trig_pt_range, assoc_pt_edges, eta_cut, eta_bins,
    phi_bins, phi_range, zyam, zyam_range, n_blocks) is a parameter rather
    than a hardcoded module constant, so callers in other scripts (e.g.
    IAA.py) can drive this with their own configuration instead of this
    module's.

    Returns a list (same length as assoc_pt_edges - 1) of compute_slice()
    results (or None for empty slices), aligned by index across variants.
    """
    tlo, thi = trig_pt_range
    filenames = sorted(glob.glob(f"{base}/dihadron_pow{pow_}_pT*_{variant}_seed*.csv"))
    if not filenames:
        raise FileNotFoundError(
            f"No files found at {base}/dihadron_pow{pow_}_pT*_{variant}_seed*.csv"
        )

    print(f"Found {len(filenames)} pTHat bin file(s) for '{variant}':")
    for f in filenames:
        print(f"  {f}")

    print(f"\nLoading and combining bins ({variant}) ...")
    data, metadata = read_and_combine_bins(
        filenames,
        trig_pt_min=tlo,
        trig_pt_max=thi,
        assoc_pt_min=assoc_pt_edges[0],
        assoc_pt_max=assoc_pt_edges[-1],
        deta_max=eta_cut,
    )
    ntrig_override = metadata["N_TRIG"]

    print(
        f"\nRunning {len(assoc_pt_edges) - 1} pT_assoc slices "
        f"({n_blocks} JK blocks each) for '{variant}' ..."
    )

    eta_range = (-eta_cut, eta_cut)
    return [
        compute_slice(
            data,
            assoc_pt_edges[j],
            assoc_pt_edges[j + 1],
            ntrig_override,
            trig_pt_range=trig_pt_range,
            eta_bins=eta_bins,
            phi_bins=phi_bins,
            eta_range=eta_range,
            phi_range=phi_range,
            zyam=zyam,
            zyam_range=zyam_range,
            n_blocks=n_blocks,
        )
        for j in range(len(assoc_pt_edges) - 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Yield integration
# ─────────────────────────────────────────────────────────────────────────────


def integrate_near_away(proj, phi_centers, near_mask, away_mask, bw, delta_pt):
    """Integrate a single (central or per-block) projection into near/away yields.

    Public (no leading underscore) since IAA.py reuses this directly for its
    quenched/vacuum ratio calculation.
    """
    signal = np.where(proj > 0, proj, 0.0)
    Y_near = float((signal[near_mask] * bw).sum() / delta_pt)
    Y_away = float((signal[away_mask] * bw).sum() / delta_pt)
    return Y_near, Y_away


def integrate_yield(
    phi_centers, phi_proj, jackknife_projs, assoc_pt_range, near_limit=NEAR_SIDE_LIMIT
):
    """
    Integrate the folded, ZYAM-subtracted Dphi projection into near- and
    away-side per-trigger yields with correct jackknife uncertainties.

    Each jackknife block's projection is integrated independently, and the
    jackknife variance formula is applied to the resulting scalar yields:

        Y_k    = sum_{phi in side} max(proj_k(phi), 0) * Dphi / DpT
        Var_JK = (N-1)/N * sum_k (Y_k - mean(Y_k))^2

    This is correct because removing one block shifts the whole projection
    coherently — the phi bins are correlated, so propagating per-bin errors
    in quadrature would undercount the true variance.

    Parameters
    ----------
    phi_centers     : 1-D array  [0, pi]
    phi_proj        : 1-D array  ZYAM-subtracted central projection
    jackknife_projs : 2-D array  (n_blocks x n_phi) per-block projections
    assoc_pt_range  : (lo, hi)
    near_limit      : float (rad)

    Returns
    -------
    Y_near, Y_away, sigma_near, sigma_away : float x 4
    """
    bw = phi_centers[1] - phi_centers[0]
    delta_pt = assoc_pt_range[1] - assoc_pt_range[0]

    near_mask = phi_centers <= near_limit
    away_mask = phi_centers > near_limit

    Y_near, Y_away = integrate_near_away(
        phi_proj, phi_centers, near_mask, away_mask, bw, delta_pt
    )

    # Per-block yields — shape (n_blocks,) each
    n_blocks = len(jackknife_projs)
    jk_near = np.array(
        [
            integrate_near_away(
                jackknife_projs[k], phi_centers, near_mask, away_mask, bw, delta_pt
            )[0]
            for k in range(n_blocks)
        ]
    )
    jk_away = np.array(
        [
            integrate_near_away(
                jackknife_projs[k], phi_centers, near_mask, away_mask, bw, delta_pt
            )[1]
            for k in range(n_blocks)
        ]
    )

    # Jackknife variance: Var_JK = (N-1)/N * sum_k (theta_k - theta_bar)^2
    factor = (n_blocks - 1) / n_blocks
    sigma_near = float(np.sqrt(factor * np.sum((jk_near - jk_near.mean()) ** 2)))
    sigma_away = float(np.sqrt(factor * np.sum((jk_away - jk_away.mean()) ** 2)))

    return Y_near, Y_away, sigma_near, sigma_away


# ─────────────────────────────────────────────────────────────────────────────
# CMS data loaders
# ─────────────────────────────────────────────────────────────────────────────


def _load_csv(path):
    """Read a two-column CSV, sort by first column. Returns (x, y) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        idx = np.argsort(x)
        return x[idx], y[idx]
    except (pd.errors.ParserError, OSError, ValueError, KeyError, IndexError) as e:
        print(f"  Warning: could not read {path}: {e}")
        return None, None


def load_cms_dphi(trig_pt_range, assoc_pt_range, data_dir="datathief"):
    tlo, thi = trig_pt_range
    alo, ahi = assoc_pt_range
    base = f"{data_dir}/CMS_{tlo:.0f}-{thi:.0f}_{alo:.0f}-{ahi:.0f}"
    for ext in (".csv", ".txt"):
        x, y = _load_csv(base + ext)
        if x is not None:
            return x, y
    return None, None


def load_cms_yield(trig_pt_range, away=True, data_dir="datathief"):
    tlo, thi = trig_pt_range
    side = "away" if away else "near"
    base = f"{data_dir}/CMS_{side}_{tlo:.0f}-{thi:.0f}"
    for ext in (".csv", ".txt"):
        x, y = _load_csv(base + ext)
        if x is not None:
            return x, y
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_PHI_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
_PHI_LABELS = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]


def plot_dphi_slices(
    results_by_variant,
    trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
    cms_dir="datathief",
    save_path=None,
    show_data_comparison=SHOW_DATA_COMPARISON,
    near_limit=NEAR_SIDE_LIMIT,
    variant_colors=VARIANT_COLORS,
    variant_labels=VARIANT_LABELS,
):
    """
    One panel per pT_assoc slice: quenched + vacuum ZYAM-subtracted Dphi
    curves overlaid on the same axes, plus CMS data where available and
    show_data_comparison is True.
    """
    res_q = [r for r in results_by_variant["quenched"] if r is not None]
    res_v = [r for r in results_by_variant["vacuum"] if r is not None]

    if len(res_q) != len(res_v):
        print(
            "  WARNING: quenched/vacuum have different numbers of valid "
            "pT_assoc slices — pairing by index up to the shorter list."
        )
    n_slices = min(len(res_q), len(res_v))
    res_q, res_v = res_q[:n_slices], res_v[:n_slices]

    ncols = 3
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
    )

    for i in range(n_slices):
        ax = axes[i // ncols][i % ncols]
        alo, ahi = res_v[i]["assoc_pt_range"]

        if show_data_comparison:
            cms_phi, cms_y = load_cms_dphi(trig_pt_range, (alo, ahi), data_dir=cms_dir)
            if cms_phi is not None:
                ax.step(
                    cms_phi,
                    cms_y,
                    where="mid",
                    color="black",
                    linestyle="-",
                    linewidth=2,
                    label="CMS data",
                    zorder=10,
                )

        for variant, res in (("vacuum", res_v[i]), ("quenched", res_q[i])):
            phi = res["phi_centers"]
            sig = res["phi_proj"]
            err = res["phi_proj_err"]
            color = variant_colors[variant]

            ax.step(
                phi,
                sig,
                where="mid",
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{variant_labels[variant]}  ZYAM={res['background']:.4f}",
                zorder=4,
            )
            ax.errorbar(
                phi,
                sig,
                yerr=err,
                fmt="none",
                ls="none",
                color=color,
                ms=5,
                lw=1.4,
                zorder=4,
            )

        ax.axvline(near_limit, color="green", lw=1.4, ls="--", alpha=0.7)
        ax.axhline(0, lw=1.4, ls="--", color="black", alpha=0.4)
        ax.set_xlim(-0.05, np.pi + 0.05)
        ax.set_xticks(_PHI_TICKS)
        ax.set_xticklabels(_PHI_LABELS, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$|\Delta\phi|$ [rad]", fontsize=13)
        ax.set_ylabel(
            r"$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d|\Delta\phi|}$", fontsize=17
        )
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(
            rf"Assoc $p_T$: {alo:.1f}–{ahi:.1f} GeV/c",
            fontsize=13,
            fontweight="bold",
            pad=6,
        )
        ax.legend(fontsize=7, framealpha=0.2)
        ax.grid(True, alpha=0.15)

    for j in range(n_slices, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"CMS Experimental Dihadron $\Delta\phi$ Distributions" + "\n"
        rf"Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c" + "\n",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig, axes


def _power_law_fit(x, y, pt_min=5.0):
    mask = (x >= pt_min) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None, None
    n, log_A = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return np.exp(log_A), n


def plot_integrated_yields(
    results_by_variant,
    trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
    cms_dir="datathief",
    save_path=None,
    baseline_variant=BASELINE_VARIANT,
    show_data_comparison=SHOW_DATA_COMPARISON,
    show_data_ratio=SHOW_DATA_RATIO,
    variant_colors=VARIANT_COLORS,
    variant_labels=VARIANT_LABELS,
):
    """
    1x2 or 2x2 grid (near-side / away-side columns):

      row 0 - integrated yields: quenched + vacuum, overlaid on the same
              axes; CMS data + power-law fit added too if
              show_data_comparison is True
      row 1 - Pythia({baseline_variant}) / CMS ratio (existing baseline
              validation against data) — only drawn if show_data_ratio is
              True (which also requires show_data_comparison, since there's
              no data to ratio against otherwise)

    Error bars are block-jackknife 1-sigma throughout. The quenched/vacuum
    ratio (I_AA-like) has moved to IAA.py.
    """
    if show_data_ratio and not show_data_comparison:
        print(
            "  show_data_ratio=True requires show_data_comparison=True "
            "(nothing to ratio against) — disabling the ratio row."
        )
        show_data_ratio = False

    res_q = [r for r in results_by_variant["quenched"] if r is not None]
    res_v = [r for r in results_by_variant["vacuum"] if r is not None]
    if not res_q or not res_v:
        print("Missing quenched or vacuum results — nothing to plot.")
        return None, None
    if len(res_q) != len(res_v):
        print(
            "  WARNING: quenched/vacuum have different numbers of valid "
            "pT_assoc slices — pairing by index up to the shorter list."
        )
    n_slices = min(len(res_q), len(res_v))
    res_q, res_v = res_q[:n_slices], res_v[:n_slices]

    # pT bin centers / errors (assumes identical ASSOC_PT_EDGES binning
    # across variants, which run() enforces)
    pt_centers, xerr_lo, xerr_hi = [], [], []
    for r in res_v:
        alo, ahi = r["assoc_pt_range"]
        pt_c = (alo + ahi) / 2
        pt_centers.append(pt_c)
        xerr_lo.append(pt_c - alo)
        xerr_hi.append(ahi - pt_c)
    pt_centers = np.array(pt_centers)
    xerr = [xerr_lo, xerr_hi]
    x_fit = np.linspace(5 * 0.95, pt_centers.max() * 1.05, 300)

    # Integrated yields per variant
    variant_yields = {}  # variant -> {'near': (Y, err), 'away': (Y, err)}
    for name, res_list in (("quenched", res_q), ("vacuum", res_v)):
        yn, ya, en, ea = [], [], [], []
        for r in res_list:
            Yn, Ya, sn, sa = integrate_yield(
                r["phi_centers"],
                r["phi_proj"],
                r["jackknife_projs"],
                assoc_pt_range=r["assoc_pt_range"],
            )
            yn.append(Yn)
            ya.append(Ya)
            en.append(sn)
            ea.append(sa)
        variant_yields[name] = {
            "near": (np.array(yn), np.array(en)),
            "away": (np.array(ya), np.array(ea)),
        }

    nrows = 2 if show_data_ratio else 1
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(10, 5.5 if nrows == 1 else 7),
        squeeze=False,
        gridspec_kw=(
            {"height_ratios": [2, 1], "hspace": 0.1, "wspace": 0.32}
            if nrows == 2
            else {"wspace": 0.32}
        ),
    )

    sides = [
        ("near", False, "Near side", r"$|\Delta\phi| < 1$ rad"),
        ("away", True, "Away side", r"$1 < |\Delta\phi| < \pi$ rad"),
    ]

    for col, (side_key, away, label, phi_label) in enumerate(sides):
        ax_top = axes[0, col]
        ax_cms_rat = axes[1, col] if show_data_ratio else None

        cms_pt, cms_y = (None, None)
        if show_data_comparison:
            cms_pt, cms_y = load_cms_yield(trig_pt_range, away=away, data_dir=cms_dir)
        if cms_pt is not None:
            cms_pt, cms_y = np.array(cms_pt), np.array(cms_y)
            ax_top.errorbar(
                cms_pt,
                cms_y,
                fmt="*",
                color="black",
                ms=7,
                markeredgewidth=0.6,
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                label="CMS data",
                zorder=6,
            )
            A_cms, n_cms = _power_law_fit(cms_pt, cms_y)
            if A_cms is not None:
                ax_top.plot(
                    x_fit,
                    A_cms * x_fit**n_cms,
                    "--",
                    color="black",
                    lw=1.6,
                    alpha=0.85,
                    label=rf"CMS fit  ($n={-n_cms:.2f}$)",
                    zorder=6,
                )
        elif show_data_comparison:
            ax_top.text(
                0.97,
                0.95,
                "no CMS file",
                transform=ax_top.transAxes,
                fontsize=7,
                ha="right",
                va="top",
            )

        # quenched + vacuum overlaid on the same axes
        for variant in ("vacuum", "quenched"):
            yields, errs = variant_yields[variant][side_key]
            color = variant_colors[variant]
            ax_top.errorbar(
                pt_centers,
                yields,
                xerr=xerr,
                yerr=errs,
                fmt="o",
                color=color,
                ms=6,
                markeredgewidth=0.6,
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                label=variant_labels[variant],
                zorder=5,
            )
            A_py, n_py = _power_law_fit(pt_centers, yields)
            if A_py is not None:
                ax_top.plot(
                    x_fit,
                    A_py * x_fit**n_py,
                    "--",
                    color=color,
                    lw=1.4,
                    alpha=0.8,
                    label=rf"{variant} fit  ($n={-n_py:.2f}$)",
                    zorder=5,
                )

        if side_key == "near":
            ax_top.set_ylabel(r"$Y^{\mathrm{near}}$", fontsize=13)
        else:
            ax_top.set_ylabel(r"$Y^{\mathrm{away}}$", fontsize=13)

        ax_top.set_title(
            rf"{label}" + "\n" + rf"{phi_label}", fontsize=13, fontweight="bold", pad=8
        )
        ax_top.legend(fontsize=6.5, framealpha=0.2, loc="upper right")
        ax_top.grid(True, alpha=0.15)
        ax_top.set_yscale("log")
        ax_top.tick_params(labelbottom=not show_data_ratio, labelsize=12)
        if not show_data_ratio:
            ax_top.set_xlabel(r"$p_T^{\rm assoc}$ [GeV/c]", fontsize=13)

        # Row 1: Pythia(baseline_variant) / CMS — existing validation ratio
        if show_data_ratio:
            base_y, base_err = variant_yields[baseline_variant][side_key]
            base_color = variant_colors[baseline_variant]
            if cms_pt is not None and len(cms_pt) == len(pt_centers):
                cms_interp = np.interp(pt_centers, cms_pt, cms_y)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cms_ratio = np.where(cms_interp != 0, base_y / cms_interp, np.nan)
                    cms_ratio_err = np.where(
                        cms_interp != 0, base_err / cms_interp, np.nan
                    )
                ax_cms_rat.errorbar(
                    pt_centers,
                    cms_ratio,
                    xerr=xerr,
                    yerr=cms_ratio_err,
                    fmt="o",
                    color=base_color,
                    ms=6,
                    markeredgewidth=0.6,
                    ecolor=base_color,
                    elinewidth=1.5,
                    capsize=4,
                    zorder=4,
                )
            else:
                ax_cms_rat.text(
                    0.5,
                    0.5,
                    "CMS data not loaded",
                    ha="center",
                    va="center",
                    transform=ax_cms_rat.transAxes,
                    fontsize=9,
                    color="gray",
                )

            ax_cms_rat.axhline(1, lw=1.4, ls="--", color="black", alpha=0.4)
            ax_cms_rat.set_xlabel(r"$p_T^{\rm assoc}$ [GeV/c]", fontsize=13)
            ax_cms_rat.set_ylabel(f"{baseline_variant.title()}/CMS", fontsize=10.5)
            ax_cms_rat.set_ylim(0.5, 2)
            ax_cms_rat.grid(True, alpha=0.15)
            ax_cms_rat.tick_params(labelsize=11)

    fig.suptitle(
        rf"CMS Experimental Per-Trigger Integrated Yields"
        "\n"
        rf"Trig $p_T$: {trig_pt_range[0]:.1f}-{trig_pt_range[1]:.1f} GeV/c",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────


def print_yield_table(slice_results, variant=""):
    W = 130

    # Column widths
    w_pt = 20
    w_y = 14
    w_rel = 12

    if variant:
        print(f"\n{variant.upper()}")
    print("─" * W)

    header = (
        f"{'pT_assoc [GeV/c]':^{w_pt}}"
        f"{'Y_near':^{w_y}}"
        f"{'rel %(near)':^{w_rel}}"
        f"{'Y_away':^{w_y}}"
        f"{'rel %(away)':^{w_rel}}"
        f"{'Y_total':^{w_y}}"
        f"{'rel %(total)':^{w_rel}}"
    )
    print(header)
    print("─" * W)

    for res in (r for r in slice_results if r is not None):
        alo, ahi = res["assoc_pt_range"]

        Yn, Ya, sn, sa = integrate_yield(
            res["phi_centers"],
            res["phi_proj"],
            res["jackknife_projs"],
            assoc_pt_range=res["assoc_pt_range"],
        )

        # Total yield and its jackknife uncertainty (integrate over all phi)
        bw = res["phi_centers"][1] - res["phi_centers"][0]
        delta_pt = ahi - alo
        signal = np.where(res["phi_proj"] > 0, res["phi_proj"], 0.0)
        Ytot = float((signal * bw).sum() / delta_pt)

        n_blocks = len(res["jackknife_projs"])
        jk_tot = np.array(
            [
                float(
                    (
                        np.where(
                            res["jackknife_projs"][k] > 0,
                            res["jackknife_projs"][k],
                            0.0,
                        )
                        * bw
                    ).sum()
                    / delta_pt
                )
                for k in range(n_blocks)
            ]
        )
        factor = (n_blocks - 1) / n_blocks
        stot = float(np.sqrt(factor * np.sum((jk_tot - jk_tot.mean()) ** 2)))

        # Relative uncertainties (%)
        rel_n = (sn / Yn * 100) if Yn != 0 else 0.0
        rel_a = (sa / Ya * 100) if Ya != 0 else 0.0
        rel_tot = (stot / Ytot * 100) if Ytot != 0 else 0.0

        row = (
            f"{f'[{alo:>2.0f}, {ahi:>3.0f}]':^{w_pt}}"
            f"{Yn:>{w_y}.5f}"
            f"{rel_n:>{w_rel}.2f}"
            f"{Ya:>{w_y}.5f}"
            f"{rel_a:>{w_rel}.2f}"
            f"{Ytot:>{w_y}.5f}"
            f"{rel_tot:>{w_rel}.2f}"
        )
        print(row)

    print("─" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────


def save_yield_csv(
    slice_results, trig_pt_range, out_dir, variant, n_blocks=N_JACKKNIFE_BLOCKS
):
    """
    Write per-trigger integrated near- and away-side yields for one variant
    to a CSV file, matching the header/comment format of
    integrated_yields_trig8-15.csv.
    """
    tlo, thi = trig_pt_range
    save_path = os.path.join(
        out_dir, f"integrated_yields_{variant}_trig{tlo:.0f}-{thi:.0f}.csv"
    )

    rows = []
    for res in (r for r in slice_results if r is not None):
        alo, ahi = res["assoc_pt_range"]
        Yn, Ya, sn, sa = integrate_yield(
            res["phi_centers"],
            res["phi_proj"],
            res["jackknife_projs"],
            assoc_pt_range=res["assoc_pt_range"],
        )
        rows.append((alo, ahi, (alo + ahi) / 2, Yn, sn, Ya, sa))

    with open(save_path, "w") as f:
        f.write(f"# Per-trigger integrated near/away yields vs pT_assoc ({variant})\n")
        f.write(f"# trigger pT : {tlo:.1f} \u2013 {thi:.1f} GeV/c\n")
        f.write(f"# jackknife blocks : {n_blocks}\n")
        f.write("# Near side : -pi/2 < Delta-phi <= pi/2\n")
        f.write("# Away side :  pi/2 < Delta-phi <= 3pi/2\n")
        f.write("#\n")
        f.write(
            "assoc_pt_lo,assoc_pt_hi,assoc_pt_center,Y_near,err_near,Y_away,err_away\n"
        )
        for alo, ahi, ptc, Yn, sn, Ya, sa in rows:
            f.write(f"{alo},{ahi},{ptc},{Yn},{sn},{Ya},{sa}\n")

    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    POW = 3
    BASE = f"pythiaData/2760/cms/{TRIG_PT_MIN}-{TRIG_PT_MAX}"
    trig_range = (TRIG_PT_MIN, TRIG_PT_MAX)

    results = {}
    for variant in VARIANTS:
        print(f"\n{'=' * 80}\nRunning '{variant}' pass\n{'=' * 80}")
        results[variant] = run_variant(variant, BASE, POW, trig_pt_range=trig_range)

    for variant in VARIANTS:
        print_yield_table(results[variant], variant=variant)

    out_dir = "outputs/plots/yields"
    os.makedirs(out_dir, exist_ok=True)

    print("Saving integrated yields to CSV ...")
    for variant in VARIANTS:
        save_yield_csv(results[variant], trig_range, out_dir, variant=variant)

    print("Generating Delta-phi slice plot (quenched + vacuum overlaid) ...")
    plot_dphi_slices(
        results,
        trig_pt_range=trig_range,
        cms_dir="datathief",
        save_path=f"{out_dir}/dphi_slices_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.svg",
    )

    print("Generating integrated yield + ratio plot ...")
    plot_integrated_yields(
        results,
        trig_pt_range=trig_range,
        cms_dir="datathief",
        save_path=f"{out_dir}/integrated_yields_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.svg",
    )


if __name__ == "__main__":
    main()
