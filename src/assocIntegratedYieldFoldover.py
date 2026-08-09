"""
Dihadron Correlation Analysis — pT_assoc Slice Yields (quenched vs. vacuum)
============================================================================

Per-pT_assoc slice correlation functions with ZYAM subtraction, compared to
CMS data, with per-trigger integrated near- and away-side yields as a function
of pT_assoc.

This version runs the full pipeline TWICE — once over the "quenched" input
files and once over the "vacuum" input files (e.g.
dihadron_pow5_pT18to600_quenched_seed100.csv vs.
dihadron_pow5_pT18to600_vacuum_seed100.csv) — and:

  * overlays the quenched and vacuum curves for a given pT_assoc slice / bin
    on the same axes (Δφ panels and integrated-yield panels),
  * adds a quenched/vacuum ratio panel (an I_AA-like medium-modification
    factor), in addition to the existing Pythia/CMS validation ratio.

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

Quenched/vacuum ratio uncertainty
----------------------------------
The quenched and vacuum files share the same seed per pTHat bin (matched
events, quenching hook toggled on/off), so jackknife block k corresponds to
the same underlying event subset in both variants. The ratio is therefore
built block-by-block *before* taking the jackknife variance:

    R_k    = Y_k^quenched / Y_k^vacuum
    Var_JK = (N-1)/N * sum_k (R_k - mean(R_k))^2

This propagates the correlation between the matched samples correctly.
Independent quadrature (sigma_R/R = sqrt((sigma_q/Y_q)^2 + (sigma_v/Y_v)^2))
would overestimate the uncertainty for matched events and is NOT used here.

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
from projectionsFoldover import (
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

zyam = True

N_JACKKNIFE_BLOCKS = 50

# Two input variants — same pTHat bins / seeds, quenching hook toggled.
VARIANTS = ("quenched", "vacuum")
VARIANT_COLORS = {"quenched": "#E2432F", "vacuum": "#4453FF"}
VARIANT_LABELS = {"quenched": "Pythia (quenched)", "vacuum": "Pythia (vacuum)"}

# Which variant is treated as the "baseline" for the existing Pythia/CMS
# validation ratio panel (row 2 of the integrated-yield figure).
BASELINE_VARIANT = "vacuum"


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_slice(data, assoc_pt_min, assoc_pt_max, ntrig_override):
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
        "CUT_TRIG_PT_RANGE": [TRIG_PT_MIN, TRIG_PT_MAX],
        "CUT_ASSOC_PT_RANGE": [assoc_pt_min, assoc_pt_max],
    }

    (phi_proj, phi_proj_err, phi_centers, ntrig, bkg, jackknife_projs) = (
        compute_phi_projection_jackknife(
            sliced,
            metadata_slice,
            eta_bins=ETA_BINS,
            phi_bins=PHI_BINS,
            eta_range=ETA_RANGE,
            phi_range=PHI_RANGE,
            zyam=zyam,
            zyam_range={"phi": (ZYAM_PHI_LO, ZYAM_PHI_HI)},
            ntrig_override=ntrig_override,
            n_blocks=N_JACKKNIFE_BLOCKS,
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


def run_variant(variant, base, pow_, trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX)):
    """
    Load one variant's ("quenched" or "vacuum") pTHat bin files, combine them,
    and run compute_slice across every pT_assoc slice.

    Returns a list (same length as ASSOC_PT_EDGES - 1) of compute_slice()
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
        assoc_pt_min=ASSOC_PT_EDGES[0],
        assoc_pt_max=ASSOC_PT_EDGES[-1],
        deta_max=ETA_CUT,
    )
    ntrig_override = metadata["N_TRIG"]

    print(
        f"\nRunning {len(ASSOC_PT_EDGES) - 1} pT_assoc slices "
        f"({N_JACKKNIFE_BLOCKS} JK blocks each) for '{variant}' ..."
    )

    return [
        compute_slice(data, ASSOC_PT_EDGES[j], ASSOC_PT_EDGES[j + 1], ntrig_override)
        for j in range(len(ASSOC_PT_EDGES) - 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Yield integration
# ─────────────────────────────────────────────────────────────────────────────


def _integrate_one(proj, phi_centers, near_mask, away_mask, bw, delta_pt):
    """Integrate a single (central or per-block) projection into near/away yields."""
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

    Y_near, Y_away = _integrate_one(
        phi_proj, phi_centers, near_mask, away_mask, bw, delta_pt
    )

    # Per-block yields — shape (n_blocks,) each
    n_blocks = len(jackknife_projs)
    jk_near = np.array(
        [
            _integrate_one(
                jackknife_projs[k], phi_centers, near_mask, away_mask, bw, delta_pt
            )[0]
            for k in range(n_blocks)
        ]
    )
    jk_away = np.array(
        [
            _integrate_one(
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


def compute_variant_ratio(res_q, res_v, near_limit=NEAR_SIDE_LIMIT):
    """
    I_AA-like ratio Y_quenched / Y_vacuum for one pT_assoc slice, with
    matched-block jackknife uncertainty (see module docstring).

    Parameters
    ----------
    res_q, res_v : dict  — compute_slice() results for the same pT_assoc
                   slice, quenched and vacuum respectively.

    Returns
    -------
    R_near, R_away, sigma_near, sigma_away : float x 4
    """
    phi_c = res_q["phi_centers"]
    bw = phi_c[1] - phi_c[0]
    dpt_q = res_q["assoc_pt_range"][1] - res_q["assoc_pt_range"][0]
    dpt_v = res_v["assoc_pt_range"][1] - res_v["assoc_pt_range"][0]

    near_mask = phi_c <= near_limit
    away_mask = phi_c > near_limit

    Yn_q, Ya_q = _integrate_one(
        res_q["phi_proj"], phi_c, near_mask, away_mask, bw, dpt_q
    )
    Yn_v, Ya_v = _integrate_one(
        res_v["phi_proj"], phi_c, near_mask, away_mask, bw, dpt_v
    )

    R_near = Yn_q / Yn_v if Yn_v != 0 else np.nan
    R_away = Ya_q / Ya_v if Ya_v != 0 else np.nan

    jk_q, jk_v = res_q["jackknife_projs"], res_v["jackknife_projs"]
    n_blocks = min(len(jk_q), len(jk_v))
    if len(jk_q) != len(jk_v):
        print(
            "  WARNING: quenched/vacuum jackknife block counts differ "
            f"({len(jk_q)} vs {len(jk_v)}) — truncating to {n_blocks} "
            "for the matched-block ratio."
        )

    r_near_k, r_away_k = [], []
    for k in range(n_blocks):
        yn_q, ya_q = _integrate_one(jk_q[k], phi_c, near_mask, away_mask, bw, dpt_q)
        yn_v, ya_v = _integrate_one(jk_v[k], phi_c, near_mask, away_mask, bw, dpt_v)
        r_near_k.append(yn_q / yn_v if yn_v != 0 else np.nan)
        r_away_k.append(ya_q / ya_v if ya_v != 0 else np.nan)
    r_near_k = np.array(r_near_k)
    r_away_k = np.array(r_away_k)

    factor = (n_blocks - 1) / n_blocks
    sigma_near = float(
        np.sqrt(factor * np.nansum((r_near_k - np.nanmean(r_near_k)) ** 2))
    )
    sigma_away = float(
        np.sqrt(factor * np.nansum((r_away_k - np.nanmean(r_away_k)) ** 2))
    )

    return R_near, R_away, sigma_near, sigma_away


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
):
    """
    One panel per pT_assoc slice: quenched + vacuum ZYAM-subtracted Dphi
    curves overlaid on the same axes, plus CMS data where available.
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
            color = VARIANT_COLORS[variant]

            ax.step(
                phi,
                sig,
                where="mid",
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{VARIANT_LABELS[variant]}  ZYAM={res['background']:.4f}",
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

        ax.axvline(NEAR_SIDE_LIMIT, color="green", lw=1.4, ls="--", alpha=0.7)
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
):
    """
    3x2 grid (near-side / away-side columns):

      row 0 - integrated yields: CMS data + quenched + vacuum, overlaid on
              the same axes
      row 1 - Pythia({baseline_variant}) / CMS ratio (existing baseline
              validation against data)
      row 2 - quenched / vacuum ratio — an I_AA-like medium-modification
              factor, with matched-block jackknife uncertainty
              (see compute_variant_ratio)

    Error bars are block-jackknife 1-sigma throughout.
    """
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

    # Quenched/vacuum ratio, matched-block jackknife
    ratio_near, ratio_away, ratio_near_err, ratio_away_err = [], [], [], []
    for rq, rv in zip(res_q, res_v):
        Rn, Ra, sn, sa = compute_variant_ratio(rq, rv)
        ratio_near.append(Rn)
        ratio_away.append(Ra)
        ratio_near_err.append(sn)
        ratio_away_err.append(sa)
    ratio_near, ratio_away = np.array(ratio_near), np.array(ratio_away)
    ratio_near_err, ratio_away_err = np.array(ratio_near_err), np.array(ratio_away_err)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 8.5),
        gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.1, "wspace": 0.32},
    )

    sides = [
        (
            "near",
            False,
            "Near side",
            r"$|\Delta\phi| < 1$ rad",
            ratio_near,
            ratio_near_err,
        ),
        (
            "away",
            True,
            "Away side",
            r"$1 < |\Delta\phi| < \pi$ rad",
            ratio_away,
            ratio_away_err,
        ),
    ]

    for col, (side_key, away, label, phi_label, r_ratio, r_ratio_err) in enumerate(
        sides
    ):
        ax_top, ax_cms_rat, ax_qv_rat = axes[0, col], axes[1, col], axes[2, col]

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
        else:
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
            color = VARIANT_COLORS[variant]
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
                label=VARIANT_LABELS[variant],
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
        ax_top.tick_params(labelbottom=False, labelsize=12)

        # Row 1: Pythia(baseline_variant) / CMS — existing validation ratio
        base_y, base_err = variant_yields[baseline_variant][side_key]
        base_color = VARIANT_COLORS[baseline_variant]
        if cms_pt is not None and len(cms_pt) == len(pt_centers):
            cms_interp = np.interp(pt_centers, cms_pt, cms_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                cms_ratio = np.where(cms_interp != 0, base_y / cms_interp, np.nan)
                cms_ratio_err = np.where(cms_interp != 0, base_err / cms_interp, np.nan)
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
        ax_cms_rat.set_ylabel(f"{baseline_variant.title()}/CMS", fontsize=10.5)
        ax_cms_rat.set_ylim(0.5, 2)
        ax_cms_rat.grid(True, alpha=0.15)
        ax_cms_rat.tick_params(labelsize=11, labelbottom=False)

        # Row 2: quenched/vacuum ratio (I_AA-like)
        ax_qv_rat.errorbar(
            pt_centers,
            r_ratio,
            xerr=xerr,
            yerr=r_ratio_err,
            fmt="o",
            color="#333333",
            ms=6,
            markeredgewidth=0.6,
            ecolor="#333333",
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )
        ax_qv_rat.axhline(1, lw=1.4, ls="--", color="black", alpha=0.4)
        ax_qv_rat.set_xlabel(r"$p_T^{\rm assoc}$ [GeV/c]", fontsize=13)
        ax_qv_rat.set_ylabel("Quenched/Vacuum", fontsize=10.5)
        ax_qv_rat.set_ylim(0, 5)
        ax_qv_rat.grid(True, alpha=0.15)
        ax_qv_rat.tick_params(labelsize=11)

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


def print_ratio_table(res_q, res_v):
    """Print the quenched/vacuum ratio (I_AA-like) summary table."""
    n = min(len(res_q), len(res_v))
    W = 70
    w_pt, w_r, w_err = 20, 14, 12

    print("\nQUENCHED / VACUUM RATIO")
    print("─" * W)
    header = (
        f"{'pT_assoc [GeV/c]':^{w_pt}}"
        f"{'R_near':^{w_r}}"
        f"{'err':^{w_err}}"
        f"{'R_away':^{w_r}}"
        f"{'err':^{w_err}}"
    )
    print(header)
    print("─" * W)

    for i in range(n):
        rq, rv = res_q[i], res_v[i]
        alo, ahi = rv["assoc_pt_range"]
        Rn, Ra, sn, sa = compute_variant_ratio(rq, rv)
        row = (
            f"{f'[{alo:>2.0f}, {ahi:>3.0f}]':^{w_pt}}"
            f"{Rn:>{w_r}.4f}"
            f"{sn:>{w_err}.4f}"
            f"{Ra:>{w_r}.4f}"
            f"{sa:>{w_err}.4f}"
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


def save_ratio_csv(res_q, res_v, trig_pt_range, out_dir):
    """Write the quenched/vacuum ratio (I_AA-like), matched-block jackknife, to CSV."""
    tlo, thi = trig_pt_range
    save_path = os.path.join(
        out_dir, f"quenched_vacuum_ratio_trig{tlo:.0f}-{thi:.0f}.csv"
    )
    n = min(len(res_q), len(res_v))

    with open(save_path, "w") as f:
        f.write("# Quenched/vacuum yield ratio (I_AA-like) vs pT_assoc\n")
        f.write(f"# trigger pT : {tlo:.1f} \u2013 {thi:.1f} GeV/c\n")
        f.write("# Uncertainty: matched-block jackknife (see compute_variant_ratio)\n")
        f.write("#\n")
        f.write(
            "assoc_pt_lo,assoc_pt_hi,assoc_pt_center,R_near,err_near,R_away,err_away\n"
        )
        for i in range(n):
            rq, rv = res_q[i], res_v[i]
            alo, ahi = rv["assoc_pt_range"]
            Rn, Ra, sn, sa = compute_variant_ratio(rq, rv)
            f.write(f"{alo},{ahi},{(alo + ahi) / 2},{Rn},{sn},{Ra},{sa}\n")

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

    res_q_valid = [r for r in results["quenched"] if r is not None]
    res_v_valid = [r for r in results["vacuum"] if r is not None]
    print_ratio_table(res_q_valid, res_v_valid)

    out_dir = "plots/yields"
    os.makedirs(out_dir, exist_ok=True)

    print("Saving integrated yields to CSV ...")
    for variant in VARIANTS:
        save_yield_csv(results[variant], trig_range, out_dir, variant=variant)
    save_ratio_csv(res_q_valid, res_v_valid, trig_range, out_dir)

    print("Generating Delta-phi slice plot (quenched + vacuum overlaid) ...")
    plot_dphi_slices(
        results,
        trig_pt_range=trig_range,
        cms_dir="datathief",
        save_path=f"{out_dir}/dphi_slices_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.png",
    )

    print("Generating integrated yield + ratio plot ...")
    plot_integrated_yields(
        results,
        trig_pt_range=trig_range,
        cms_dir="datathief",
        save_path=f"{out_dir}/integrated_yields_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.png",
    )


if __name__ == "__main__":
    main()
