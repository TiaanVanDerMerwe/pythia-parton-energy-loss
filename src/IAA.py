"""
I_AA — Quenched/Vacuum Yield Ratio
===================================

The medium-modification factor Y_quenched / Y_vacuum (I_AA-like) for the
per-trigger integrated away-side dihadron yield, as a function of pT_assoc.
Near side is not computed — only the away side is of interest here.

Multiple trigger-pT ranges can be run in one go (TRIG_PT_RANGES below); the
resulting ratio panels are all drawn on a single figure, laid out at most
two panels per row.

This used to be a panel inside assocYieldSpectrum.py's integrated-yield
figure; it now lives on its own here. Loading the pTHat bins, slicing by
pT_assoc, and running the block-jackknife projection is identical machinery
to assocYieldSpectrum.py, so this script imports compute_slice / run_variant
/ integrate_yield / integrate_near_away from there rather than duplicating
them — but it does NOT import that module's analysis constants (TRIG_PT_MIN,
ASSOC_PT_EDGES, ETA_CUT, ...). Every constant below is this script's own, so
changing assocYieldSpectrum.py's configuration never silently changes this
script's, and vice versa.

Ratio uncertainty
------------------
The quenched and vacuum files share the same seed per pTHat bin (matched
events, quenching hook toggled on/off), so jackknife block k corresponds to
the same underlying event subset in both variants. The ratio is therefore
built block-by-block *before* taking the jackknife variance:

    R_k    = Y_k^quenched / Y_k^vacuum
    Var_JK = (N-1)/N * sum_k (R_k - mean(R_k))^2

This propagates the correlation between the matched samples correctly.
Independent quadrature (sigma_R/R = sqrt((sigma_q/Y_q)^2 + (sigma_v/Y_v)^2))
would overestimate the uncertainty for matched events and is NOT used here.

Usage:
    python IAA.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from assocYieldSpectrum import integrate_near_away, run_variant

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer-Modern-style math, matches LaTeX default
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis configuration — independent of assocYieldSpectrum.py's constants
# ─────────────────────────────────────────────────────────────────────────────

# One (lo, hi) pair per trigger-pT range to run. Each gets its own panel on
# the combined ratio figure (at most 2 panels per row).
TRIG_PT_RANGES = [
    (19.2, 24.0),
    (24.0, 28.8),
    (28.8, 35.2),
    (35.2, 48.0),  # GeV/c
]

ASSOC_PT_EDGES = [0.5, 1, 2, 3, 4, 6, 8, 12, 14]  # GeV/c

NEAR_SIDE_LIMIT = 1.0  # |Dphi| <= this is near side; only used to exclude it

ETA_CUT = 1
ETA_BINS = 15
PHI_BINS = 33
PHI_RANGE = (-np.pi, np.pi)

SQRT_S_TEV = 2.76  # collision energy, annotated on the plots

ZYAM_PHI_LO = 0.5  # ZYAM search window (rad, in folded [0,pi] coords)
ZYAM_PHI_HI = 1.5

N_JACKKNIFE_BLOCKS = 5

POW = 3

VARIANTS = ("quenched", "vacuum")

RATIO_COLOR = "#4453FF"
CMS_COLOR = "black"

MAX_PANELS_PER_ROW = 2

CMS_DATA_DIR = "data"  # holds CMS_iaa_<trig_lo>-<trig_hi>.csv (rounded ints)


# ─────────────────────────────────────────────────────────────────────────────
# Quenched/vacuum ratio
# ─────────────────────────────────────────────────────────────────────────────


def compute_variant_ratio(res_q, res_v, near_limit=NEAR_SIDE_LIMIT):
    """
    I_AA-like away-side ratio Y_quenched / Y_vacuum for one pT_assoc slice,
    with matched-block jackknife uncertainty (see module docstring).

    Parameters
    ----------
    res_q, res_v : dict  — compute_slice() results for the same pT_assoc
                   slice, quenched and vacuum respectively.

    Returns
    -------
    R_away, sigma_away : float x 2
    """
    phi_c = res_q["phi_centers"]
    bw = phi_c[1] - phi_c[0]
    dpt_q = res_q["assoc_pt_range"][1] - res_q["assoc_pt_range"][0]
    dpt_v = res_v["assoc_pt_range"][1] - res_v["assoc_pt_range"][0]

    near_mask = phi_c <= near_limit
    away_mask = phi_c > near_limit

    _, Ya_q = integrate_near_away(
        res_q["phi_proj"], phi_c, near_mask, away_mask, bw, dpt_q
    )
    _, Ya_v = integrate_near_away(
        res_v["phi_proj"], phi_c, near_mask, away_mask, bw, dpt_v
    )

    R_away = Ya_q / Ya_v if Ya_v != 0 else np.nan

    jk_q, jk_v = res_q["jackknife_projs"], res_v["jackknife_projs"]
    n_blocks = min(len(jk_q), len(jk_v))
    if len(jk_q) != len(jk_v):
        print(
            "  WARNING: quenched/vacuum jackknife block counts differ "
            f"({len(jk_q)} vs {len(jk_v)}) — truncating to {n_blocks} "
            "for the matched-block ratio."
        )

    r_away_k = []
    for k in range(n_blocks):
        _, ya_q = integrate_near_away(jk_q[k], phi_c, near_mask, away_mask, bw, dpt_q)
        _, ya_v = integrate_near_away(jk_v[k], phi_c, near_mask, away_mask, bw, dpt_v)
        r_away_k.append(ya_q / ya_v if ya_v != 0 else np.nan)
    r_away_k = np.array(r_away_k)

    factor = (n_blocks - 1) / n_blocks
    sigma_away = float(
        np.sqrt(factor * np.nansum((r_away_k - np.nanmean(r_away_k)) ** 2))
    )

    return R_away, sigma_away


def compute_ratio_series(res_q, res_v):
    """
    Compute the away-side quenched/vacuum ratio for every pT_assoc slice.

    Returns a dict of parallel arrays: pt_lo, pt_hi, pt_center, ratio, err.
    """
    n = min(len(res_q), len(res_v))
    pt_lo, pt_hi, pt_center, ratio, err = [], [], [], [], []
    for i in range(n):
        rq, rv = res_q[i], res_v[i]
        alo, ahi = rv["assoc_pt_range"]
        Ra, sa = compute_variant_ratio(rq, rv)
        pt_lo.append(alo)
        pt_hi.append(ahi)
        pt_center.append((alo + ahi) / 2)
        ratio.append(Ra)
        err.append(sa)

    return {
        "pt_lo": np.array(pt_lo),
        "pt_hi": np.array(pt_hi),
        "pt_center": np.array(pt_center),
        "ratio": np.array(ratio),
        "err": np.array(err),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def print_ratio_table(series, trig_pt_range=None):
    """Print the quenched/vacuum away-side ratio (I_AA-like) summary table."""
    W = 44
    w_pt, w_r, w_err = 20, 14, 12

    title = "QUENCHED / VACUUM AWAY-SIDE RATIO"
    if trig_pt_range is not None:
        tlo, thi = trig_pt_range
        title += f"  (trig pT {tlo:.1f}-{thi:.1f} GeV/c)"
    print(f"\n{title}")
    print("─" * W)
    header = f"{'pT_assoc [GeV/c]':^{w_pt}}{'R_away':^{w_r}}{'err':^{w_err}}"
    print(header)
    print("─" * W)

    for alo, ahi, Ra, sa in zip(
        series["pt_lo"], series["pt_hi"], series["ratio"], series["err"]
    ):
        row = f"{f'[{alo:>2.0f}, {ahi:>3.0f}]':^{w_pt}}{Ra:>{w_r}.4f}{sa:>{w_err}.4f}"
        print(row)

    print("─" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export / import
# ─────────────────────────────────────────────────────────────────────────────


def _ratio_csv_path(trig_pt_range, csv_dir):
    tlo, thi = trig_pt_range
    return os.path.join(
        csv_dir, f"quenched_vacuum_ratio_away_trig{tlo:.0f}-{thi:.0f}.csv"
    )


def save_ratio_csv(series, trig_pt_range, csv_dir):
    """Write the quenched/vacuum away-side ratio (I_AA-like), matched-block jackknife, to CSV."""
    tlo, thi = trig_pt_range
    save_path = _ratio_csv_path(trig_pt_range, csv_dir)

    with open(save_path, "w") as f:
        f.write("# Quenched/vacuum away-side yield ratio (I_AA-like) vs pT_assoc\n")
        f.write(f"# trigger pT : {tlo:.1f} – {thi:.1f} GeV/c\n")
        f.write("# Uncertainty: matched-block jackknife (see compute_variant_ratio)\n")
        f.write("#\n")
        f.write("assoc_pt_lo,assoc_pt_hi,assoc_pt_center,R_away,err_away\n")
        f.writelines(
            f"{alo},{ahi},{pt_c},{Ra},{sa}\n"
            for alo, ahi, pt_c, Ra, sa in zip(
                series["pt_lo"],
                series["pt_hi"],
                series["pt_center"],
                series["ratio"],
                series["err"],
            )
        )

    print(f"  Saved: {save_path}")


def load_ratio_csv(trig_pt_range, csv_dir):
    """
    Load a previously saved quenched/vacuum away-side ratio CSV for one
    trigger-pT range, if it exists.

    Returns a series dict (see compute_ratio_series), or None if the CSV
    has not been generated yet.
    """
    load_path = _ratio_csv_path(trig_pt_range, csv_dir)
    if not os.path.exists(load_path):
        return None

    data = np.genfromtxt(load_path, delimiter=",", skip_header=5)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return {
        "pt_lo": data[:, 0],
        "pt_hi": data[:, 1],
        "pt_center": data[:, 2],
        "ratio": data[:, 3],
        "err": data[:, 4],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CMS experimental I_AA data
# ─────────────────────────────────────────────────────────────────────────────


def load_cms_iaa(trig_pt_range, data_dir=CMS_DATA_DIR):
    """
    Load experimental CMS I_AA data for one trigger-pT range from
    CMS_iaa_<trig_lo>-<trig_hi>.csv (trig_lo/trig_hi rounded to the
    nearest integer), with columns "pT,Iaa".

    Returns (pt, iaa) arrays, or (None, None) if no matching file exists.
    """
    tlo, thi = trig_pt_range
    fname = f"CMS_iaa_{round(tlo)}-{round(thi)}.csv"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print(f"  [warn] No CMS I_AA data found: {fpath}")
        return None, None

    data = np.genfromtxt(fpath, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


Y_TICKS_FULL = [0, 1, 2, 3, 4, 5]
Y_TICKS_TRIMMED = [0, 1, 2, 3, 4]  # drop the topmost tick (5)

X_TICKS_FULL = [0, 2, 4, 6, 8, 10, 12, 14]
X_TICKS_TRIMMED = [2, 4, 6, 8, 10, 12, 14]  # drop the leftmost tick (0)


def _draw_ratio_panel(
    ax,
    trig_pt_range,
    pt_centers,
    xerr,
    ratio_away,
    ratio_color,
    show_xlabel,
    show_ylabel,
    is_top_row,
    is_first_col,
    extra_lines=(),
    cms_pt=None,
    cms_iaa=None,
    cms_xerr=None,
    cms_color=CMS_COLOR,
    show_legend=False,
):
    """Draw one away-side ratio panel into *ax*."""
    ax.errorbar(
        pt_centers,
        ratio_away,
        xerr=xerr,
        fmt="o",
        color=ratio_color,
        ms=10,
        markeredgewidth=0.9,
        ecolor=ratio_color,
        elinewidth=2.0,
        capsize=5,
        capthick=2.0,
        zorder=5,
        label="PYTHIA",
    )
    if cms_pt is not None:
        ax.errorbar(
            cms_pt,
            cms_iaa,
            xerr=cms_xerr,
            fmt="o",
            color=cms_color,
            ms=10,
            markerfacecolor=cms_color,
            markeredgewidth=1.0,
            ecolor=cms_color,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            zorder=6,
            label="CMS Data",
        )
    ax.axhline(1, lw=2, ls="--", color="black", alpha=0.35)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_ylim(0, 5)
    ax.set_yticks(Y_TICKS_FULL if is_top_row else Y_TICKS_TRIMMED)
    ax.set_xticks(X_TICKS_FULL if is_first_col else X_TICKS_TRIMMED)

    if show_legend:
        ax.legend(fontsize=20, frameon=False, loc="center right")

    # trigger pT range (+ optional extra lines) printed inside the panel
    # (right side), instead of a title
    tlo, thi = trig_pt_range
    lines = [rf"{tlo:.1f} < $p_T^{{\rm trig}}$ < {thi:.1f} GeV/c", *extra_lines]
    ax.text(
        0.95,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        zorder=10,
    )

    # ticks point inward so they aren't clipped by the touching neighbor
    ax.tick_params(axis="both", direction="in", top=True, right=True)

    if show_ylabel:
        ax.set_ylabel(r"$I_{AA}^{\mathrm{away}}$", fontsize=20)
        ax.tick_params(axis="y", labelsize=20, labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if show_xlabel:
        ax.set_xlabel(r"$p_T^{\rm assoc}$ [GeV/c]", fontsize=20)
        ax.tick_params(axis="x", labelsize=22, labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_iaa_ratio_grid(
    all_results,
    ratio_color=RATIO_COLOR,
    max_per_row=MAX_PANELS_PER_ROW,
    save_path=None,
):
    """
    Plot the away-side quenched/vacuum ratio for one or more trigger-pT
    ranges on a single figure, one panel per range, at most `max_per_row`
    panels per row. Panels touch (no gaps): x ticks/labels only appear on
    the bottom row, y ticks/labels only on the left column, and the
    trigger pT range for each panel is printed inside it.

    Parameters
    ----------
    all_results : list of (trig_pt_range, series) tuples — one per trigger
                  pT range, where series is a dict as returned by
                  compute_ratio_series() / load_ratio_csv().
    """
    n_panels = len(all_results)
    ncols = min(max_per_row, n_panels)
    nrows = -(-n_panels // ncols)  # ceil division

    sqrt_s_line = rf"$\sqrt{{s}} = {SQRT_S_TEV:.2f}\ \mathrm{{TeV}}$"
    eta_cut_line = rf"$|\Delta\eta| < {ETA_CUT}$"

    # One panel: both annotations stack under its trigger-range text.
    # Two or more panels: sqrt(s) goes under panel 0, eta cut under panel 1
    # (each printed once, not repeated on every panel).
    if n_panels == 1:
        panel_extra_lines = {0: [sqrt_s_line, eta_cut_line]}
    else:
        panel_extra_lines = {0: [sqrt_s_line], 1: [eta_cut_line]}

    fig = plt.figure(figsize=(5.5 * ncols, 4.2 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.0, wspace=0.0)

    axes = np.full((nrows, ncols), None, dtype=object)

    for idx, (trig_pt_range, series) in enumerate(all_results):
        grow, gcol = divmod(idx, ncols)

        sharex = axes[0, gcol] if grow > 0 else None
        sharey = axes[grow, 0] if gcol > 0 else None
        ax = fig.add_subplot(gs[grow, gcol], sharex=sharex, sharey=sharey)
        axes[grow, gcol] = ax

        pt_centers = series["pt_center"]
        xerr_lo = pt_centers - series["pt_lo"]
        xerr_hi = series["pt_hi"] - pt_centers
        ratio_away = series["ratio"]

        cms_pt, cms_iaa = load_cms_iaa(trig_pt_range)
        cms_xerr = None
        if cms_pt is not None:
            if len(cms_pt) == len(pt_centers):
                cms_xerr = [xerr_lo, xerr_hi]
            else:
                print(
                    f"  [warn] CMS I_AA point count ({len(cms_pt)}) does not match "
                    f"pT_assoc bin count ({len(pt_centers)}) for trig "
                    f"{trig_pt_range} — plotting without x error bars."
                )

        _draw_ratio_panel(
            ax,
            trig_pt_range,
            pt_centers,
            [xerr_lo, xerr_hi],
            ratio_away,
            ratio_color,
            show_xlabel=(grow == nrows - 1),
            show_ylabel=(gcol == 0),
            is_top_row=(grow == 0),
            is_first_col=(gcol == 0),
            extra_lines=panel_extra_lines.get(idx, []),
            cms_pt=cms_pt,
            cms_iaa=cms_iaa,
            cms_xerr=cms_xerr,
            show_legend=(idx == 0),
        )

    # Hide any unused cells in the last row
    for idx in range(n_panels, nrows * ncols):
        grow, gcol = divmod(idx, ncols)
        fig.add_subplot(gs[grow, gcol]).set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    zyam_range = {"phi": (ZYAM_PHI_LO, ZYAM_PHI_HI)}

    plot_dir = "outputs/plots/iaa"
    csv_dir = "outputs/csv/iaa"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    all_results = []
    for trig_range in TRIG_PT_RANGES:
        tlo, thi = trig_range

        cached = load_ratio_csv(trig_range, csv_dir)
        if cached is not None:
            print(
                f"\nFound cached ratio CSV for trig pT {tlo:.1f}-{thi:.1f} "
                "— skipping analysis."
            )
            print_ratio_table(cached, trig_pt_range=trig_range)
            all_results.append((trig_range, cached))
            continue

        base = f"pythiaData/2760/cms/{tlo}-{thi}"

        results = {}
        for variant in VARIANTS:
            print(
                f"\n{'=' * 80}\nRunning '{variant}' pass "
                f"(trig pT {tlo:.1f}-{thi:.1f})\n{'=' * 80}"
            )
            results[variant] = run_variant(
                variant,
                base,
                POW,
                trig_pt_range=trig_range,
                assoc_pt_edges=ASSOC_PT_EDGES,
                eta_cut=ETA_CUT,
                eta_bins=ETA_BINS,
                phi_bins=PHI_BINS,
                phi_range=PHI_RANGE,
                zyam_range=zyam_range,
                n_blocks=N_JACKKNIFE_BLOCKS,
            )

        res_q = [r for r in results["quenched"] if r is not None]
        res_v = [r for r in results["vacuum"] if r is not None]

        series = compute_ratio_series(res_q, res_v)
        print_ratio_table(series, trig_pt_range=trig_range)
        save_ratio_csv(series, trig_range, csv_dir)

        all_results.append((trig_range, series))

    plot_iaa_ratio_grid(
        all_results,
        save_path=f"{plot_dir}/iaa_ratio_away_side.svg",
    )
    plt.show()


if __name__ == "__main__":
    main()
