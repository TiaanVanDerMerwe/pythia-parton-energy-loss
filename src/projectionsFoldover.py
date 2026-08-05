"""
Dihadron Correlation Analysis

Computes the dihadron correlation function:
    1/Ntrig * dNpair/deta*dphi

The idea is simple: for each "trigger" hadron (high-pT), count how many
"associated" hadrons (lower pT) show up nearby in angle space. The result
tells you about jet structure and back-to-back correlations.

When combining multiple pTHat bins, each bin comes with its own sigmaGen
from Pythia. We rescale every pair's weight by sigmaGen/weightSum before
merging, so the final histogram reflects the physically correct
cross-section-weighted distribution rather than just counting events.

Uncertainties on the Δφ projection are estimated via block jackknife:
events are partitioned into N_JACKKNIFE_BLOCKS groups; each sample
leaves one group out, recomputes the full (ZYAM-subtracted) projection,
and the standard jackknife formula gives the standard error per bin.

This version runs the full pipeline once per simulation "variant"
(e.g. 'quenched' and 'vacuum'), each variant being fed only its own
matching input files, and produces its own independent figure window
and its own set of saved outputs.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
import pandas as pd
import glob
import os

# ──────────────────────────────────────────────────────────────────────────────
# Analysis cuts — adjust these to match whatever kinematic window you want
# ──────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN  =  19.2  # GeV/c  (lower edge of trigger pT window)
TRIG_PT_MAX  =  24.0  # GeV/c  (upper edge of trigger pT window)
ASSOC_PT_MIN =  0.5     # GeV/c  (associated particle pT range)
ASSOC_PT_MAX =  1.0     # GeV/c
ETA_CUT      =  1.0     # only keep pairs with |Δη| < 1

# Number of blocks used for the block-jackknife uncertainty estimate.
# More blocks → finer resolution but more CPU time.  50 is a good default.
N_JACKKNIFE_BLOCK = 50
N_JACKKNIFE_BLOCKS = [N_JACKKNIFE_BLOCK]

# Which simulation variants to run, and how each maps onto filenames.
# Each entry's filename filter is applied as a substring match against the
# pTHat bin files found under BASE, so filenames like
#   dihadron_pow5_pT18to600_quenched_seed100.csv
#   dihadron_pow5_pT18to600_vacuum_seed100.csv
# are routed to the correct variant automatically.
VARIANTS = ['quenched', 'vacuum']

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_dihadron_data(filename):
    """
    Read one dihadron CSV file. The file has a comment-header block
    (lines starting with #) that carries run metadata, followed by
    normal CSV rows.

    We parse the header first, then let pandas handle the data rows.
    Numeric values in the header come back as floats; range strings like
    "1.9e+01 - 2.4e+01" become a [lo, hi] list; anything else stays as a
    plain string.

    Parameters
    ----------
    filename : str
        Path to the data file.

    Returns
    -------
    data : pandas.DataFrame
    metadata : dict
    """
    metadata = {}

    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                continue
            if ':' not in line:
                continue

            key, value = line[1:].split(':', 1)
            key   = key.strip()
            value = value.strip()

            try:
                metadata[key] = float(value)
                continue
            except ValueError:
                pass

            if ' - ' in value:
                try:
                    lo, hi = value.split(' - ', 1)
                    metadata[key] = [float(lo), float(hi)]
                    continue
                except ValueError:
                    pass

            metadata[key] = value

    data = pd.read_csv(filename, comment='#')
    return data, metadata


def read_and_combine_bins(filenames, trig_pt_min=TRIG_PT_MIN, trig_pt_max=TRIG_PT_MAX,
                          assoc_pt_min=ASSOC_PT_MIN, assoc_pt_max=ASSOC_PT_MAX,
                          deta_max=ETA_CUT):
    """
    Load every pTHat bin, rescale weights to physical units, and stack
    them into one big DataFrame.

    Pythia gives each event a weight relative to its own run. To compare
    across bins with different cross-sections we multiply by:

        scale = sigmaGen_bin / weightSum_bin

    Parameters
    ----------
    filenames : list of str

    Returns
    -------
    combined_data : pandas.DataFrame
    combined_metadata : dict
    """
    frames      = []
    bin_info    = []
    total_ntrig  = 0.0

    if not filenames:
        raise ValueError("No input files provided.")

    for idx, fname in enumerate(filenames):
        print(f"  Loading bin {idx}: {os.path.basename(fname)}")
        data, meta = read_dihadron_data(fname)

        trig_mask  = (data['trigger_pT'] > trig_pt_min) & (data['trigger_pT'] <= trig_pt_max)
        assoc_mask = (data['assoc_pT']   > assoc_pt_min) & (data['assoc_pT']  <= assoc_pt_max)
        deta_mask  = (data['assoc_eta'] - data['trigger_eta']).abs() < deta_max

        data = data[trig_mask & assoc_mask & deta_mask].copy()

        sigma           = meta.get('sigmaGEN', None)
        weightSum       = meta.get('weightSum', None)
        trig_weight_sum = meta.get(
            f"triggerWeightSum_{trig_pt_min}to{trig_pt_max}", None
        )

        if (sigma is None or weightSum is None or weightSum == 0
                or trig_weight_sum is None or trig_weight_sum == 0):
            print(
                f"File {fname} is missing 'sigmaGEN' or 'weightSum' or "
                f"weightSum = 0 or 'trig_weight_sum' or trig_weight_sum = 0. "
                f"Cannot rescale weights."
            )
            continue

        scale = sigma / weightSum
        trig_weight_sum_bin = trig_weight_sum * scale
        print(f"    sigmaGEN={sigma:.6e}  weightSum={weightSum:.6e}  "
              f"scale={scale:.6e}  pairs={len(data)}")

        data['weight'] = data['weight'] * scale
        data['bin_index'] = idx

        frames.append(data)
        bin_info.append({
            'bin_index':        idx,
            'filename':         fname,
            'sigmaGEN':         sigma,
            'weightSum':        weightSum,
            'triggerWeightSum': trig_weight_sum,
            'scale':            scale,
            'n_pairs':          len(data),
            'pthat_range':      meta.get('PTHAT_RANGE', 'unknown'),
        })

        total_ntrig  += trig_weight_sum_bin

        combined_metadata = {}
        combined_metadata['TRIG_PT_RANGE']   = meta.get('TRIG_PT_RANGE')
        combined_metadata['TRIG_ETA_RANGE']  = meta.get('TRIG_ETA_RANGE')
        combined_metadata['ASSOC_PT_RANGE']  = meta.get('ASSOC_PT_RANGE')
        combined_metadata['ASSOC_ETA_RANGE'] = meta.get('ASSOC_ETA_RANGE')

    combined_data = pd.concat(frames, ignore_index=True)

    combined_metadata['CUT_TRIG_PT_RANGE']  = [trig_pt_min,  trig_pt_max]
    combined_metadata['CUT_ASSOC_PT_RANGE'] = [assoc_pt_min, assoc_pt_max]
    combined_metadata['N_TRIG']           = total_ntrig
    combined_metadata['N_BINS']          = len(filenames)
    combined_metadata['BIN_INFO']        = bin_info

    print(f"\n  Combined: {len(filenames)} bins, {len(combined_data):,} total pairs")
    return combined_data, combined_metadata

# ──────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────
def calculate_delta_phi(phi1, phi2):
    """
    Return phi2 - phi1 wrapped into [-π, π] using arctan2.
    """
    dphi = phi2 - phi1
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def _fold_phi(phi_projection, phi_centers):
    """
    Fold a 1D Δφ distribution (defined over the full [-π, π] or
    [-π/2, 3π/2] range) into [0, π] following the CMS convention.

    The bin at Δφ = 0 is unmodified; all other bins are averaged with
    their mirror image.

    Parameters
    ----------
    phi_projection : 1D array, length n
    phi_centers    : 1D array, length n  (must be symmetric around 0)

    Returns
    -------
    phi_proj_folded  : 1D array, length n//2 + 1
    phi_centers_folded : 1D array, same length
    """
    n = len(phi_projection)
    h = n // 2

    phi_proj_folded = np.empty(h + 1)
    phi_proj_folded[0] = 1 * phi_projection[h]
    phi_proj_folded[1:] = 0.5 * (
        phi_projection[h + 1:] +
        phi_projection[:h][::-1]
    )
    phi_centers_folded = phi_centers[h:]
    return phi_proj_folded, phi_centers_folded

def _find_zyam_level(phi_proj_folded, phi_centers_folded, zyam_range):
    """
    Find the ZYAM background level from an already-folded 1D Δφ projection.

    Scans the folded distribution within ``zyam_range['phi']`` and returns
    the minimum value.  Separated from subtraction so the level can always
    be computed for display regardless of whether the caller subtracts it.

    Parameters
    ----------
    phi_proj_folded    : 1D array  — projection in [0, π]
    phi_centers_folded : 1D array  — bin centres in [0, π]
    zyam_range         : dict      — {'phi': (lo, hi)} in [0, π] coordinates

    Returns
    -------
    background_level : float
    """
    phi_mask = ((phi_centers_folded >= zyam_range['phi'][0]) &
                (phi_centers_folded <= zyam_range['phi'][1]))
    masked = np.where(phi_mask, phi_proj_folded, np.inf)
    return float(np.min(masked))


def _apply_zyam_to_folded(phi_proj_folded, phi_centers_folded, zyam_range):
    """
    Apply ZYAM subtraction to an already-folded 1D Δφ projection.

    Finds the minimum of ``phi_proj_folded`` within ``zyam_range['phi']``
    (coordinates in [0, π]) and returns the subtracted projection together
    with the background level.

    Parameters
    ----------
    phi_proj_folded    : 1D array  — projection in [0, π]
    phi_centers_folded : 1D array  — bin centres in [0, π]
    zyam_range         : dict      — {'phi': (lo, hi)} in [0, π] coordinates

    Returns
    -------
    phi_proj_zyam    : 1D array
    background_level : float
    """
    background_level = _find_zyam_level(phi_proj_folded, phi_centers_folded, zyam_range)
    return phi_proj_folded - background_level, background_level


def compute_dihadron_correlation(data, metadata,
                                 eta_bins=33, phi_bins=33,
                                 eta_range=(-1, 1),
                                 phi_range=(-np.pi/2, 3*np.pi/2),
                                 zyam=True,
                                 zyam_range={'phi': (0.5, 1.5)},
                                 ntrig_override=None):
    """
    Build the per-trigger-normalised 2D correlation:

        C(Δη, Δφ) = 1/Ntrig × dNpair / (dΔη dΔφ)

    ZYAM handling
    -------------
    The background level is **always** computed via the correct sequence:

        1. Project the 2D surface onto Δφ (average over Δη).
        2. Fold the 1D projection into [0, π].
        3. Find the minimum of the *folded* distribution inside
           ``zyam_range`` → this is ``background_level``.

    Whether that background is actually *subtracted* depends on the
    ``zyam`` flag:

        zyam=True  → subtract background_level from the correlation surface.
        zyam=False → return the raw surface unchanged; background_level is
                     still computed and returned so callers can display the
                     ZYAM line without applying the subtraction.

    Parameters
    ----------
    data            : pandas.DataFrame
    metadata        : dict
    eta_bins        : int
    phi_bins        : int
    eta_range       : tuple
    phi_range       : tuple
    zyam            : bool
        If True, subtract the ZYAM background from the correlation surface.
        If False, compute but do not subtract the background level.
    zyam_range      : dict  {'phi': (lo, hi)} or None → defaults to [0, π]
    ntrig_override  : float or None

    Returns
    -------
    correlation      : 2D array
    eta_centers      : 1D array
    phi_centers      : 1D array
    ntrig            : float
    background_level : float  — always set, even when zyam=False
    raw_counts       : 2D array
    weighted_counts  : 2D array
    """

    deta    = data['assoc_eta'].values   - data['trigger_eta'].values
    dphi    = calculate_delta_phi(data['trigger_phi'].values,
                                  data['assoc_phi'].values)
    weights = data['weight'].values

    if ntrig_override is not None:
        ntrig = ntrig_override
    else:
        group_cols = ['event', 'trigger_id']
        if 'bin_index' in data.columns:
            group_cols = ['bin_index'] + group_cols
        trigger_data = data.groupby(group_cols).first()
        ntrig = trigger_data['weight'].sum()

    hist, eta_edges, phi_edges = np.histogram2d(
        deta, dphi,
        bins=[eta_bins, phi_bins],
        range=[eta_range, phi_range],
        weights=weights,
    )

    raw_hist, _, _ = np.histogram2d(
        deta, dphi,
        bins=[eta_bins, phi_bins],
        range=[eta_range, phi_range],
    )

    deta_bin = (eta_range[1] - eta_range[0]) / eta_bins
    dphi_bin = (phi_range[1] - phi_range[0]) / phi_bins
    bin_area = deta_bin * dphi_bin

    correlation = hist / (ntrig * bin_area)

    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Always compute the ZYAM level: project → fold → find minimum.
    # background_level is valid for display even when zyam=False.
    dphi_distribution              = np.mean(correlation, axis=0)
    phi_proj_folded, phi_centers_f = _fold_phi(dphi_distribution, phi_centers)
    background_level               = _find_zyam_level(phi_proj_folded, phi_centers_f,
                                                       zyam_range)
    if zyam:
        correlation = correlation - background_level

    return (correlation, eta_centers, phi_centers, ntrig,
            background_level, raw_hist, hist)


# ──────────────────────────────────────────────────────────────────────────────
# Jackknife uncertainty on the Δφ projection
# ──────────────────────────────────────────────────────────────────────────────
def _ntrig_removed_by_block(block_data, trig_cols):
    """
    Sum the rescaled weights of the unique triggers present in
    ``block_data``.
 
    We deduplicate on ``trig_cols`` (e.g. [bin_index, event, trigger_id])
    and sum the 'weight' column of the first occurrence of each trigger.
 
    This is the correct partial-Ntrig correction for a jackknife block:
    the pair CSV records only triggers that had at least one associate, so
    triggers with zero associates are absent from the DataFrame entirely
    and will never appear in any block.  The missing triggers therefore
    contribute the same (unknown) amount to every jackknife sample and
    cancel out of all differences — they do not need to be tracked.
 
    Parameters
    ----------
    block_data : pandas.DataFrame  — rows belonging to the removed block
    trig_cols  : list of str       — columns that uniquely identify a trigger
                                     within the full dataset
 
    Returns
    -------
    float  — sum of rescaled trigger weights for the removed block
    """
    return block_data.groupby(trig_cols)['weight'].first().sum()


def compute_phi_projection_jackknife(data, metadata,
                                     eta_bins=33, phi_bins=33,
                                     eta_range=(-1, 1),
                                     phi_range=(-np.pi, np.pi),
                                     zyam=True,
                                     zyam_range={'phi': (0, np.pi)},
                                     ntrig_override=None,
                                     n_blocks=N_JACKKNIFE_BLOCKS):
    """
    Estimate statistical uncertainties on the folded Δφ projection using
    a block-jackknife resampling scheme.
 
    Events are partitioned into ``n_blocks`` contiguous blocks.  For each
    block k we:
 
        1.  Remove all pairs that belong to block k from the DataFrame.
        2.  Subtract the trigger weights of the removed block from
            ntrig_override to obtain ntrig_k (see note below).
        3.  Recompute the 2D correlation with ntrig_k (no ZYAM yet).
        4.  Project onto Δφ and fold into [0, π].
        5.  Apply ZYAM to the *folded* projection if zyam=True (find
            minimum, subtract).  ZYAM uncertainty is therefore folded
            into the error bars automatically.
            If zyam=False, the ZYAM level is still computed and returned
            for display purposes, but no subtraction is performed.
 
    The standard jackknife variance formula for N blocks is:
 
        Var_JK[θ] = (N-1)/N × Σ_k (θ_k − θ̄_JK)²
 
    where θ_k is the estimate from the k-th leave-one-out sample and
    θ̄_JK is the mean of all jackknife estimates.
 
    Note on Ntrig
    -------------
    The pair CSV only records triggers that have at least one associated
    particle; triggers whose associate search came up empty are absent from
    the file entirely.  We therefore cannot reconstruct the full Ntrig from
    the pair DataFrame from scratch — the authoritative value is the
    triggerWeightSum accumulated by read_and_combine_bins and stored in
    ntrig_override.
 
    However, Ntrig *should* change between jackknife samples: removing
    block k removes real triggers from the denominator.  We handle this
    correctly by subtracting only the trigger weights that ARE visible in
    the removed block's pairs.  The triggers that were never written to
    file (zero associates) are absent from every block equally, so they
    cancel out of all jackknife differences and do not bias the variance
    estimate.
 
    Parameters
    ----------
    data            : pandas.DataFrame  (must contain a 'weight' column)
    metadata        : dict
    eta_bins        : int
    phi_bins        : int
    eta_range       : tuple
    phi_range       : tuple   — should be symmetric around 0 for folding
    zyam            : bool
        If True, subtract the ZYAM level from each jackknife sample's
        folded projection.  If False, skip subtraction but still compute
        and return the background level for display.
    zyam_range      : dict or None
        Search window for the ZYAM minimum, in *folded* [0, π] coordinates.
        Defaults to {'phi': (0, np.pi)} (entire folded range).
    ntrig_override  : float
        Effective trigger count from the combined metadata.  Must be
        provided — pass metadata['NTRIG'] from read_and_combine_bins.
    n_blocks        : int
        Number of jackknife blocks.  More blocks → higher resolution but
        O(n_blocks) calls to compute_dihadron_correlation.
 
    Returns
    -------
    phi_proj_central   : 1D array  — folded projection (ZYAM-subtracted if zyam=True)
    phi_proj_err       : 1D array  — jackknife standard errors, same length
    phi_centers_folded : 1D array
    ntrig              : float
    background_level   : float     — ZYAM level from the full-sample computation,
                                     always set regardless of zyam flag
    """

    if ntrig_override is None:
        raise ValueError(
            "ntrig_override is required for the jackknife.  "
            "Pass metadata['NTRIG'] from read_and_combine_bins."
        )

    # columns that uniquely identify a single trigger particle
    trig_cols = (['bin_index', 'event', 'trigger_id']
                 if 'bin_index' in data.columns
                 else ['event', 'trigger_id'])

    # ── central value ────────────────────────────────────────────────────────
    # Always use zyam=False here; ZYAM is applied after projection + fold.
    corr_full, eta_centers, phi_centers, ntrig, _, _, _ = compute_dihadron_correlation(
        data, metadata,
        eta_bins=eta_bins, phi_bins=phi_bins,
        eta_range=eta_range, phi_range=phi_range,
        zyam=False,
        ntrig_override=ntrig_override,
    )

    eta_mask = np.abs(eta_centers) < ETA_CUT

    # project → fold
    phi_proj_full                      = np.mean(corr_full[eta_mask, :], axis=0)
    phi_proj_folded, phi_centers_folded = _fold_phi(phi_proj_full, phi_centers)

    # always compute the ZYAM level; subtract only if requested
    background_level = _find_zyam_level(phi_proj_folded, phi_centers_folded, zyam_range)
    phi_proj_central = (phi_proj_folded - background_level) if zyam else phi_proj_folded

    # ── assign block labels to every pair ───────────────────────────────────
    event_cols = ['bin_index', 'event'] if 'bin_index' in data.columns else ['event']
    unique_events = data[event_cols].drop_duplicates().reset_index(drop=True)
    n_events = len(unique_events)

    if n_blocks > n_events:
        print(f"  Warning: n_blocks ({n_blocks}) > n_events ({n_events}). "
              f"Falling back to leave-one-event-out jackknife.")
        n_blocks = n_events

    # round-robin assignment keeps the pT-hat composition of each block
    # roughly uniform, which stabilises the jackknife variance estimate
    unique_events = unique_events.copy()
    unique_events['_jk_block'] = np.arange(n_events) % n_blocks

    data_jk = data.merge(unique_events, on=event_cols, how='left')

    # ── jackknife loop ───────────────────────────────────────────────────────
    print(f"\nRunning block jackknife ({n_blocks} blocks) …")
    jackknife_projs = np.empty((n_blocks, len(phi_proj_central)))

    for k in range(n_blocks):
        removed  = data_jk[data_jk['_jk_block'] == k]
        data_k   = data_jk[data_jk['_jk_block'] != k].copy()

        ntrig_k = ntrig_override - _ntrig_removed_by_block(removed, trig_cols)

        corr_k, _, _, _, _, _, _ = compute_dihadron_correlation(
            data_k, metadata,
            eta_bins=eta_bins, phi_bins=phi_bins,
            eta_range=eta_range, phi_range=phi_range,
            zyam=False,
            ntrig_override=ntrig_k,
        )

        # project → fold → (conditionally) subtract
        phi_proj_k               = np.mean(corr_k[eta_mask, :], axis=0)
        phi_proj_k_folded, _     = _fold_phi(phi_proj_k, phi_centers)
        if zyam:
            phi_proj_k_final, _  = _apply_zyam_to_folded(
                phi_proj_k_folded, phi_centers_folded, zyam_range
            )
        else:
            phi_proj_k_final = phi_proj_k_folded

        jackknife_projs[k] = phi_proj_k_final

        if (k + 1) % max(1, n_blocks // 10) == 0:
            print(f"  Block {k+1}/{n_blocks} done")

    # ── standard jackknife variance ──────────────────────────────────────────
    # Var_JK = (N-1)/N * Σ_k (θ_k - θ̄_JK)²
    jk_mean      = np.mean(jackknife_projs, axis=0)
    jk_var       = ((n_blocks - 1) / n_blocks
                    * np.sum((jackknife_projs - jk_mean) ** 2, axis=0))
    phi_proj_err = np.sqrt(jk_var)

    print(f"  Jackknife done.  Max error: {phi_proj_err.max():.4e}  "
          f"Median error: {np.median(phi_proj_err):.4e}  "
          f"Mean error: {np.mean(phi_proj_err):.4e}  "
          f"First bin (Δφ≈0) error: {phi_proj_err[0]:.4e}  "
          f"Last bin (Δφ≈π) error: {phi_proj_err[-1]:.4e}")

    rel_err = phi_proj_err / np.abs(phi_proj_central)
    print(f"Max relative error: {rel_err.max():.2%}  "
          f"Median relative error: {np.median(rel_err):.2%}  "
          f"Mean relative error: {np.mean(rel_err):.2%}  "
          f"First bin (Δφ≈0) relative error: {rel_err[0]:.2%}  "
          f"Last bin (Δφ≈π) relative error: {rel_err[-1]:.2%}")

    delta_phi = phi_centers_folded[1] - phi_centers_folded[0]

    yield_k       = jackknife_projs.sum(axis=1) * delta_phi
    yield_central = phi_proj_central.sum() * delta_phi

    yield_jk_mean = yield_k.mean()
    yield_jk_var  = ((n_blocks - 1) / n_blocks
                     * np.sum((yield_k - yield_jk_mean) ** 2))
    yield_err = np.sqrt(yield_jk_var)

    print(f"\nIntegrated yield:  {yield_central:.4e} ± {yield_err:.4e}  "
          f"(rel. err: {yield_err / abs(yield_central):.2%})")

    return phi_proj_central, phi_proj_err, phi_centers_folded, ntrig, background_level, jackknife_projs


def run_jackknife_multiple_blocks(data, metadata,
                                 block_sizes,
                                 eta_bins=33, phi_bins=33,
                                 eta_range=(-1, 1),
                                 phi_range=(-np.pi, np.pi),
                                 zyam=True,
                                 zyam_range=None,
                                 ntrig_override=None):
    """
    Run the jackknife multiple times with different block sizes.

    Parameters
    ----------
    block_sizes : iterable of int
        List or array of block counts to try.

    Returns
    -------
    results : dict
        Keys = block size
        Values = dict with:
            'phi_proj_central'
            'phi_proj_err'
            'phi_centers'
    """
    results = {}

    for n_blocks in block_sizes:
        print(f"\n=== Running jackknife with {n_blocks} blocks ===")

        (phi_proj_central,
         phi_proj_err,
         phi_centers_folded,
         ntrig,
         bkg, jackknife_projs) = compute_phi_projection_jackknife(
            data, metadata,
            eta_bins=eta_bins,
            phi_bins=phi_bins,
            eta_range=eta_range,
            phi_range=phi_range,
            zyam=zyam,
            zyam_range=zyam_range,
            ntrig_override=ntrig_override,
            n_blocks=n_blocks,
        )

        results[n_blocks] = {
            'phi_proj_central': phi_proj_central,
            'phi_proj_err': phi_proj_err,
            'phi_centers': phi_centers_folded,
            'ntrig': ntrig,
            'background': bkg,
            'jackknife_projs': jackknife_projs
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────
def save_projections_to_csv(
    phi_proj_central, phi_proj_err, phi_centers_folded,
    correlation, eta_centers, phi_centers,
    background_level, metadata,
    save_dir='plots/Correlations',
    tag='',
):
    """
    Write the Δφ and Δη projections to separate CSV files.

    Δφ file columns
    ---------------
    delta_phi_rad       : folded bin centre in [0, π]  (radians)
    dNpair_dDeltaPhi    : 1/Ntrig × dNpair/dΔφ  (ZYAM-subtracted if applied)
    stat_err_jackknife  : jackknife standard error (0 if not computed)
    zyam_background     : constant ZYAM level subtracted from every bin

    Δη file columns
    ---------------
    delta_eta           : folded bin centre in [0, |Δη|_max]
    dNpair_dDeltaEta    : 1/Ntrig × dNpair/dΔη  (averaged over all Δφ)

    Parameters
    ----------
    phi_proj_central    : 1D array — folded Δφ projection
    phi_proj_err        : 1D array or None — jackknife errors on Δφ projection
    phi_centers_folded  : 1D array — bin centres for phi_proj_central
    correlation         : 2D array — C(Δη, Δφ) from compute_dihadron_correlation
    eta_centers         : 1D array — bin centres along Δη
    phi_centers         : 1D array — bin centres along Δφ (unfolded)
    background_level    : float   — ZYAM constant
    metadata            : dict
    save_dir            : str     — directory to write files into
    tag                 : str     — optional label appended to filenames

    Returns
    -------
    dphi_path : str — path of the written Δφ CSV
    deta_path : str — path of the written Δη CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    trig_lo, trig_hi   = metadata.get('CUT_TRIG_PT_RANGE',  [TRIG_PT_MIN,  TRIG_PT_MAX])
    assoc_lo, assoc_hi = metadata.get('CUT_ASSOC_PT_RANGE', [ASSOC_PT_MIN, ASSOC_PT_MAX])

    base = f'trig{trig_lo:.1f}-{trig_hi:.1f}_assoc{assoc_lo:.1f}-{assoc_hi:.1f}'
    if tag:
        base = f'{base}_{tag}'

    # ── Δφ projection ─────────────────────────────────────────────────────────
    dphi_df = pd.DataFrame({
        'delta_phi_rad':      phi_centers_folded,
        'dNpair_dDeltaPhi':   phi_proj_central,
        'stat_err_jackknife': (phi_proj_err
                               if phi_proj_err is not None
                               else np.zeros_like(phi_proj_central)),
        'zyam_background':    background_level,
    })

    dphi_path = os.path.join(save_dir, f'dphi_projection_{base}.csv')
    header_lines = [
        f'# Dihadron Δφ projection',
        f'# trigger pT : {trig_lo:.1f} – {trig_hi:.1f} GeV/c',
        f'# assoc   pT : {assoc_lo:.1f} – {assoc_hi:.1f} GeV/c',
        f'# |Δη| cut   : {ETA_CUT}',
        f'# ZYAM level : {background_level:.6e}',
        f'# ntrig      : {metadata.get("NTRIG", float("nan")):.6e}',
        f'# n_bins     : {metadata.get("n_bins", "?")}',
        f'# jackknife blocks : {metadata.get("jackknife_n_blocks", "not run")}',
        f'#',
        f'# delta_phi_rad   : folded bin centre in [0, π] (rad)',
        f'# dNpair_dDeltaPhi: 1/Ntrig × dNpair/dΔφ  (ZYAM-subtracted)',
        f'# stat_err_jackknife: jackknife standard error',
        f'# zyam_background : constant subtracted from every bin',
    ]

    with open(dphi_path, 'w') as f:
        f.write('\n'.join(header_lines) + '\n')
        dphi_df.to_csv(f, index=False)

    print(f"Δφ projection saved → {dphi_path}  ({len(dphi_df)} bins)")

    # ── Δη projection ─────────────────────────────────────────────────────────
    eta_proj_raw = np.mean(correlation, axis=1)           # average over all Δφ
    eta_proj_folded, eta_centers_folded = _fold_phi(eta_proj_raw, eta_centers)

    deta_df = pd.DataFrame({
        'delta_eta':          eta_centers_folded,
        'dNpair_dDeltaEta':   eta_proj_folded,
    })

    deta_path = os.path.join(save_dir, f'deta_projection_{base}.csv')
    deta_header = [
        f'# Dihadron Δη projection',
        f'# trigger pT : {trig_lo:.1f} – {trig_hi:.1f} GeV/c',
        f'# assoc   pT : {assoc_lo:.1f} – {assoc_hi:.1f} GeV/c',
        f'# averaged over all Δφ bins',
        f'# ZYAM subtraction applied to the underlying 2D correlation: yes',
        f'# ntrig      : {metadata.get("NTRIG", float("nan")):.6e}',
        f'#',
        f'# delta_eta        : folded bin centre in [0, |Δη|_max]',
        f'# dNpair_dDeltaEta : 1/Ntrig × dNpair/dΔη',
    ]

    with open(deta_path, 'w') as f:
        f.write('\n'.join(deta_header) + '\n')
        deta_df.to_csv(f, index=False)

    print(f"Δη projection saved → {deta_path}  ({len(deta_df)} bins)")

    return dphi_path, deta_path


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_with_projections(correlation, eta_centers, phi_centers,
                                      background_level=0.0,
                                      phi_proj_central=None,
                                      phi_proj_err=None,
                                      phi_centers_folded=None,
                                      metadata=None,
                                      save_path=None,
                                      fig_num=None):
    """
    Main physics figure: 3D correlation surface on the left, and on the
    right (top to bottom):
        1. Δφ projection with jackknife error bars, compared against CMS data
        2. CMS / Pythia ratio
        3. Δη projection

    The ZYAM minimum line is always drawn on both projection panels
    regardless of whether ZYAM subtraction was applied, so it remains
    visible during testing runs with zyam=False.

    Parameters
    ----------
    correlation          : 2D array  — C(Δη, Δφ) from compute_dihadron_correlation
    eta_centers          : 1D array
    phi_centers          : 1D array
    background_level     : float     — ZYAM constant; always drawn as a red
                                       dashed line even when zyam=False
    phi_proj_central     : 1D array or None
    phi_proj_err         : 1D array or None
    phi_centers_folded   : 1D array or None
    metadata             : dict or None
    save_path            : str or None
    fig_num              : str or int or None
        Passed straight through to plt.figure(num=...). Giving each
        variant a distinct fig_num guarantees each run opens its own
        separate figure window instead of being drawn into (or
        clobbering) a previously-open one.
    """

    fig = plt.figure(num=fig_num, figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig,
                   width_ratios=[3, 0.5, 0.5],
                   wspace=0.4, hspace=0.5)

    ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
    ax_phi = fig.add_subplot(gs[0, 1:])
    ax_rat = fig.add_subplot(gs[1, 1:])
    ax_eta = fig.add_subplot(gs[2, 1:])

    # ── colour palette ────────────────────────────────────────────────────────
    blue   = plt.get_cmap('Blues')(0.7)
    orange = plt.get_cmap('Oranges')(0.7)
    red    = plt.get_cmap('Reds')(0.7)
    green  = plt.get_cmap('Greens')(0.7)

    d_eta = eta_centers[1] - eta_centers[0] if len(eta_centers) > 1 else 0.1
    d_phi = phi_centers[1] - phi_centers[0] if len(phi_centers) > 1 else 0.1

    # ── 3D surface ────────────────────────────────────────────────────────────
    # remap φ display range from (-π, π) → (-π/2, 3π/2) without touching
    # phi_centers; roll Z to match the shifted coordinate array
    split = np.searchsorted(phi_centers, -np.pi / 2)   # first index >= -π/2
    phi_display = np.concatenate([
        phi_centers[split:],
        phi_centers[:split] + 2 * np.pi
    ])

    Z         = np.clip(correlation, 0, 0.4)
    Z_display = np.roll(Z, -split, axis=1)
    max_height = Z_display.max() if Z_display.max() > 0 else 1.0

    ETA, PHI = np.meshgrid(eta_centers, phi_display, indexing='ij')
    
    n_colors = 20   # however many discrete steps you want
    levels   = np.linspace(0, max_height, n_colors + 1)   # bin edges
    colourticks   = np.linspace(0, max_height, 5)   # bin edges

    cmap = plt.cm.get_cmap('turbo', n_colors)              # n discrete colours
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    colors = cmap(norm(Z_display))

    ax3d.plot_surface(ETA, PHI, Z_display,
                      facecolors=colors, shade=True, alpha=0.9,
                      linewidth=0, antialiased=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.15,
                 ticks=colourticks)
    cbar.ax.tick_params(labelsize=12)

    ax3d.set_zlim(0, max_height * 1.05)
    ax3d.set_xlabel(r'$\Delta\eta$',       fontsize=13, labelpad=10)
    ax3d.set_ylabel(r'$\Delta\phi$ [rad]', fontsize=13, labelpad=10)
    ax3d.zaxis.set_rotate_label(False)
    # REPLACE with:
    ax3d.text2D(
        -0.08, 0.5,                          # x, y in axes-fraction coords; nudge x left/right to taste
        r'$\frac{1}{N_{\rm trig}} \frac{d^2N_{\rm pair}}{d\Delta\eta\, d\Delta\phi}$',
        transform=ax3d.transAxes,
        fontsize=17,
        ha='center', va='center',
        rotation=90
    )
    ax3d.tick_params(axis='both', labelsize=12)
    ax3d.tick_params(axis='z',    labelsize=12)
    ax3d.set_yticks([-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2])
    ax3d.set_yticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
                           r'$\pi$', r'$\frac{3\pi}{2}$'], fontsize=13)
    ax3d.view_init(elev=25, azim=45)
    
    # ── Δφ projection ─────────────────────────────────────────────────────────
    if phi_proj_central is not None and phi_centers_folded is not None:
        pf_central = phi_proj_central
        pf_centers = phi_centers_folded
        pf_err     = phi_proj_err
    else:
        eta_mask       = np.abs(eta_centers) < ETA_CUT
        phi_projection = np.mean(correlation[eta_mask, :], axis=0) 
        pf_central, pf_centers = _fold_phi(phi_projection, phi_centers)
        pf_err = None

    phi_ticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    phi_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']

    if pf_err is not None:
        ax_phi.errorbar(pf_centers, pf_central, yerr=pf_err,
                        fmt='o', color=blue, ms=4, capsize=3, capthick=1,
                        elinewidth=1, label='Pythia (JK errors)')
    else:
        ax_phi.plot(pf_centers, pf_central,
                    'o', color=blue, ms=4, label='Pythia')

    # Always draw the ZYAM level — visible even when subtraction is off.
    ax_phi.axhline(y=background_level, color=red, linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'ZYAM min: {background_level:.4f}')

    # ── try to load the CMS data file ─────────────────────────────────────────
    cms_data_loaded = False
    cms_phi = None
    cms_y   = None

    if metadata:
        trig_range  = metadata.get('CUT_TRIG_PT_RANGE')  or metadata.get('TRIG_PT_RANGE')
        assoc_range = metadata.get('CUT_ASSOC_PT_RANGE') or metadata.get('ASSOC_PT_RANGE')

        if isinstance(trig_range, list) and isinstance(assoc_range, list):
            cms_path = (
                f"datathief/CMS_{trig_range[0]:.0f}-{trig_range[1]:.0f}"
                f"_{assoc_range[0]:.0f}-{assoc_range[1]:.0f}"
            )
            for ext in ('.csv', '.txt'):
                candidate = cms_path + ext
                if os.path.exists(candidate):
                    try:
                        cms_df  = pd.read_csv(candidate)
                        cms_phi = cms_df['DeltaPhi'].values
                        cms_y   = cms_df['d2Npair'].values
                        sort_idx = np.argsort(cms_phi)
                        cms_phi, cms_y = cms_phi[sort_idx], cms_y[sort_idx]
                        ax_phi.plot(cms_phi, cms_y, 's', color=green, ms=4,
                                    markerfacecolor='none', markeredgewidth=1.2,
                                    label='CMS Pythia data')
                        cms_data_loaded = True
                    except Exception as e:
                        print(f"Warning: could not load CMS data from {candidate}: {e}")
                    break

            if not cms_data_loaded:
                print(f"Warning: no CMS data file found at {cms_path}(.csv/.txt)")

    ax_phi.legend(fontsize=8)
    ax_phi.set_xlabel(r'|$\Delta\phi$| [rad]', fontsize=10)
    ax_phi.set_ylabel(
        r'$\frac{1}{N_{\rm trig}} \frac{dN_{\rm pair}}{d\Delta\phi}$', fontsize=10
    )
    ax_phi.set_title(rf'Projection onto $|\Delta\phi|$  ($|\Delta\eta|< {ETA_CUT}$)', fontsize=10)
    ax_phi.set_xlim(-0.1, np.pi)
    ax_phi.set_xticks(phi_ticks)
    ax_phi.set_xticklabels(phi_labels, fontsize=8)
    ax_phi.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_phi.grid(True, alpha=0.3)

    # ── CMS / Pythia ratio ────────────────────────────────────────────────────
    if cms_data_loaded and cms_phi is not None:
        pythia_dict = dict(zip(pf_centers, pf_central))
        err_dict    = dict(zip(pf_centers, pf_err)) if pf_err is not None else {}

        ratio_phi  = []
        ratio_vals = []
        ratio_errs = []

        for cp, cy in zip(cms_phi, cms_y):
            closest = min(pf_centers, key=lambda x: abs(x - cp))
            py_val  = pythia_dict[closest]
            if abs(closest - cp) <= d_phi / 2.0 and py_val != 0:
                ratio_phi.append(cp)
                ratio_vals.append(cy / py_val)
                if pf_err is not None and py_val != 0:
                    ratio_errs.append(abs(cy / py_val) * (err_dict[closest] / abs(py_val)))
                else:
                    ratio_errs.append(0.0)

        if ratio_phi:
            ratio_phi  = np.array(ratio_phi)
            ratio_vals = np.array(ratio_vals)
            ratio_errs = np.array(ratio_errs)

            if pf_err is not None and ratio_errs.any():
                ax_rat.errorbar(ratio_phi, ratio_vals, yerr=ratio_errs,
                                fmt='D', color='purple', ms=4,
                                markerfacecolor='none', markeredgewidth=1.2,
                                capsize=3, capthick=1, elinewidth=1,
                                label='CMS / Pythia')
            else:
                ax_rat.plot(ratio_phi, ratio_vals,
                            'D', color='purple', ms=4,
                            markerfacecolor='none', markeredgewidth=1.2,
                            label='CMS / Pythia')
        else:
            ax_rat.text(0.5, 0.5,
                        'No matching bins found\n(CMS & Pythia φ grids differ)',
                        ha='center', va='center',
                        transform=ax_rat.transAxes, fontsize=9, color='gray')
    else:
        ax_rat.text(0.5, 0.5, 'CMS data not loaded',
                    ha='center', va='center',
                    transform=ax_rat.transAxes, fontsize=9, color='gray')

    ax_rat.axhline(y=1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    ax_rat.set_xlabel(r'$|\Delta\phi|$ [rad]', fontsize=10)
    ax_rat.set_ylabel(r'CMS Pythia / Pythia',         fontsize=10)
    ax_rat.set_title(r'Ratio CMS Pythia / Pythia  ($|\Delta\eta|<1$)', fontsize=10)
    ax_rat.set_xlim(-0.1, np.pi)
    ax_rat.set_ylim(0.5, 1.5)
    ax_rat.set_xticks(phi_ticks)
    ax_rat.set_xticklabels(phi_labels, fontsize=8)
    ax_rat.legend(fontsize=8)
    ax_rat.grid(True, alpha=0.3)

    # ── Δη projection ─────────────────────────────────────────────────────────
    eta_projection = np.mean(correlation, axis=1)
    eta_proj_folded, eta_centers_folded = _fold_phi(eta_projection, eta_centers)

    ax_eta.plot(eta_centers_folded, eta_proj_folded,
                'o', color=orange, ms=4, label='Pythia')

    # Always draw the ZYAM line here too for consistency.
    ax_eta.axhline(y=background_level, color=red, linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'ZYAM min: {background_level:.4f}')
    ax_eta.legend(fontsize=8)

    ax_eta.set_xlabel(r'$|\Delta\eta|$', fontsize=10)
    ax_eta.set_ylabel(
        r'$\frac{1}{N_{\rm trig}} \frac{dN_{\rm pair}}{d\Delta\eta}$', fontsize=10
    )
    ax_eta.set_title(r'Projection onto $|\Delta\eta|$  (all $\Delta\phi$)', fontsize=10)
    ax_eta.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_eta.grid(True, alpha=0.3)

    # ── figure-level title ────────────────────────────────────────────────────
    suptitle = 'No FSR & ISR Dihadron Correlation'
    if fig_num:
        suptitle += f'  —  {fig_num}'
    if metadata:
        parts = []
        trig_range  = metadata.get('CUT_TRIG_PT_RANGE')  or metadata.get('TRIG_PT_RANGE')
        assoc_range = metadata.get('CUT_ASSOC_PT_RANGE') or metadata.get('ASSOC_PT_RANGE')
        if isinstance(trig_range, list):
            parts.append(f'Trigger $p_T$: {trig_range[0]:.1f}–{trig_range[1]:.1f} GeV/c')
        if isinstance(assoc_range, list):
            parts.append(f'Assoc $p_T$: {assoc_range[0]:.1f}–{assoc_range[1]:.1f} GeV/c')
            parts.append(rf'$\eta$ < 2.0')
        n_blocks_used = metadata.get('jackknife_n_blocks')
        if n_blocks_used:
            parts.append(f'JK blocks: {n_blocks_used}')
        if metadata.get('n_bins'):
            parts.append(f'{metadata["n_bins"]} pTHat bins combined')
        if parts:
            suptitle += '\n' + '   |   '.join(parts)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax3d, ax_phi, ax_rat, ax_eta)


# ──────────────────────────────────────────────────────────────────────────────
# Per-variant driver
# ──────────────────────────────────────────────────────────────────────────────
def run_variant(variant, pow_=5, base_dir='pythiaData/2760/cms'):
    """
    Run the full pipeline (load → correlate → jackknife → save CSV → plot)
    for a single simulation variant, e.g. 'quenched' or 'vacuum'.

    Only files whose name contains ``variant`` are picked up, so
    'quenched' and 'vacuum' runs never mix bins with each other even
    though they live in the same directory.

    Parameters
    ----------
    variant  : str   — substring used to filter input filenames, and used
                       to tag output files/figures (e.g. 'quenched')
    pow_     : int    — POW exponent used in the filename pattern
    base_dir : str    — directory holding the pTHat bin CSVs

    Returns
    -------
    fig : matplotlib Figure for this variant's plot window
    """
    base = f'{base_dir}/'

    filenames = sorted(glob.glob(f'{base}dihadron_pow{pow_}_pT*_{variant}_seed*.csv'))

    if not filenames:
        raise FileNotFoundError(
            f"No bin files found matching "
            f"{base}dihadron_pow{pow_}_pT*_{variant}_seed*.csv"
        )

    print(f"\n{'='*70}\nVariant: {variant}\n{'='*70}")
    print(f"Found {len(filenames)} pTHat bin file(s):")
    for f in filenames:
        print(f"  {f}")

    # ------------------------------------------------------------------
    # Load and merge all pTHat bins
    # ------------------------------------------------------------------
    print("\nLoading and combining bins...")
    data, metadata = read_and_combine_bins(filenames)

    print("\nCombined metadata:")
    for key, value in metadata.items():
        if key != 'BIN_INFO':
            print(f"  {key}: {value}")

    ntrig_override = metadata.get('N_TRIG')

    # ------------------------------------------------------------------
    # Full 2D correlation (needed for the 3D surface and Δη projection)
    # ------------------------------------------------------------------
    print("\nComputing 2D dihadron correlation...")
    (correlation, eta_centers, phi_centers,
     ntrig, bkg, raw_counts, weighted_counts) = compute_dihadron_correlation(
        data, metadata,
        eta_bins=15,
        phi_bins=33,
        eta_range=(-ETA_CUT, ETA_CUT),
        phi_range=(-np.pi, np.pi),
        zyam=True,
        zyam_range={'phi': (0.5, 1.5)},
        ntrig_override=ntrig_override,
    )

    print(f"Number of triggers (weighted): {ntrig:.4e}")
    print(f"Max correlation value:         {np.max(correlation):.6f}")
    print(f"Min correlation value:         {np.min(correlation):.6f}")
    print(f"Background level (ZYAM):       {bkg:.6f}")

    # weight-sanity diagnostics
    top_weights = data.nlargest(10, 'weight')[
        ['event', 'bin_index', 'weight', 'trigger_phi', 'assoc_phi',
         'trigger_eta', 'assoc_eta']
    ]
    print("\nTop-10 pairs by (rescaled) weight:")
    print(top_weights.to_string(index=False))
    print(f"\nWeight stats:")
    print(f"  max/mean ratio:         {data['weight'].max() / data['weight'].mean():.1f}x")
    print(f"  top-10 weight fraction: "
          f"{data.nlargest(10, 'weight')['weight'].sum() / data['weight'].sum():.1%}")

    # ------------------------------------------------------------------
    # Jackknife uncertainty on the Δφ projection
    # ------------------------------------------------------------------
    print(f"\nComputing jackknife uncertainties "
          f"({N_JACKKNIFE_BLOCKS} blocks) on Δφ projection...")

    jk_results = run_jackknife_multiple_blocks(
        data, metadata,
        block_sizes=N_JACKKNIFE_BLOCKS,
        eta_bins=15,
        phi_bins=33,
        eta_range=(-ETA_CUT, ETA_CUT),
        phi_range=(-np.pi, np.pi),
        zyam=True,
        zyam_range={'phi': (0.5, 1.5)},
        ntrig_override=ntrig_override,
    )

    phi_proj_central   = jk_results[N_JACKKNIFE_BLOCK]['phi_proj_central']
    phi_proj_err       = jk_results[N_JACKKNIFE_BLOCK]['phi_proj_err']
    phi_centers_folded = jk_results[N_JACKKNIFE_BLOCK]['phi_centers']

    metadata['jackknife_n_blocks'] = N_JACKKNIFE_BLOCKS

    # ------------------------------------------------------------------
    # Save projections to CSV — each variant gets its own subdirectory
    # ------------------------------------------------------------------
    print("\nSaving projections to CSV...")
    out_dir = f'plots/correlations/cms/{variant}'
    dphi_csv, deta_csv = save_projections_to_csv(
        phi_proj_central   = phi_proj_central,
        phi_proj_err       = phi_proj_err,
        phi_centers_folded = phi_centers_folded,
        correlation        = correlation,
        eta_centers        = eta_centers,
        phi_centers        = phi_centers,
        background_level   = bkg,
        metadata           = metadata,
        save_dir           = out_dir,
        tag                = f'pow{pow_}_{variant}',
    )

    # ------------------------------------------------------------------
    # Plot — its own figure window, its own save path
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plot_correlation_with_projections(
        correlation, eta_centers, phi_centers,
        background_level=bkg,              # always passed; always drawn
        phi_proj_central=phi_proj_central,
        phi_proj_err=phi_proj_err,
        phi_centers_folded=phi_centers_folded,
        metadata=metadata,
        save_path=(
            f'{out_dir}/dihadron_2d_combined_and_projections'
            f'_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}'
            f'_{ASSOC_PT_MIN:.1f}-{ASSOC_PT_MAX:.1f}_{variant}.png'
        ),
        fig_num=f'Dihadron Correlation — {variant}',
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    POW = 3
    BASE_DIR = 'pythiaData/2760/cms'

    figs = {}
    for variant in VARIANTS:
        figs[variant] = run_variant(variant, pow_=POW, base_dir=BASE_DIR)

    # All figures were created with distinct fig_num values above, so each
    # lives in its own window; a single blocking show() at the end displays
    # every one of them simultaneously.
    plt.show()


if __name__ == '__main__':
    main()