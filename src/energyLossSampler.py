"""
energy_loss_sampler.py
----------------------
Builds a per-parton Gaussian energy-loss sampler on top of DGLVInterpolator.

For a parton of flavour f and transverse momentum pT, the DGLV table gives
the mean energy-loss fraction μ = ΔE/E(f, pT).  A Gaussian centred on μ with
width sigma(μ, pT) is constructed, and one sample ε ∈ [0, 1] is drawn from it.
The parton pT is then rescaled as  pT_final = (1 - ε) · pT.

Usage
-----
    from energy_loss_sampler import EnergyLossSampler

    sampler = EnergyLossSampler("energyLoss/energyLossFraction")

    # Single parton
    eps = sampler.sample("charm", pT=15.0)
    pT_quenched = (1 - eps) * 15.0

    # Vectorised over many partons of the same flavour
    eps_array = sampler.sample("gluon", pT=np.array([10., 20., 35.]))

    # Full rescaling in one call
    pT_q = sampler.quench("light", pT=22.0)
"""

import matplotlib.pyplot as plt
import numpy as np

from dglvInterpolator import DGLVInterpolator

# ── Width strategies (placeholder + built-in options) ────────────────────────


def sigma(mu: float | np.ndarray, pT: float | np.ndarray) -> float | np.ndarray:
    """
    PLACEHOLDER — replace with a physically motivated prescription.

    Current default: σ = 0.1 · μ  (10 % of the mean loss fraction).
    This keeps the Gaussian narrow relative to the mean and is easy to swap out.

    Parameters
    ----------
    mu : mean energy-loss fraction ΔE/E  (scalar or array)
    pT : parton transverse momentum in GeV (scalar or array, not used here)

    Returns
    -------
    σ  : Gaussian width (same shape as mu / pT)
    """
    return np.sqrt(mu / pT)


# ── Main sampler class ────────────────────────────────────────────────────────


class EnergyLossSampler:
    """
    Samples a radiative energy-loss fraction ε for a single parton per event.

    The distribution is a Gaussian  N(μ, σ²)  clipped to [0, 1], where
        μ = DGLVInterpolator(flavour, pT)   — the DGLV mean loss fraction
        σ = sigma_fn(μ, pT)                 — pluggable width strategy

    Parameters
    ----------
    data_dir   : path to the directory containing the four JSON tables
    sigma_fn   : callable(mu, pT) → σ.  Defaults to sigma_placeholder.
                 Swap this out once the physical prescription is decided.
    filenames  : optional override of the default JSON filename mapping
    rng        : optional numpy Generator for reproducibility
                 (e.g. np.random.default_rng(seed=42))
    """

    def __init__(
        self,
        data_dir: str,
        filenames: dict | None = None,
        rng: np.random.Generator = None,
    ):

        self.interpolator = DGLVInterpolator(data_dir, filenames=filenames)
        self.sigma_fn = sigma
        self.rng = rng or np.random.default_rng()

    # ── Core sampling ─────────────────────────────────────────────────────────

    def mean(self, flavour: str, pT: float | np.ndarray) -> float | np.ndarray:
        """Return μ = ΔE/E for a given flavour and pT."""
        return self.interpolator(flavour, pT)

    def sigma(self, flavour: str, pT: float | np.ndarray) -> float | np.ndarray:
        """Return σ for a given flavour and pT, via the pluggable sigma_fn."""
        mu = self.mean(flavour, pT)
        return self.sigma_fn(mu, pT)

    def sample(self, flavour: str, pT: float | np.ndarray) -> float | np.ndarray:
        """
        Draw one ε per parton from N(μ, σ²), clipped to [0, 1].

        Parameters
        ----------
        flavour : one of 'bottom', 'gluon', 'light', 'charm'
        pT      : parton pT in GeV (scalar or array)

        Returns
        -------
        ε : sampled energy-loss fraction, same shape as pT
        """
        mu = self.mean(flavour, pT)
        sig = self.sigma_fn(mu, pT)

        eps = self.rng.normal(loc=mu, scale=sig)
        return np.clip(eps, 0.0, 1.0)

    def quench(self, flavour: str, pT: float | np.ndarray) -> float | np.ndarray:
        """
        Sample ε and return the rescaled pT:  pT_quenched = (1 - ε) · pT.
        """
        eps = self.sample(flavour, pT)
        return (1.0 - eps) * pT

    # ── Diagnostics / visualisation ───────────────────────────────────────────

    def plot_distribution(
        self,
        flavour: str,
        pT: float,
        n_bins: int = 60,
        n_samples: int = 100_000,
        ax=None,
    ) -> plt.Axes:
        """
        Plot the ε distribution for a single (flavour, pT) point by
        drawing n_samples from the Gaussian and overlaying the theoretical PDF.

        Parameters
        ----------
        flavour   : parton flavour string
        pT        : parton pT in GeV
        n_bins    : histogram bins
        n_samples : Monte Carlo draws for the histogram
        ax        : existing Axes (creates new figure if None)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        mu = float(self.mean(flavour, pT))
        sig = float(self.sigma_fn(mu, pT))

        # Draw samples
        samples = self.sample(flavour, np.full(n_samples, pT))

        # Histogram
        ax.hist(
            samples,
            bins=n_bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="Sampled ε",
        )

        # Theoretical Gaussian PDF (unclipped, for reference)
        eps_range = np.linspace(0, 1, 400)
        pdf = (
            1.0
            / (sig * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((eps_range - mu) / sig) ** 2)
        )
        ax.plot(eps_range, pdf, "r-", lw=2, label=r"$\mathcal{N}(\mu,\sigma^2)$ PDF")

        ax.axvline(mu, color="k", ls="--", lw=1.2, label=rf"$\mu={mu:.4f}$")

        ax.set_xlabel(r"$\varepsilon = \Delta E / E$", fontsize=12)
        ax.set_ylabel("Probability density", fontsize=12)
        ax.set_title(
            rf"Energy-loss distribution: {flavour}, $p_T = {pT}$ GeV"
            "\n"
            rf"$\mu = {mu:.4f}$, $\sigma = {sig:.4f}$  "
            rf"[{self.sigma_fn.__name__}]",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_mean_and_sigma(
        self, pT_range: tuple = (5, 100), n_points: int = 300, ax=None
    ) -> plt.Axes:
        """
        Plot μ(pT) with ±σ bands for all four flavours.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        pT_arr = np.linspace(*pT_range, n_points)
        colours = DGLVInterpolator.COLOURS

        for flavour in self.interpolator.flavours:
            mu = self.mean(flavour, pT_arr)
            sig = self.sigma_fn(mu, pT_arr)
            c = colours.get(flavour)

            ax.plot(pT_arr, mu, color=c, lw=2, label=flavour.capitalize())
            ax.fill_between(
                pT_arr,
                np.clip(mu - sig, 0, 1),
                np.clip(mu + sig, 0, 1),
                color=c,
                alpha=0.15,
            )

        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=13)
        ax.set_ylabel(r"$\varepsilon = \Delta E / E$", fontsize=13)
        ax.set_title(
            r"DGLV mean loss $\mu(p_T)$ with $\pm\sigma$ band"
            "\n"
            rf"[width strategy: {self.sigma_fn.__name__}]",
            fontsize=11,
        )
        ax.legend(fontsize=11)
        ax.set_xlim(pT_range[0], pT_range[1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        return ax

    # ── Pluggable width ───────────────────────────────────────────────────────

    def set_sigma(self, sigma_fn) -> None:
        """
        Swap in a new width strategy at runtime.

        Example
        -------
            from energy_loss_sampler import sigma_fixed
            sampler.set_sigma(sigma_fixed(0.02))
        """
        self.sigma_fn = sigma_fn

    def __repr__(self) -> str:
        return (
            f"EnergyLossSampler("
            f"flavours={self.interpolator.flavours}, "
            f"sigma_fn={self.sigma_fn.__name__!r})"
        )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    sampler = EnergyLossSampler(
        os.path.join("energyLoss", "energyLossFraction"),
        rng=np.random.default_rng(seed=42),
    )

    print(sampler)
    print()

    # Single-parton examples
    for flavour in sampler.interpolator.flavours:
        eps = sampler.sample(flavour, pT=20.0)
        pT_q = sampler.quench(flavour, pT=20.0)
        print(f"  {flavour:6s}  pT=20 GeV  →  ε={eps:.4f}  pT_quenched={pT_q:.3f} GeV")

    print()

    # μ ± σ overview plot
    ax1 = sampler.plot_mean_and_sigma()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "energyloss", "mean_sigma_bands.png"), dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sampler.plot_distribution("gluon", pT=20.0, ax=axes[0, 0])
    sampler.plot_distribution("light", pT=20.0, ax=axes[0, 1])
    sampler.plot_distribution("charm", pT=20.0, ax=axes[1, 0])
    sampler.plot_distribution("bottom", pT=20.0, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(
        os.path.join("plots", "energyloss", "epsilon_distribution_20GeV.png"), dpi=150
    )
    plt.show()
