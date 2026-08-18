"""
Loads DGLV radiative energy-loss fraction JSON files makes a
interpolating interface for use in other scripts.

Usage:
    from dglv_interpolator import DGLVTable, DGLVInterpolator

    # Single flavour
    bottom = DGLVTable.from_json("energyLoss/energyLossFraction/0.25-5.-bottom.json")
    eps = bottom.interpolator(12.5)   # ΔE/E at pT = 12.5 GeV

    # All four flavours at once
    eloss = DGLVInterpolator("energyLoss/energyLossFraction")
    eps = eloss("charm", 15.0)        # ΔE/E for charm at pT = 15 GeV
"""

import json
import os
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# ── Single-flavour container ──────────────────────────────────────────────────


class DGLVTable:
    """
    Holds the raw (pT, ΔE/E) data and a cubic-spline interpolator
    for a single parton flavour.
    """

    def __init__(self, pT: np.ndarray, eps: np.ndarray, meta: dict | None = None):
        """
        Parameters
        ----------
        pT   : 1-D array of transverse momenta (GeV)
        eps  : 1-D array of ΔE/E values (dimensionless)
        meta : optional dict of metadata from the JSON file
        """
        if len(pT) != len(eps):
            raise ValueError("pT and eps arrays must have the same length.")

        self.pT = pT
        self.eps = eps
        self.meta = meta or {}

        # Build spline on strictly-monotonic subset (drops duplicate x values)
        mask = np.concatenate(([True], np.diff(pT) > 0))
        self._spline = CubicSpline(pT[mask], eps[mask], extrapolate=False)

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "DGLVTable":
        """Load a DGLVTable directly from a JSON file."""
        with open(path, "r") as f:
            raw = json.load(f)

        values = raw["data"]["DGLV Delta E / E"]["value"]
        pts = np.array(values)
        meta = {**raw.get("meta", {}), **raw.get("parameters", {})}
        return cls(pts[:, 0], pts[:, 1], meta=meta)

    # ── Interpolation ─────────────────────────────────────────────────────────

    def __call__(self, pT: float | np.ndarray) -> float | np.ndarray:
        """Evaluate ΔE/E at the given pT (GeV). Alias for self.interpolate()."""
        return self.interpolate(pT)

    def interpolate(self, pT: float | np.ndarray) -> float | np.ndarray:
        """
        Return ΔE/E at pT, clamped to the table's pT range so that
        values outside [pT_min, pT_max] return the nearest boundary value
        rather than NaN.
        """
        pT_clamped = np.clip(pT, self.pT_min, self.pT_max)
        return self._spline(pT_clamped)

    def interpolate_or_nan(self, pT: float | np.ndarray) -> float | np.ndarray:
        """Like interpolate(), but returns NaN outside the table range."""
        return self._spline(pT)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pT_min(self) -> float:
        return (
            float(self.pT[self.pT > 0].min())
            if (self.pT > 0).any()
            else float(self.pT.min())
        )

    @property
    def pT_max(self) -> float:
        return float(self.pT.max())

    def __repr__(self) -> str:
        flavour = self.meta.get("flavour", {}).get("type", "unknown")
        return (
            f"DGLVTable(flavour={flavour!r}, "
            f"N={len(self.pT)}, pT∈[{self.pT_min:.1f}, {self.pT_max:.1f}] GeV)"
        )


# ── Multi-flavour manager ─────────────────────────────────────────────────────


class DGLVInterpolator:
    """
    Loads DGLV energy-loss tables for all four parton flavours
    from a directory and provides a unified query interface.

    Expected filenames (customisable via `filenames` argument):
        bottom : 0.25-5.-bottom.json
        gluon  : 0.25-5.-gluons.json
        light  : 0.25-5.-light.json
        charm  : 0.25-5.-charm.json
    """

    DEFAULT_FILENAMES: ClassVar[dict[str, str]] = {
        "bottom": "0.25-5.-bottom.json",
        "gluon": "0.25-5.-gluons.json",
        "light": "0.25-5.-light.json",
        "charm": "0.25-5.-charm.json",
    }

    COLOURS: ClassVar[dict[str, str]] = {
        "bottom": "#e6194b",
        "gluon": "#3cb44b",
        "light": "#4363d8",
        "charm": "#f58231",
    }

    def __init__(self, data_dir: str, filenames: dict | None = None):
        """
        Parameters
        ----------
        data_dir  : directory containing the JSON files
        filenames : optional dict mapping flavour name → filename,
                    overrides DEFAULT_FILENAMES
        """
        self.data_dir = data_dir
        self.filenames = filenames or self.DEFAULT_FILENAMES
        self.tables: dict[str, DGLVTable] = {}
        self._load_all()

    def _load_all(self):
        for flavour, filename in self.filenames.items():
            path = os.path.join(self.data_dir, filename)
            self.tables[flavour] = DGLVTable.from_json(path)
            print(f"Loaded {flavour:6s}: {self.tables[flavour]}")

    # ── Query interface ───────────────────────────────────────────────────────

    def __call__(self, flavour: str, pT: float | np.ndarray) -> float | np.ndarray:
        """Evaluate ΔE/E for a given flavour and pT (GeV)."""
        return self[flavour].interpolate(pT)

    def __getitem__(self, flavour: str) -> DGLVTable:
        """Access a DGLVTable by flavour name, e.g. eloss['charm']."""
        if flavour not in self.tables:
            raise KeyError(
                f"Unknown flavour {flavour!r}. Available: {list(self.tables)}"
            )
        return self.tables[flavour]

    @property
    def flavours(self) -> list[str]:
        return list(self.tables.keys())

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        pT_range: tuple = (5, 100),
        n_points: int = 500,
        show_data: bool = False,
        ax=None,
    ) -> plt.Axes:
        """
        Plot all four interpolated curves on a single axis.

        Parameters
        ----------
        pT_range  : (pT_min, pT_max) for the smooth curve
        n_points  : number of points on the interpolated curve
        show_data : overlay the raw tabulated data points
        ax        : existing matplotlib Axes (creates a new figure if None)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        pT_plot = np.linspace(*pT_range, n_points)

        for flavour, table in self.tables.items():
            colour = self.COLOURS.get(flavour, None)
            y_plot = table(pT_plot)

            ax.plot(pT_plot, y_plot, color=colour, lw=6, label=flavour.capitalize())

            if show_data:
                ax.scatter(table.pT, table.eps, color=colour, s=25, zorder=5, alpha=0.8)

        ax.set_xlabel(r"$p_T$ (GeV)", fontsize=20)
        ax.set_ylabel(r"$\Delta E / E$", fontsize=20)
        ax.set_title(
            r"DGLV Radiative Energy Loss Fraction"
            "\n"
            r"$T = 0.25\,\mathrm{GeV},\; L = 5\,\mathrm{fm},\; \alpha_s = 0.3$",
            fontsize=20,
        )
        ax.legend(fontsize=18)
        ax.set_xlim(0, pT_range[1] + 5)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self) -> str:
        return f"DGLVInterpolator(flavours={self.flavours}, dir={self.data_dir!r})"


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    eloss = DGLVInterpolator(os.path.join("energyLoss", "energyLossFraction"))

    # Example queries
    for flavour in eloss.flavours:
        eps = eloss(flavour, 20.0)
        print(f"  ΔE/E({flavour:6s}, pT=20 GeV) = {eps:.4f}")

    ax = eloss.plot()
    plt.tight_layout()
    plt.savefig(
        os.path.join("outputs", "plots", "energyloss", "energy_loss_fractions.svg"),
        dpi=150,
    )
    plt.show()
    print("Plot saved to energy_loss_fractions.svg")
