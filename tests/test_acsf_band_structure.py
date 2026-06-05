"""
test_acsf_band_structure.py
===========================
Tests for ACSF linear tight-binding band structures.

Hamiltonian convention
----------------------
``get_acsf_hopping_descriptors`` uses ``NeighborList(bothways=False)``, which
is a **half-list** — each bond (i,j) is stored once.  The Bloch Hamiltonian is
therefore built as ``H = H + H.conj().T`` (no factor of ½).  Dividing by 2
would halve every hopping amplitude and produce a bandwidth 2× too small.

Two test classes:

1. ``TestACSFBandsABBilayer``
   - Primitive 4-atom AB-stacked bilayer graphene (dense Hamiltonian).
   - k-path: K → M → Γ → K
   - Asserts: all eigenvalues finite; low-energy gap at K < GAP_THRESHOLD_EV.
   - Saves figure: tests/figures/acsf_ab_bilayer_bands_M15_W6.png

2. ``TestACSFBandsTBG``  (marked ``slow``)
   - 9.43° twisted bilayer graphene (dense eigensolver).
   - k-path: K → M → Γ → K; energy window [-3.0, 3.0] eV.
   - Asserts: all eigenvalues finite; correct shape.
   - Saves figure: tests/figures/acsf_tblg_9p43deg_bands_M15_W6.png

ACSF parameters
---------------
If ``best_fit_params/ACSF_hoppings_M_15_W_6_best_fit_params.npz`` is absent,
the ``acsf_params`` fixture auto-fits the model from the HDF5 training data in
``data/hoppings/`` and saves the result.  Requires the training data to be
present; otherwise tests are skipped.

Run all:          pytest tests/test_acsf_band_structure.py -v
Run without slow: pytest tests/test_acsf_band_structure.py -v -m "not slow"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

# ---------------------------------------------------------------------------
# Package import path setup (mirrors other tests in this directory)
# ---------------------------------------------------------------------------

def _ensure_importable_package() -> None:
    try:
        import blg_model_builder_v2  # noqa: F401
        return
    except Exception:
        pass
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import blg_model_builder_v2  # noqa: F401
        return
    except Exception:
        import types
        pkg = types.ModuleType("blg_model_builder_v2")
        pkg.__path__ = [str(src)]  # type: ignore[attr-defined]
        sys.modules["blg_model_builder_v2"] = pkg


_ensure_importable_package()

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------

_BEST_FIT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "uncertainty_quantification", "best_fit_params")
)
_FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACSF_M: int = 15
ACSF_W: int = 6
ACSF_RCUT: float = 6.0

NK: int = 60       # k-points per high-symmetry segment (3 segments → ~180 total)
N_EIGS: int = 40   # number of eigenvalues for sparse eigsh (TBG only)

# AB bilayer: low-energy pair of bands at K must be within this gap (eV)
GAP_THRESHOLD_EV: float = 0.5

# k-path nodes (reduced coordinates, hexagonal cell)
# a1 = [a, 0, 0]  a2 = [a/2, a√3/2, 0]
_K_NODE     = [1 / 3, 2 / 3, 0.0]
_M_NODE     = [1 / 2, 0.0,   0.0]
_GAMMA_NODE = [0.0,   0.0,   0.0]
_SYM_PTS    = [_K_NODE, _GAMMA_NODE, _M_NODE, _K_NODE]   # K → Γ → M → K
_SYM_LABELS = ["K", "Γ", "M", "K"]

# ---------------------------------------------------------------------------
# Band-structure helper
# ---------------------------------------------------------------------------

def _build_bands(
    atoms,
    params: np.ndarray,
    kvec_cart: np.ndarray,
    n_eigs: int | None = None,
) -> tuple[np.ndarray, float]:
    """Compute ACSF tight-binding band structure along *kvec_cart*.

    Parameters
    ----------
    atoms : ase.Atoms
    params : ndarray, shape (n_features,) — ACSF linear model weights.
    kvec_cart : ndarray, shape (n_kpts, 3) — Cartesian k-points (Å⁻¹, with 2π).
    n_eigs : int or None
        * ``None``  → full dense diagonalization (``np.linalg.eigh``); the
          Fermi level is computed from the resulting bands and subtracted so
          that E=0 sits at mid-gap.
        * ``int``   → sparse ``eigsh(H, k=n_eigs, sigma=0)``; no Fermi shift
          (shift-invert already centres results near 0).

    Returns
    -------
    evals : ndarray, shape (n_kpts, n_bands)
        Sorted real eigenvalues (eV) at each k-point.  For the dense path
        these are shifted so that the Fermi level is at 0 eV.
    fermi_level : float
        For the dense path: midgap energy (eV) subtracted from eigenvalues.
        For the sparse path: 0.0 (no shift applied).
    """
    from blg_model_builder_v2.tb_descriptors import get_acsf_hopping_descriptors
    from blg_model_builder_v2.tb_models import get_acsf_hoppings

    descriptors, (pair_i, pair_j, pair_v) = get_acsf_hopping_descriptors(
        atoms, M=ACSF_M, W=ACSF_W, r_cut=ACSF_RCUT,
    )
    hoppings = get_acsf_hoppings(descriptors, params)  # (n_pairs,)

    N = len(atoms)
    evals_list: list[np.ndarray] = []

    for k_cart in kvec_cart:
        phases = np.exp(1j * (pair_v @ k_cart))   # (n_pairs,)
        hop_vals = hoppings * phases               # complex (n_pairs,)

        H = scipy.sparse.coo_matrix(
            (hop_vals, (pair_i, pair_j)),
            shape=(N, N),
            dtype=np.complex128,
        ).tocsr()
        # NeighborList(bothways=False) is a half-list: each bond (i,j) is stored
        # only once.  H therefore has entries only on one triangle.  Adding H†
        # fills the other triangle to make H Hermitian.  No factor of ½ —
        # dividing by 2 would halve every hopping amplitude and produce a
        # bandwidth that is 2× too small.
        H = H + H.conj().T

        if n_eigs is None:
            vals = np.linalg.eigh(H.toarray())[0]
        else:
            k_req = min(n_eigs, N - 2)  # eigsh requires k < N
            # sigma=0 activates shift-invert mode: ARPACK solves (H - 0·I)⁻¹ v = ν v,
            # so eigenvalues closest to 0 (Fermi level) map to the largest ν and are
            # found first.  No explicit `which` is needed — the default in shift-invert
            # is "LM" on the transformed problem, which recovers the eigenvalues
            # nearest to sigma regardless of sign.
            vals, _ = scipy.sparse.linalg.eigsh(H, k=k_req, sigma=0.0)
            vals = np.sort(vals.real)

        evals_list.append(vals)

    evals = np.array(evals_list)  # (n_kpts, n_bands)

    # For the dense path, shift eigenvalues so the Fermi level sits at E = 0.
    # nocc = nbands // 2 (one pz electron per carbon atom → half-filling).
    # Fermi level = midpoint between the band just above and just below nocc.
    if n_eigs is None:
        n_bands = evals.shape[1]
        nocc = n_bands // 2
        fermi_level = float((evals[0, nocc] + evals[0, nocc-1]) / 2)
        evals = evals - fermi_level
    else:
        fermi_level = 0.0

    return evals, fermi_level


def _save_band_figure(
    evals: np.ndarray,
    k_dist: np.ndarray,
    k_node: np.ndarray,
    sym_labels: list[str],
    title: str,
    filepath: str,
    scatter: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot and save a band-structure figure.

    Parameters
    ----------
    evals    : (n_kpts, n_bands) eigenvalues in eV.
    k_dist   : (n_kpts,) accumulated k-distance for x-axis.
    k_node   : (n_nodes,) x-positions of high-symmetry points.
    sym_labels: label strings for each node.
    title    : figure title.
    filepath : output path (PNG).
    scatter  : if True, use a scatter plot instead of connected lines.
    ylim     : optional (ymin, ymax) energy window in eV.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not available; skip silently

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    if scatter:
        # Flatten to (n_kpts * n_bands,) arrays for a single scatter call
        k_rep = np.repeat(k_dist, evals.shape[1])
        e_flat = evals.ravel()
        ax.scatter(k_rep, e_flat, s=1.0, color="steelblue", alpha=0.5, linewidths=0)
    else:
        for band_idx in range(evals.shape[1]):
            ax.plot(k_dist, evals[:, band_idx], color="steelblue", linewidth=0.8, alpha=0.7)

    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, label="E = 0")

    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6)

    ax.set_xlim(k_dist[0], k_dist[-1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels, fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_overlay_figure(
    evals_dense: np.ndarray,
    evals_sparse: np.ndarray,
    k_dist: np.ndarray,
    k_node: np.ndarray,
    sym_labels: list[str],
    title: str,
    filepath: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Overlay sparse eigsh bands on top of the full dense band structure.

    Dense bands are drawn as thin light-grey lines in the background; sparse
    bands (shifted to the same Fermi level) are drawn as thicker coloured
    lines in the foreground.

    Parameters
    ----------
    evals_dense  : (n_kpts, n_bands_dense) — dense eigenvalues, E_F = 0.
    evals_sparse : (n_kpts, n_eigs) — sparse eigenvalues, shifted to same E_F.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Dense bands — background reference
    for band_idx in range(evals_dense.shape[1]):
        ax.plot(
            k_dist, evals_dense[:, band_idx],
            color="lightgrey", linewidth=0.7, alpha=0.8, zorder=1,
        )

    # Sparse bands — foreground overlay
    for band_idx in range(evals_sparse.shape[1]):
        ax.scatter(
            k_dist, evals_sparse[:, band_idx],
            color="black", s=1.0, alpha=0.85, zorder=2,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, zorder=3,
               label="E = 0 (E_F)")

    for xv in k_node:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.6, zorder=3)

    ax.set_xlim(k_dist[0], k_dist[-1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(k_node)
    ax.set_xticklabels(sym_labels, fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title(title, fontsize=10)

    # Custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lightgrey",  linewidth=1.5, label=f"Dense (all bands)"),
        Line2D([0], [0], color="black",  linewidth=1.5, label=f"Sparse (N={evals_sparse.shape[1]})"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.0, label="E_F"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared fixture: ACSF best-fit parameters
# ---------------------------------------------------------------------------

def _fit_and_save_acsf_params(npz_path: str) -> np.ndarray:
    """Fit ACSF linear hopping model and save weights to *npz_path*.

    Changes to the ``uncertainty_quantification/`` directory during the fit
    so that ``DataLoader`` can resolve relative paths to ``../data/hoppings/``,
    then restores the original working directory.
    """
    uq_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "uncertainty_quantification"))
    orig_dir = os.getcwd()
    try:
        os.chdir(uq_dir)
        if uq_dir not in sys.path:
            sys.path.insert(0, uq_dir)
        from blg_model_builder_v2.DataLoader import load_data_for_model
        from model_fit import fit_acsf_linear_hopping

        xdata_tr, _, _, ydata_tr, _, _ = load_data_for_model(
            "ACSF_hoppings", M=ACSF_M, W=ACSF_W,
        )
        params, _ = fit_acsf_linear_hopping(
            xdata_tr["hopping"], ydata_tr["hopping"],
        )
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(
            npz_path,
            params=params,
            acsf_M=np.int32(ACSF_M),
            acsf_W=np.int32(ACSF_W),
        )
        print(f"\n[acsf_params] fitted M={ACSF_M} W={ACSF_W} → {npz_path}", flush=True)
        return np.asarray(params, dtype=np.float64)
    finally:
        os.chdir(orig_dir)


@pytest.fixture(scope="module")
def acsf_params() -> np.ndarray:
    npz_path = os.path.join(
        _BEST_FIT_DIR,
        f"ACSF_hoppings_M_{ACSF_M}_W_{ACSF_W}_best_fit_params.npz",
    )
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        return np.asarray(data["params"], dtype=np.float64)
    # Auto-fit: requires training data in data/hoppings/*.hdf5
    try:
        return _fit_and_save_acsf_params(npz_path)
    except Exception as exc:
        pytest.skip(
            f"ACSF best-fit params not found and auto-fit failed: {exc}\n"
            f"Expected: {npz_path}\n"
            "Ensure data/hoppings/*.hdf5 training files are present, or run "
            "uncertainty_quantification/get_MCMC_inputs.py to generate the npz."
        )


# ---------------------------------------------------------------------------
# Class 1: AB-stacked bilayer graphene
# ---------------------------------------------------------------------------

class TestACSFBandsABBilayer:
    """ACSF band structure for primitive AB-stacked bilayer graphene (4 atoms)."""

    @pytest.fixture(scope="class")
    def ab_atoms(self):
        from blg_model_builder_v2.geom_tools import get_bilayer_atoms
        return get_bilayer_atoms(d=3.35, disregistry=0.0, sc=1)

    @pytest.fixture(scope="class")
    def ab_bands(self, ab_atoms, acsf_params):
        """Pre-compute bands once for the whole class."""
        from blg_model_builder_v2.tb_models import k_path, get_recip_cell
        kvec, k_dist, k_node = k_path(_SYM_PTS, NK)
        cell = np.array(ab_atoms.get_cell())
        kvec_cart = kvec @ get_recip_cell(cell.T)
        evals, _ = _build_bands(ab_atoms, acsf_params, kvec_cart, n_eigs=None)
        return evals, kvec, k_dist, k_node

    def test_ab_bands_finite(self, ab_bands):
        """All AB bilayer eigenvalues must be finite."""
        evals, *_ = ab_bands
        assert np.all(np.isfinite(evals)), (
            f"Non-finite eigenvalues in AB bilayer bands: "
            f"{np.sum(~np.isfinite(evals))} bad values."
        )

    def test_ab_bands_correct_shape(self, ab_bands, ab_atoms):
        """Shape: (n_kpts, n_atoms) for dense diagonalization."""
        evals, _, k_dist, _ = ab_bands
        n_kpts = len(k_dist)
        n_atoms = len(ab_atoms)
        assert evals.shape == (n_kpts, n_atoms), (
            f"Expected shape ({n_kpts}, {n_atoms}), got {evals.shape}."
        )

    def test_ab_bands_gapless_at_K(self, ab_bands):
        """AB-Bernal bilayer: low-energy pair of bands touch at K (gap < threshold).

        The primitive cell has 4 atoms → 4 bands.  At the K point the two
        central bands (indices 1 and 2, sorted by energy) should be nearly
        degenerate for physically reasonable hopping parameters.
        """
        evals, kvec, k_dist, _ = ab_bands
        from blg_model_builder_v2.tb_models import k_path, get_recip_cell

        # Find the k-index closest to the starting K node (first point in path)
        k_K_frac = np.array(_K_NODE)
        dists = np.linalg.norm(kvec - k_K_frac, axis=1)
        k_idx = int(np.argmin(dists))

        vals_at_K = np.sort(evals[k_idx])  # 4 values
        gap = float(abs(vals_at_K[2] - vals_at_K[1]))
        assert gap < GAP_THRESHOLD_EV, (
            f"AB bilayer: gap between central bands at K = {gap:.4f} eV, "
            f"expected < {GAP_THRESHOLD_EV} eV.  "
            "Check ACSF params or interlayer coupling."
        )

    def test_ab_bands_save_figure(self, ab_bands):
        """Save band-structure plot to tests/figures/."""
        evals, _, k_dist, k_node = ab_bands
        outpath = os.path.join(
            _FIGURES_DIR,
            f"acsf_ab_bilayer_bands_M{ACSF_M}_W{ACSF_W}.png",
        )
        _save_band_figure(
            evals, k_dist, k_node, _SYM_LABELS,
            title=f"AB Bilayer Graphene — ACSF M={ACSF_M}, W={ACSF_W}",
            filepath=outpath,
        )
        assert os.path.isfile(outpath), f"Figure not written to {outpath}"


# ---------------------------------------------------------------------------
# Class 2: Twisted bilayer graphene at 9.4°
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestACSFBandsTBG:
    """ACSF band structure for 9.4° twisted bilayer graphene (dense eigensolver).

    9.4° gives a commensurate supercell of ~148 atoms (p=4, q=3), which is
    small enough for full dense diagonalization.  Bands are plotted over the
    energy window [-3.0, 3.0] eV.

    Skip with: pytest -m "not slow"
    """

    TWIST_ANGLE: float = 9.43
    LAYER_SEP: float = 3.35
    EMIN: float = -1.5
    EMAX: float = 1.5
    A_TOL: float = 0.1

    @pytest.fixture(scope="class")
    def tblg_atoms(self):
        fg = pytest.importorskip(
            "flatgraphene",
            reason="flatgraphene not installed; skipping TBG tests.",
        )
        p, q, _ = fg.twist.find_p_q(self.TWIST_ANGLE, a_tol = 0.1)
        atoms = fg.twist.make_graphene(
            cell_type="hex",
            n_layer=2,
            p=p,
            q=q,
            lat_con=2.46,
            sym=["C", "C"],
            mass=[12.01, 12.01],
            sep=self.LAYER_SEP,
            h_vac=20,
        )
        return atoms

    @pytest.fixture(scope="class")
    def tblg_bands(self, tblg_atoms, acsf_params):
        """Pre-compute TBG bands once for the whole class (dense solver).

        Returns
        -------
        evals : ndarray (n_kpts, n_atoms) — eigenvalues shifted to E_F = 0
        fermi_level : float — Fermi energy subtracted from dense eigenvalues (eV)
        kvec : ndarray (n_kpts, 3) — reduced k-coordinates
        k_dist : ndarray (n_kpts,) — accumulated k-distance
        k_node : ndarray (n_nodes,) — x-positions of high-symmetry points
        """
        from blg_model_builder_v2.tb_models import k_path, get_recip_cell
        kvec, k_dist, k_node = k_path(_SYM_PTS, NK)
        cell = np.array(tblg_atoms.get_cell())
        kvec_cart = kvec @ get_recip_cell(cell.T)
        # n_eigs=None → full dense diagonalization via np.linalg.eigh
        evals, fermi_level = _build_bands(tblg_atoms, acsf_params, kvec_cart, n_eigs=None)
        return evals, fermi_level, kvec, k_dist, k_node

    @pytest.fixture(scope="class")
    def tblg_bands_sparse(self, tblg_atoms, acsf_params, tblg_bands):
        """Compute N_EIGS sparse eigenvalues, shifted by the dense Fermi level.

        Uses the Fermi level from ``tblg_bands`` so both calculations share the
        same E=0 reference, enabling direct overlay comparison.
        """
        from blg_model_builder_v2.tb_models import k_path, get_recip_cell
        _, fermi_level, _, k_dist, k_node = tblg_bands
        kvec, _, _ = k_path(_SYM_PTS, NK)
        cell = np.array(tblg_atoms.get_cell())
        kvec_cart = kvec @ get_recip_cell(cell.T)
        evals, _ = _build_bands(tblg_atoms, acsf_params, kvec_cart, n_eigs=N_EIGS)
        evals = evals - fermi_level
        return evals, k_dist, k_node

    def test_tblg_atoms_size(self, tblg_atoms):
        """Sanity check: 9.4° TBG supercell should have > 20 atoms."""
        assert len(tblg_atoms) > 20, (
            f"Expected > 20 atoms for 9.4° TBG, got {len(tblg_atoms)}."
        )

    def test_tblg_bands_finite(self, tblg_bands):
        """All TBG eigenvalues must be finite."""
        evals, *_ = tblg_bands
        assert np.all(np.isfinite(evals)), (
            f"Non-finite eigenvalues in TBG bands: "
            f"{np.sum(~np.isfinite(evals))} bad values."
        )

    def test_tblg_bands_correct_shape(self, tblg_bands, tblg_atoms):
        """Shape: (n_kpts, n_atoms) for dense diagonalization."""
        evals, _, _, k_dist, _ = tblg_bands
        n_kpts = len(k_dist)
        n_atoms = len(tblg_atoms)
        assert evals.shape == (n_kpts, n_atoms), (
            f"Expected shape ({n_kpts}, {n_atoms}), got {evals.shape}."
        )

    def test_tblg_bands_save_figure(self, tblg_bands):
        """Save dense TBG band-structure plot to tests/figures/."""
        evals, _, _, k_dist, k_node = tblg_bands
        angle_str = str(self.TWIST_ANGLE).replace(".", "p")
        outpath = os.path.join(
            _FIGURES_DIR,
            f"acsf_tblg_{angle_str}deg_bands_M{ACSF_M}_W{ACSF_W}.png",
        )
        _save_band_figure(
            evals, k_dist, k_node, _SYM_LABELS,
            title=f"TBG {self.TWIST_ANGLE}° — ACSF M={ACSF_M}, W={ACSF_W} (dense)",
            filepath=outpath,
            scatter=False,
            ylim=(self.EMIN, self.EMAX),
        )
        assert os.path.isfile(outpath), f"Figure not written to {outpath}"

    def test_tblg_bands_overlay_figure(self, tblg_bands, tblg_bands_sparse):
        """Overlay sparse (N_EIGS eigsh) on top of the full dense band structure.

        Both sets of eigenvalues share the same Fermi level reference so the
        comparison is direct.  Dense bands are drawn first (thin grey lines) and
        the sparse bands are overlaid as thicker coloured lines.
        """
        evals_dense, _, _, k_dist, k_node = tblg_bands
        evals_sparse, _, _ = tblg_bands_sparse

        angle_str = str(self.TWIST_ANGLE).replace(".", "p")
        outpath = os.path.join(
            _FIGURES_DIR,
            f"acsf_tblg_{angle_str}deg_overlay_M{ACSF_M}_W{ACSF_W}.png",
        )
        _save_overlay_figure(
            evals_dense, evals_sparse,
            k_dist, k_node, _SYM_LABELS,
            title=(
                f"TBG {self.TWIST_ANGLE}° — ACSF M={ACSF_M}, W={ACSF_W} "
                f"(dense vs sparse N={N_EIGS})"
            ),
            filepath=outpath,
            ylim=(self.EMIN, self.EMAX),
        )
        assert os.path.isfile(outpath), f"Overlay figure not written to {outpath}"

