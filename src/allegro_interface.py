"""
Python Allegro calculator for BLG UQ workflows.

Loads a trained NequIP/Allegro checkpoint and exposes flat ``get_parameters`` /
``set_parameters`` for MCMC, plus batch evaluation and ASE relaxation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = (
    _REPO_ROOT
    / "uncertainty_quantification"
    / "initial_allegro_tests"
    / "allegro_blg_output"
)
# Small default model (~1760 trainable params; num_scalar_features=8, r_max=6 Å).
# The legacy ``best.ckpt`` in the same folder is a larger 32-feature model (~11k params).
_DEFAULT_CHECKPOINT = _DEFAULT_MODEL_DIR / "best-v2.ckpt"
_DEFAULT_CHECKPOINT_FALLBACKS = ("best-v2.ckpt", "best.ckpt")
_DEFAULT_R_MAX = 6.0


def checkpoint_tag(checkpoint_path: str | Path) -> str:
    """Stable short tag for cache / ensemble folder names."""
    return hashlib.sha256(str(Path(checkpoint_path).resolve()).encode()).hexdigest()[:12]


def _allegro_search_roots() -> List[Path]:
    roots = [
        _REPO_ROOT / "uncertainty_quantification" / "allegro_blg_rcut6",
        _REPO_ROOT / "uncertainty_quantification" / "initial_allegro_tests",
        _REPO_ROOT / "uncertainty_quantification",
        _REPO_ROOT,
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen and root.is_dir():
            seen.add(key)
            out.append(root)
    return out


def resolve_allegro_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve an Allegro ``.ckpt`` path (explicit or default trained model)."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"Allegro checkpoint not found: {p}")

    if _DEFAULT_CHECKPOINT.is_file():
        return _DEFAULT_CHECKPOINT.resolve()

    for root in _allegro_search_roots():
        # Root may itself be a model directory (e.g. allegro_blg_rcut6).
        search_dirs = [root, root / "allegro_blg_output", root / "allegro_blg_rcut6"]
        for out_dir in search_dirs:
            if not out_dir.is_dir():
                continue
            for name in _DEFAULT_CHECKPOINT_FALLBACKS:
                candidate = out_dir / name
                if candidate.is_file():
                    return candidate.resolve()
            candidates = sorted(out_dir.glob("best*.ckpt"))
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "No Allegro checkpoint found. Train one with "
        "uncertainty_quantification/fit_allegro_and_relax.py or "
        "uncertainty_quantification/initial_allegro_tests/fit_allegro.py, "
        "or pass allegro_checkpoint=..."
    )


def resolve_allegro_checkpoint_by_tag(tag: str) -> Path:
    """Find a checkpoint whose :func:`checkpoint_tag` matches *tag*."""
    tag = str(tag).strip().lower()
    for root in _allegro_search_roots():
        for ckpt in sorted(root.rglob("*.ckpt")):
            if checkpoint_tag(ckpt).lower() == tag:
                return ckpt.resolve()
    raise FileNotFoundError(
        f"No Allegro checkpoint with tag {tag!r} under {_allegro_search_roots()}"
    )


def allegro_bounds_from_params(
    params: np.ndarray,
    *,
    bound_scale: float = 1e2,
    min_half_width: float = 1e-6,
) -> np.ndarray:
    """Symmetric bounds around a reference weight vector."""
    p0 = np.asarray(params, dtype=float).ravel()
    half = float(bound_scale) * np.maximum(np.abs(p0), min_half_width)
    return np.column_stack([p0 - half, p0 + half])


class AllegroCalculator:
    """Checkpoint-backed Allegro model with flat parameter get/set."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        r_max: float = _DEFAULT_R_MAX,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = str(resolve_allegro_checkpoint(checkpoint_path))
        self.r_max = float(r_max)
        self._device = str(device)
        self._batch_atoms: Optional[List[Any]] = None
        self._param_names: List[str] = []
        self._param_shapes: List[Tuple[int, ...]] = []
        self._param_numel: List[int] = []
        self.results: Dict[str, Any] = {}
        self._torch = None
        self._wrapped = None
        self._ase_calc = None
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from nequip.data.transforms import (
            ChemicalSpeciesToAtomTypeMapper,
            NeighborListTransform,
        )
        from nequip.integrations.ase import NequIPCalculator
        from nequip.train import EMALightningModule
        from nequip.utils.global_state import set_global_state

        set_global_state()

        ckpt = Path(self.checkpoint_path)
        module = EMALightningModule.load_from_checkpoint(
            str(ckpt), map_location=self._device
        )
        module.eval()

        inner_model = module.model["sole_model"]
        inner_model.eval()

        class _GradModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self._m = model

            def forward(self, data):
                with torch.enable_grad():
                    return self._m(data)

        wrapped = _GradModel(inner_model)
        wrapped.eval()

        self._torch = torch
        self._wrapped = wrapped
        self._build_param_meta()

        transforms = [
            ChemicalSpeciesToAtomTypeMapper(
                model_type_names=["C"],
                chemical_species_to_atom_type_map={"C": "C"},
            ),
            NeighborListTransform(r_max=self.r_max),
        ]
        self._ase_calc = NequIPCalculator(
            model=wrapped, device=self._device, transforms=transforms
        )

    def _build_param_meta(self) -> None:
        self._param_names = []
        self._param_shapes = []
        self._param_numel = []
        for name, param in self._wrapped.named_parameters():
            if not param.requires_grad:
                continue
            self._param_names.append(name)
            self._param_shapes.append(tuple(int(s) for s in param.shape))
            self._param_numel.append(int(param.numel()))

    @property
    def nparams(self) -> int:
        return int(sum(self._param_numel))

    def get_parameters(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        named = dict(self._wrapped.named_parameters())
        for name in self._param_names:
            parts.append(named[name].detach().cpu().numpy().ravel())
        return np.concatenate(parts).astype(np.float64)

    def set_parameters(self, params: Sequence[float] | np.ndarray) -> None:
        flat = np.asarray(params, dtype=np.float64).ravel()
        expected = self.nparams
        if flat.size != expected:
            raise ValueError(f"Allegro: expected {expected} params, got {flat.size}")
        offset = 0
        named = dict(self._wrapped.named_parameters())
        for name, shape, n in zip(self._param_names, self._param_shapes, self._param_numel):
            chunk = flat[offset : offset + n]
            tensor = named[name]
            with self._torch.no_grad():
                tensor.copy_(
                    self._torch.as_tensor(
                        chunk.reshape(shape),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )
            offset += n

    def close(self) -> None:
        """Release references (no persistent subprocess)."""
        self._ase_calc = None
        self._wrapped = None

    # ── ASE-compatible single-structure API ───────────────────────────────────

    def calculate(
        self,
        atoms=None,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes=...,
    ) -> None:
        from ase.calculators.calculator import all_changes

        if system_changes is ...:
            system_changes = all_changes
        if atoms is None:
            raise ValueError("AllegroCalculator.calculate requires atoms")
        atoms.calc = self._ase_calc
        self.results = {
            "energy": float(atoms.get_potential_energy()),
            "forces": np.asarray(atoms.get_forces(), dtype=float),
        }

    def get_potential_energy(self, atoms=None, force_consistent=None) -> float:
        self.calculate(atoms, properties=["energy"])
        return float(self.results["energy"])

    def get_forces(self, atoms=None) -> np.ndarray:
        self.calculate(atoms, properties=["forces"])
        return np.asarray(self.results["forces"], dtype=float)

    # ── Batch evaluation (MCMC) ───────────────────────────────────────────────

    def prepare_batch(self, atoms_list, batch_dir: Optional[str] = None) -> None:
        self._batch_atoms = list(atoms_list)

    def evaluate_batch(
        self,
        atoms_list: Optional[Sequence[Any]] = None,
        params: Optional[Sequence[float] | np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if params is not None:
            self.set_parameters(params)
        if atoms_list is None:
            atoms_list = self._batch_atoms
        if atoms_list is None:
            raise RuntimeError(
                "prepare_batch() must be called before evaluate_batch() "
                "when atoms_list is not supplied."
            )

        energies: List[float] = []
        forces_list: List[np.ndarray] = []
        for atoms in atoms_list:
            try:
                struct = atoms.copy()
                struct.calc = self._ase_calc
                e = float(struct.get_potential_energy())
                f = np.asarray(struct.get_forces(), dtype=float)
            except Exception:
                e = float("nan")
                f = np.full((len(atoms), 3), np.nan)
            energies.append(e)
            forces_list.append(f)
        return np.asarray(energies, dtype=float), forces_list

    # ── Relaxation ────────────────────────────────────────────────────────────

    def relax_structure(
        self,
        atoms,
        relax_backend: str = "ase",
        etol: float = 1e-10,
        ftol: float = 1e-10,
        maxiter: int = 1000,
        maxeval: int = 8000,
    ):
        from blg_model_builder.potentials import (
            _annotate_relax_energies,
            _normalize_relax_backend,
            _relax_structure_ase,
            _reraise_relax_failed,
        )

        backend = _normalize_relax_backend(relax_backend)
        if backend != "ase":
            raise ValueError(
                "AllegroCalculator only supports relax_backend='ase' "
                f"(got {relax_backend!r})."
            )
        atoms = atoms.copy()
        atoms.calc = self._ase_calc
        try:
            relaxed = _relax_structure_ase(
                atoms,
                etol=etol,
                ftol=ftol,
                maxiter=maxiter,
                maxeval=maxeval,
            )
        except Exception as exc:
            _reraise_relax_failed(exc, "AllegroCalculator.relax_structure (ASE FIRE)")
        return _annotate_relax_energies(relaxed, self)


AllegroASECalculator = AllegroCalculator
