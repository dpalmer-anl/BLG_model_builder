"""
mlp_model.py — PyTorch MLP for POD-descriptor-based energy/force prediction.

Architecture
------------
Global MLP:  E_total = NN(D_1, ..., D_nd)

  - Input:  global POD descriptors  D ∈ R^n_desc  (summed over all atoms)
  - Output: total potential energy  E ∈ R  (eV)

Force computation via the chain rule (no finite differences):

    F_{n,m} = -Σ_α  (∂E / ∂D_α)  ×  (∂D_α / ∂r_{n,m})

where ∂E/∂D_α is from PyTorch autograd and ∂D_α/∂r is the pre-cached
LAMMPS analytical Jacobian (see pod_descriptor.py).  This exactly mirrors
the force formula in mlpod.cpp:

    DGEMV('N', nforce, nd1234, gdd, c1, force)

with c1[α] = ∂E/∂D_α from autograd instead of fixed linear coefficients.

Classes
-------
MLP          — configurable feed-forward network (hidden layers + activation)
MLPPODModel  — wrapper: differentiable forward + force computation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

DTYPE = torch.float64


# ── Activation helpers ────────────────────────────────────────────────────────

def _silu_numpy(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _silu_numpy_deriv(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def _get_activation(name: str):
    """Return a torch activation module and its numpy equivalent."""
    name = name.lower()
    if name == "silu":
        return nn.SiLU(), _silu_numpy, _silu_numpy_deriv
    elif name == "tanh":
        return nn.Tanh(), np.tanh, lambda x: 1.0 - np.tanh(x) ** 2
    elif name == "relu":
        return nn.ReLU(), lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose: silu, tanh, relu")


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Configurable feed-forward network for total energy prediction.

    Parameters
    ----------
    input_dim  : int   — number of input features (= n_desc)
    hidden_dim : int   — width of each hidden layer  (default 30)
    n_layers   : int   — number of hidden layers     (default 2)
    activation : str   — "silu" | "tanh" | "relu"    (default "silu")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 30,
        n_layers: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.activation_name = activation

        act_module, self._act_np, self._act_np_d = _get_activation(activation)

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=DTYPE))
            layers.append(act_module.__class__())   # fresh instance per layer
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1, dtype=DTYPE))
        self.net = nn.Sequential(*layers)

        # Initialize weights with Xavier uniform, biases to zero
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (n_desc,) → E: scalar tensor."""
        return self.net(x).squeeze(-1)

    # ── Flat parameter helpers ────────────────────────────────────────────

    def get_flat_params(self) -> np.ndarray:
        """Return all parameters as a 1-D numpy array."""
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.parameters()]
        )

    def set_flat_params(self, params: np.ndarray) -> None:
        """Load all parameters from a 1-D numpy array (in-place)."""
        params = np.asarray(params, dtype=np.float64)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(
                    torch.tensor(
                        params[offset:offset + n].reshape(p.shape), dtype=DTYPE
                    )
                )
                offset += n

    # ── Fast numpy forward (for MCMC — no torch overhead) ────────────────

    def _extract_numpy_weights(self):
        """Extract (W, b) pairs for each Linear layer as numpy arrays."""
        wb = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                wb.append(
                    (
                        m.weight.detach().cpu().numpy(),   # (out, in)
                        m.bias.detach().cpu().numpy(),     # (out,)
                    )
                )
        return wb

    def forward_numpy(self, D: np.ndarray) -> float:
        """Pure numpy forward pass — no torch overhead, for fast MCMC evaluation.

        Parameters
        ----------
        D : np.ndarray, shape (n_desc,)

        Returns
        -------
        float  — predicted total energy
        """
        x = np.asarray(D, dtype=np.float64)
        wb = self._extract_numpy_weights()
        for i, (W, b) in enumerate(wb):
            x = W @ x + b
            if i < len(wb) - 1:           # hidden layers get activation
                x = self._act_np(x)
        return float(x.squeeze())

    def forward_numpy_batch(self, D_batch: np.ndarray) -> np.ndarray:
        """Vectorized numpy forward for a batch of descriptor vectors.

        Parameters
        ----------
        D_batch : np.ndarray, shape (n_struct, n_desc)

        Returns
        -------
        np.ndarray, shape (n_struct,)
        """
        x = np.asarray(D_batch, dtype=np.float64)   # (n_struct, n_desc)
        wb = self._extract_numpy_weights()
        for i, (W, b) in enumerate(wb):
            x = x @ W.T + b                          # (n_struct, out_dim)
            if i < len(wb) - 1:
                x = self._act_np(x)
        return x.ravel()

    def forward_numpy_batch_with_params(
        self,
        D_batch: np.ndarray,
        flat_params: np.ndarray,
    ) -> np.ndarray:
        """Vectorized numpy forward using an *external* flat parameter vector.

        Used during MCMC: avoids modifying the model's state for each proposal.

        Parameters
        ----------
        D_batch    : (n_struct, n_desc)
        flat_params: (n_params,)

        Returns
        -------
        (n_struct,)
        """
        # Rebuild weight matrices from flat_params
        shapes = [(p.shape, p.numel()) for p in self.parameters()]
        wb = []
        offset = 0
        for shape, n in shapes:
            arr = flat_params[offset:offset + n].reshape(shape)
            offset += n
            wb.append(arr)

        # wb is now [W1, b1, W2, b2, ..., Wout, bout]
        linear_wb = list(zip(wb[::2], wb[1::2]))   # pair (W, b)

        x = np.asarray(D_batch, dtype=np.float64)
        for i, (W, b) in enumerate(linear_wb):
            x = x @ W.T + b
            if i < len(linear_wb) - 1:
                x = self._act_np(x)
        return x.ravel()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── MLPPODModel ───────────────────────────────────────────────────────────────

class MLPPODModel:
    """Wrapper combining an MLP with force computation via LAMMPS Jacobians.

    Parameters
    ----------
    mlp : MLP
    n_atoms : int   — number of atoms per structure (must be fixed)
    """

    def __init__(self, mlp: MLP, n_atoms: int) -> None:
        self.mlp     = mlp
        self.n_atoms = n_atoms

    # ── Differentiable forward (used in training) ─────────────────────────

    def forward_from_descriptors(self, D: torch.Tensor) -> torch.Tensor:
        """Differentiable energy prediction.

        Parameters
        ----------
        D : torch.Tensor, shape (n_desc,)   — global descriptors

        Returns
        -------
        E : torch.Tensor, scalar
        """
        return self.mlp(D)

    # ── Force computation via chain rule + LAMMPS Jacobian ────────────────

    def compute_forces(
        self,
        D_numpy: np.ndarray,
        J_numpy: np.ndarray,
    ) -> np.ndarray:
        """Compute forces using PyTorch autograd + pre-cached LAMMPS Jacobian.

        Formula (mirrors mlpod.cpp DGEMV at line 689):

            F_{n,m} = -Σ_α  (∂E / ∂D_α)  ×  (∂D_α / ∂r_{n,m})
                    = -(J^T @ dE_dD)

        where:
            J[α, nm]   = ∂D_α / ∂r_{n,m}  (from LAMMPS unit-vector evaluations)
            dE_dD[α]   = ∂E / ∂D_α         (from PyTorch autograd)

        Parameters
        ----------
        D_numpy : np.ndarray, shape (n_desc,)
            Global descriptors for this structure.
        J_numpy : np.ndarray, shape (n_desc, n_atoms*3)
            Analytical Jacobian ∂D/∂r from ``PODDescriptorCalculator``.

        Returns
        -------
        forces : np.ndarray, shape (n_atoms, 3)  in eV/Å
        """
        D_t = torch.tensor(D_numpy, dtype=DTYPE, requires_grad=True)
        E   = self.mlp(D_t)
        dE_dD = torch.autograd.grad(E, D_t)[0].detach().numpy()  # (n_desc,)

        # F = -(J^T @ dE_dD)  →  shape (n_atoms*3,)
        F_flat = -(J_numpy.T @ dE_dD)
        return F_flat.reshape(self.n_atoms, 3)
