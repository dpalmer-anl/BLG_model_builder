#!/usr/bin/env python3
"""Fit continuum LJ 9-3 (Steele) ε, σ for graphene on an hBN bulk continuum.

Potential (substrate plane at z=sh)::

    E(z) = (4π/3) ε σ³ q [ (1/15)(σ/z)^9 − (1/2)(σ/z)^3 ]

Targets
-------
* equilibrium height z0 = 3.35 Å  (from substrate plane)
* binding energy E(z0) = −0.0415 eV / atom

Number density q is the hBN continuum density from
``example_lj_continuum_substrate.in`` (0.077495 Å^-3).
"""
from __future__ import annotations

import math

Z0 = 3.35  # Å
E_BIND = 0.0415  # eV/atom (positive magnitude)
HBN_Q = 0.077495  # Å^-3


def sigma_from_zeq(z0: float = Z0) -> float:
    """σ such that dE/dz = 0 at z=z0 for continuum LJ 9-3.

    From F_z ∝ 0:
        (9/15) r^{-10} = (3/2) r^{-4}
        r^6 = 2/5
        r = z/σ = (2/5)^{1/6}
        σ = z0 * (5/2)^{1/6}
    """
    return float(z0 * (2.5) ** (1.0 / 6.0))


def energy_bracket(r: float) -> float:
    """Dimensionless factor [ (1/15) r^{-9} − (1/2) r^{-3} ]."""
    return (1.0 / 15.0) * r ** (-9) - 0.5 * r ** (-3)


def epsilon_from_binding(
    z0: float = Z0,
    e_bind: float = E_BIND,
    q: float = HBN_Q,
    sigma: float | None = None,
) -> float:
    """ε (eV) so that E(z0) = −e_bind."""
    if sigma is None:
        sigma = sigma_from_zeq(z0)
    r = z0 / sigma
    bracket = energy_bracket(r)  # negative at equilibrium
    # E = (4π/3) ε σ³ q * bracket = -e_bind
    prefactor = (4.0 * math.pi / 3.0) * (sigma ** 3) * q
    eps = -e_bind / (prefactor * bracket)
    return float(eps)


def continuum_energy(z: float, eps: float, sigma: float, q: float, sh: float = 0.0) -> float:
    dz = z - sh
    r = dz / sigma
    return (4.0 * math.pi / 3.0) * eps * (sigma ** 3) * q * energy_bracket(r)


def continuum_force_z(z: float, eps: float, sigma: float, q: float, sh: float = 0.0) -> float:
    """Fz matching the LAMMPS example (positive = +z direction on the atom)."""
    dz = z - sh
    r = dz / sigma
    inner = (-9.0 / 15.0) * (1.0 / sigma) * r ** (-10) + 1.5 * (1.0 / sigma) * r ** (-4)
    return (-4.0 * math.pi / 3.0) * eps * (sigma ** 3) * q * inner


def main():
    q = HBN_Q
    sigma = sigma_from_zeq()
    eps = epsilon_from_binding(q=q, sigma=sigma)
    e0 = continuum_energy(Z0, eps, sigma, q)
    f0 = continuum_force_z(Z0, eps, sigma, q)
    f_m = continuum_force_z(Z0 - 1e-4, eps, sigma, q)
    f_p = continuum_force_z(Z0 + 1e-4, eps, sigma, q)
    print(f"q     = {q:.8f} 1/A^3  (hBN bulk)")
    print(f"sigma = {sigma:.8f} A")
    print(f"eps   = {eps:.8f} eV")
    print(f"E(z0) = {e0:.8f} eV  (target {-E_BIND})")
    print(f"F(z0) = {f0:.6e} eV/A  (target ~0)")
    print(f"F either side: {f_m:.6e}, {f_p:.6e}")
    print(f"example sigma was 2.9845; ours {sigma:.4f}")
    print(f"example eps was 0.0415; ours {eps:.6f} (eps is not binding energy for continuum)")


if __name__ == "__main__":
    main()
