#!/usr/bin/env python3
"""Count Allegro trainable params vs num_scalar / num_tensor features."""
from __future__ import annotations

import torch
from allegro.model import AllegroModel
from nequip.utils.global_state import set_global_state

set_global_state()


def build_model(
    *,
    num_scalar_features: int,
    num_tensor_features: int,
    l_max: int = 1,
    parity: bool = True,
    num_layers: int = 1,
    num_bessels: int = 8,
    mlp_width: int | None = None,
    readout_width: int | None = None,
    r_max: float = 6.0,
) -> torch.nn.Module:
    """Match fit_allegro.py defaults (mlp widths default to num_scalar_features)."""
    w = num_scalar_features if mlp_width is None else int(mlp_width)
    rw = num_scalar_features if readout_width is None else int(readout_width)
    model = AllegroModel(
        seed=0,
        model_dtype="float32",
        type_names=["C"],
        r_max=r_max,
        radial_chemical_embed={
            "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
            "num_bessels": num_bessels,
            "bessel_trainable": False,
            "polynomial_cutoff_p": 6,
        },
        l_max=l_max,
        parity=parity,
        num_layers=num_layers,
        num_scalar_features=num_scalar_features,
        num_tensor_features=num_tensor_features,
        allegro_mlp_hidden_layers_width=w,
        readout_mlp_hidden_layers_width=rw,
        avg_num_neighbors=12.0,
        per_type_energy_shifts=0.0,
        per_type_energy_scales=1.0,
    )
    if not isinstance(model, torch.nn.Module) and callable(model):
        model = model()
    return model


def n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sweep(*, mlp_mode: str, max_ns: int = 32, max_nt: int = 64) -> list[tuple[int, int, int]]:
    """Return (ns, nt, n_params) with n_params < 1000.

    ``num_scalar_features`` must be even (Allegro embedding constraint).
    """
    feasible: list[tuple[int, int, int]] = []
    print(f"\n=== mlp_mode={mlp_mode}  (n_params < 1000) ===")
    print(f"{'ns':>4} {'nt':>4} {'n_params':>10}")
    for ns in range(2, max_ns + 1, 2):  # even only
        for nt in range(1, max_nt + 1):
            kw = dict(num_scalar_features=ns, num_tensor_features=nt)
            if mlp_mode == "fixed4":
                kw["mlp_width"] = 4
                kw["readout_width"] = 4
            # else: widths track num_scalar_features (fit_allegro default)
            try:
                n = n_trainable(build_model(**kw))
            except Exception as exc:  # noqa: BLE001
                print(f"{ns:4d} {nt:4d}  FAILED: {exc}")
                continue
            if n < 1000:
                feasible.append((ns, nt, n))
                print(f"{ns:4d} {nt:4d} {n:10d}")
    return feasible


def summarize(feasible: list[tuple[int, int, int]], label: str) -> None:
    if not feasible:
        print(f"\n[{label}] no feasible (ns, nt)")
        return
    by_ns = max(feasible, key=lambda t: (t[0], t[1], -t[2]))
    by_nt = max(feasible, key=lambda t: (t[1], t[0], -t[2]))
    by_prod = max(feasible, key=lambda t: (t[0] * t[1], -t[2], t[0], t[1]))
    # For each ns, max nt still under budget
    print(f"\n[{label}] summary (n_params < 1000):")
    print(f"  max num_scalar_features: {by_ns[0]}  (nt={by_ns[1]}, n={by_ns[2]})")
    print(f"  max num_tensor_features: {by_nt[1]}  (ns={by_nt[0]}, n={by_nt[2]})")
    print(f"  max ns*nt:               ns={by_prod[0]} nt={by_prod[1]} n={by_prod[2]}")
    print("  frontier (max nt for each ns):")
    from collections import defaultdict

    best_nt: dict[int, tuple[int, int]] = {}
    for ns, nt, n in feasible:
        if ns not in best_nt or nt > best_nt[ns][0]:
            best_nt[ns] = (nt, n)
    for ns in sorted(best_nt):
        nt, n = best_nt[ns]
        print(f"    ns={ns:2d} → max nt={nt:2d}  (n_params={n})")


def main() -> None:
    print("Sanity checks (mlp_width = num_scalar_features):")
    for ns, nt, note in [
        (8, 8, "fit_allegro comment ~1760"),
        (4, 8, "current allegro_blg_output config"),
        (4, 4, "small"),
    ]:
        n = n_trainable(build_model(num_scalar_features=ns, num_tensor_features=nt))
        print(f"  ns={ns:3d} nt={nt:3d}  n_params={n:6d}  ({note})")

    # Also with fixed width=4 like the saved config.yaml
    n = n_trainable(
        build_model(
            num_scalar_features=4,
            num_tensor_features=8,
            mlp_width=4,
            readout_width=4,
        )
    )
    print(f"  ns=  4 nt=  8  n_params={n:6d}  (mlp/readout width fixed=4, as in config.yaml)")

    f1 = sweep(mlp_mode="track_ns", max_ns=16, max_nt=48)
    summarize(f1, "mlp_width = num_scalar_features")

    f2 = sweep(mlp_mode="fixed4", max_ns=16, max_nt=48)
    summarize(f2, "mlp_width = readout_width = 4")


if __name__ == "__main__":
    main()
