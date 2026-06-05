
import pickle
import numpy as np, scipy.sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flatgraphene as fg
from blg_model_builder.tb_descriptors import get_acsf_hopping_descriptors
from blg_model_builder.tb_models import get_acsf_hoppings, get_recip_cell, k_path
from blg_model_builder.ensemble_io import resolve_ensemble_pickle

# unrelaxed 2.88° structure
theta = 2.88
p, q, _ = fg.twist.find_p_q(theta, a_tol=0.01)
atoms = fg.twist.make_graphene(
    cell_type="hex", n_layer=2, p=p, q=q, lat_con=2.46,
    sym=["C", "C"], mass=[12.01, 12.01], sep=3.35, h_vac=20)
print(f"  {len(atoms)} atoms")

# optimal-T ensemble → posterior mean as best-fit params
pkl, T = resolve_ensemble_pickle("ACSF_hoppings_M_14_W_5", "ensembles", None,
    calibration_metrics_dir="calibration_metrics", calibration_target="hopping")
print(f"  T={T}  pkl={pkl}")
with open(pkl, "rb") as fh:
    d = pickle.load(fh)
params = np.mean(d["ensemble"]["hopping"], axis=0)

# k-path K -> Gamma -> M -> K (denser)
kvec_red, k_dist, k_node = k_path([[1/3,2/3,0],[0,0,0],[1/2,0,0],[1/3,2/3,0]], 40)
kvec_cart = kvec_red @ get_recip_cell(np.array(atoms.get_cell()).T)

# ACSF tight-binding, dense diagonalization
desc, (pi, pj, pv) = get_acsf_hopping_descriptors(atoms, M=14, W=5, r_cut=6.0)
hop = get_acsf_hoppings(desc, params)
N = len(atoms)
ev = []
for kc in kvec_cart:
    H = scipy.sparse.coo_matrix(
        (hop * np.exp(1j * (pv @ kc)), (pi, pj)),
        shape=(N, N), dtype=complex).tocsr()
    H = H + H.conj().T
    ev.append(np.linalg.eigh(H.toarray())[0])
evals = np.array(ev)
evals -= (evals[0, N//2] + evals[0, N//2 - 1]) / 2   # shift E_F to 0

# scatter plot
fig, ax = plt.subplots(figsize=(6, 5))
for b in range(evals.shape[1]):
    if np.any(np.abs(evals[:, b]) < 0.5):
        ax.plot(k_dist, evals[:, b], color="steelblue", lw=0.8, alpha=0.7)
ax.axhline(0, color="red", ls="--", lw=0.9, zorder=3)
for xv in k_node:
    ax.axvline(xv, color="k", ls="--", lw=0.6)
ax.set_xlim(k_dist[0], k_dist[-1]); ax.set_ylim(-0.5, 0.5)
ax.set_xticks(k_node); ax.set_xticklabels(["K", "G", "M", "K"], fontsize=12)
ax.set_ylabel("Energy (eV)", fontsize=12)
ax.set_title(f"TBLG {theta}° – ACSF M=14 W=5 (posterior mean, unrelaxed)", fontsize=10)
fig.tight_layout()
fig.savefig(f"bands_{theta}deg_bestfit.png", dpi=150, bbox_inches="tight")
print(f"Saved: bands_{theta}deg_bestfit.png")
