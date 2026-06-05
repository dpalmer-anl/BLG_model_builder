import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["TETB interlayer","TETB intralayer","Kolmogorov-Crespi","REBO", "LETB","Moon-Koshino"]
miscal_opt_vals_subsamp = np.array([0.98,0.98,0.04,0.86,0.88,0.96])
miscal_opt_vals_mcmc = np.array([0.21,0.8,0.02,0.27,0.23,0.42])
xlabel = r"$|\mathcal{M}_{opt}|$"
ylabel = "Model"

# Backwards/typo compatibility with earlier variable name in notes
miscal_opt_vals_subamp = miscal_opt_vals_subsamp

sns.set_theme(style="whitegrid", context="talk")

# Build "long-form" arrays for seaborn without adding pandas as a dependency
models = np.array(labels)
x = np.concatenate([miscal_opt_vals_subamp, miscal_opt_vals_mcmc])
y = np.concatenate([models, models])
hue = np.array(["Subsample"] * len(models) + ["MCMC"] * len(models))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=x,
    y=y,
    hue=hue,
    orient="h",
    order=labels,  # preserve desired model ordering
    ax=ax,
)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.legend(title="", loc="lower right", frameon=True)
fig.tight_layout()
plt.savefig("figures/opt_miscal_ensemble_bar.png")
plt.clf()
