import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
import scipy.special
csfont = {'fontname':'serif',"size":13}
model_names =["TETB interlayer", "TETB intralayer", "Classical Potential interlayer","Classical Potential intralayer", "TB Moon-Koshino",
             "Local Environment TB interlayer","Local Environment TB intralayer 1","Local Environment TB intralayer 2","Local Environment TB intralayer 3",
             "Multi-Layer Perceptron TB"]
uq_types = ["SubSampling","MCMC"]
mcmc_opt_ks_metric = [0.07,0.1145746031746032,0.04765593203093198,0.10010934744268075, 0.08112289947384171,0.05843237399343869,0.2542272436789982]
cv_opt_ks_metric = [0.3,0.4050833333333333,0.08579365079365081,0.12319047619047614,np.nan, 0.31132567849686854,0.3335416666666667]
opt_ks_metric = np.vstack((cv_opt_ks_metric,mcmc_opt_ks_metric)).T
df = pd.DataFrame({
    'Model': np.repeat(model_names, len(uq_types)),
    'uq type': np.tile(uq_types, len(model_names)),
    r'optimized $|\mathcal{M}|$': opt_ks_metric.flatten()
})
sns.set(font_scale=1.1)
plt.figure(figsize=(8, 6))
sns.barplot(data=df, y="Model", x="optimized KS metric", hue="uq type", palette='viridis',orient="y")

# Customize plot
#plt.title('')
plt.legend(fontsize=13)
#plt.ylabel('Model',**csfont)
plt.xlabel('optimized KS metric',**csfont)
plt.tight_layout()
plt.savefig("figures/optimized_ks_metric_model.png")
plt.clf()