import numpy as np 
import matplotlib.pyplot as plt 

theta = np.array([0.83,0.88,0.93, 0.99,1.05,1.08,1.12,1.16,1.2,1.47])
aa_sep = np.array([3.6013,3.6002,3.5989,3.5974,3.5956,3.5946,3.5526,3.5522,3.5517,3.5516])
ab_sep = np.array([3.4027,3.4028,3.4028,3.4028,3.4029,3.4029,3.4042,3.4048,3.4055,3.4110])

plt.plot(theta,aa_sep)
plt.xlabel(r"$\theta$")
plt.ylabel("layer sep at AA stacking")
plt.savefig("../figures/POD_best_fit_tblg_aa_sep.png")
plt.clf()
plt.plot(theta,ab_sep)
plt.xlabel(r"$\theta$")
plt.ylabel("layer sep at AB stacking")
plt.savefig("../figures/POD_best_fit_tblg_ab_sep.png")
plt.clf()
plt.plot(theta,aa_sep-ab_sep)
plt.xlabel(r"$\theta$")
plt.ylabel("corrugation")
plt.savefig("../figures/POD_best_fit_tblg_corrugation.png")