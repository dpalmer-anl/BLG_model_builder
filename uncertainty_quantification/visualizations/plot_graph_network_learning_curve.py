import numpy as np 
import matplotlib.pyplot as plt 

x_l0 = np.array([10.043457338758873, 25.293747725022186, 49.966822499934196, 98.28042895037582,
                247.5124145332373, 497.50619547221476, 741.4066842920248, 995.6730698112132, 1285.9612197790136])
y_l0 = np.array([0.13415258859863702, 0.12425456309597355, 0.10796592515883574, 0.10762167928072827,
                0.0911545933668571, 0.08828950772125337, 0.08469909963872876, 0.0860624142977465, 0.07895251625258819])

x = np.geomspace(np.min(x_l0), np.max(x_l0), 100)
coeffs_l0 = np.polyfit(np.log10(x_l0), np.log10(y_l0), 1)
y = 10 ** (coeffs_l0[0] * np.log10(x) + coeffs_l0[1])
plt.plot(x, y, label=f'Artificial Neural Network potential, slope = {coeffs_l0[0]:.3f}')

x_l3 = np.array([10.043457338758873, 24.858803761777704, 50.18396501314263, 99.56730698112132,
                252.93747725022197, 499.66822499934244, 735.0045352199567, 991.3648619472849,
                1297.162396146531])

y_l3 = np.array([0.09411265399860032, 0.06581274997637311, 0.055211302533391515, 0.04572968005561136,
                0.032286503784218476, 0.038119063746759656, 0.028143773648939857, 0.02823379619958009, 0.023088284233158528])

x = np.geomspace(np.min(x_l3), np.max(x_l3), 100)
coeffs_l3 = np.polyfit(np.log10(x_l3), np.log10(y_l3), 1)
y = 10 ** (coeffs_l3[0] * np.log10(x) + coeffs_l3[1])
plt.plot(x, y, label=f'Equivariant Graph Network potential, slope = {coeffs_l3[0]:.3f}')

plt.scatter(x_l0, y_l0)
plt.scatter(x_l3, y_l3)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("number of training frames")
plt.ylabel("Force MAE [eV/Å]")
plt.legend(loc='upper right')
plt.savefig('../figures/graph_network_learning_curve.png', dpi=300, bbox_inches='tight')
