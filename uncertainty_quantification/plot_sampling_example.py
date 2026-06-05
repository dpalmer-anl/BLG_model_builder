import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-5,5,300)

std = 1
y_mcmc = np.exp(-x**2/std**2)/np.sqrt(2*np.pi*std**2)
std = 0.2
y_sub = 0.3*np.exp(-x**2/std**2)/np.sqrt(2*np.pi*std**2)
plt.plot(x,y_mcmc,label="MCMC")
plt.plot(x,y_sub,label="Subsampling")
plt.plot(0.7*np.ones(3),np.linspace(0,np.max(y_sub),3),label="True y value")
plt.xlabel(r"$y_{pred}$")
plt.ylabel(r"$P(y_{pred})$")
plt.legend()
plt.savefig("figures/posterior_predictive_distribution_example.png")
plt.show()  