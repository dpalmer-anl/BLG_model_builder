import numpy as np
import matplotlib.pyplot as plt

def solve_poly(x,y):
    degree = len(y)
    exponents = -1*np.arange(degree)

    D = np.power(x[:,np.newaxis],exponents[np.newaxis,:])
    theta = np.linalg.solve(D,y)
    return theta

def poly_eval(x,theta):
    degree = len(theta)
    exponents = -1*np.arange(degree)
    D = np.power(x[:,np.newaxis],exponents[np.newaxis,:])
    return np.sum(D * theta,axis=1)

def rbf(w,a,xtrain,x):
    dist = xtrain - x
    psi = np.exp(-a*dist**2)
    y = w * psi
    return y
    
hopping_data = np.loadtxt("../data/popov_hopping_pp_sigma.txt")
x = hopping_data[:,0]
y = hopping_data[:,1]

n_params = len(y)
n_samples = 1 #2000

param_ensemble = np.zeros((n_samples,n_params))
y_fit = np.zeros((n_samples,len(y)))
for i in range(n_samples):
    choose_ind = np.random.choice(range(len(y)),n_params, replace=False)
    x_sub = x[choose_ind]
    y_sub = y[choose_ind]
    param_ensemble[i,:] = solve_poly(x_sub,y_sub)
    y_fit[i,:] = poly_eval(x,param_ensemble[i,:])

mean_yfit = np.mean(y_fit,axis=0)
std_yfit = np.std(y_fit,axis=0)

plt.scatter(x,y,label="y abinitio")
plt.plot(x,mean_yfit,label="ensemble fit",c="red")
plt.fill_between(x,mean_yfit-std_yfit,mean_yfit+std_yfit,alpha=0.3)
plt.legend()
plt.ylim((np.min(y),np.max(y)))
plt.savefig("poly_ensemble_fit.png")
plt.clf()


