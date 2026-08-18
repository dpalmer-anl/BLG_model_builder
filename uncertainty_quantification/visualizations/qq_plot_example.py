import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee
from scipy import stats

np.random.seed(0)

# ----------------------------
# Simulate "true" data
# ----------------------------
Ndata = 200
x = np.linspace(0, 1, Ndata)

# True function
f_true = np.sin(2 * np.pi * x)

# True noise
sigma_true = 0.2
y_true = f_true + np.random.normal(0, sigma_true, size=Ndata)

# ----------------------------
# Simulate posterior predictive samples
# ----------------------------
Nsamples = 500

# Case 1: WELL CALIBRATED
y_pred_samples = (
    f_true[:, None] +
    np.random.normal(0, sigma_true, size=(Ndata, Nsamples))
)

# Uncomment this to see MIS-CALIBRATION (too confident)
# y_pred_samples = (
#     f_true[:, None] +
#     np.random.normal(0, 0.05, size=(Ndata, Nsamples))  # too small variance
# )

# ----------------------------
# Compute PIT values
# ----------------------------
# u = np.mean(y_pred_samples <= y_true[:, None], axis=1)

# # ----------------------------
# # QQ plot vs Uniform(0,1)
# # ----------------------------
# u_sorted = np.sort(u)
# uniform_quantiles = np.linspace(0, 1, Ndata)

# plt.figure()
# plt.plot(uniform_quantiles, u_sorted, 'o', label='PIT QQ')
# plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
# plt.xlabel('Uniform quantiles')
# plt.ylabel('Empirical quantiles')
# plt.title('QQ plot for calibration')
# plt.legend()
# plt.show()

x = np.linspace(1,4,100)
A6 = -5.4
A10 = 9.1
A12 = 8.4
A18 = -1.9
A24 =  -1
Y_true = A6*np.power(x,-6) + A10*np.power(x,-10) + A12*np.power(x,-12) + A18*np.power(x,-18) + A24* np.sin(x)*np.exp(-x/2) #*np.power(x,-24)

def get_ypred(x,a,b,a_exp,b_exp):
    #a,b,a_exp,b_exp = params
    return a*np.power(x,a_exp) + b*np.power(x,b_exp)

def get_ypred_vect(x,params):
    a,b,a_exp,b_exp = params
    return a*np.power(x,a_exp) + b*np.power(x,b_exp)

params, cov = curve_fit(get_ypred, x, Y_true, p0=[-5.4,8.4,-6,-12])
Y_pred = get_ypred(x,*params)
C0 = np.sum((Y_true-Y_pred)**2)
print((np.sqrt(C0))/len(Y_true))
T0 = C0/4

def get_log_prob(params,x,y,T):
    ypred = get_ypred_vect(x,params)
    if np.any(params <-1e2 ) or np.any(params > 1e2):
        return -np.inf
    cost = np.sum((y-ypred)**2)
    return -0.5*cost/T

nwalkers  = 40
ndim = 4
nsteps = 1000

T = 2 * T0
p0 = params + 1e-2*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, get_log_prob, args=[x, Y_true, T])
sampler.run_mcmc(p0, nsteps)

samples = sampler.get_chain(flat=True)
acceptance_fraction = sampler.acceptance_fraction
print("Mean acceptance fraction: {:.8f}".format(np.mean(acceptance_fraction)))
print(np.mean(samples, axis=0))

Ypred_posterior = np.array([get_ypred_vect(x,samples[i]) for i in range(len(samples))])
MAE_dist = (np.mean(Ypred_posterior, axis=0) - Y_true)

u = np.mean(Ypred_posterior.T <= Y_true[:, None], axis=1)

TW_array = np.linspace(-1,0.1,10)
TW_array = 10**TW_array
ks_stat_array = np.zeros(len(TW_array))
miscal_area_stat = np.zeros(len(TW_array))
for i,TW in enumerate(TW_array):
    T = TW * T0
    p0 = params + 1e-2*np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, get_log_prob, args=[x, Y_true, T])
    sampler.run_mcmc(p0, nsteps)

    samples = sampler.get_chain(flat=True)

    Ypred_posterior = np.array([get_ypred_vect(x,samples[i]) for i in range(len(samples))])
    ks_stat_array[i] = stats.ks_2samp(Y_true, Ypred_posterior.flatten()).statistic

    u = np.mean(Ypred_posterior.T <= Y_true[:, None], axis=1)
    miscal_area_stat[i] = np.trapezoid(np.abs(u - np.linspace(0,1,len(u))),np.linspace(0,1,len(u)))

plt.plot(np.log10(TW_array), ks_stat_array,label='KS statistic')
plt.plot(np.log10(TW_array), miscal_area_stat,label='Miscalibration area')
plt.xlabel('log10(Temperature)')
plt.ylabel('KS statistic')
plt.title('KS statistic vs Temperature')
plt.legend()
plt.show()

# ----------------------------
# QQ plot vs Uniform(0,1)
# ----------------------------
u_sorted = np.sort(u)
uniform_quantiles = np.linspace(0, 1, len(Y_true))

plt.figure()
plt.plot(uniform_quantiles, u_sorted, 'o', label='PIT QQ')
plt.fill_between(uniform_quantiles, u_sorted, np.linspace(0,1,len(u_sorted)), alpha=0.5)
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('Uniform quantiles')
plt.ylabel('Empirical quantiles')
plt.title('QQ plot for calibration')
plt.legend()
plt.savefig('../figures/qq_plot_example.png')
plt.clf()



# ----------------------------
# QQ plot vs Uniform(0,1)
# ----------------------------
u_true = np.mean(Y_true[:,np.newaxis] <= Y_true, axis=1)
u_sorted = np.sort(u_true)
uniform_quantiles = np.linspace(0, 1, len(Y_true))

plt.figure()
plt.plot(uniform_quantiles, u_sorted, 'o', label='PIT QQ')
plt.fill_between(uniform_quantiles, u_sorted, np.linspace(0,1,len(u_sorted)), alpha=0.5)
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('Uniform quantiles')
plt.ylabel('Ytrue Empirical quantiles')
plt.title('QQ plot for calibration')
plt.legend()
plt.show()

# ----------------------------
# QQ plot vs Uniform(0,1)
# ----------------------------
Ypred_posterior_std = (Ypred_posterior - np.mean(Ypred_posterior, axis=0)).flatten()
plt.hist(MAE_dist,bins=20,weights = np.ones_like(MAE_dist)/len(MAE_dist),label='MAE',histtype='step',range=(0,np.max(MAE_dist)))
plt.hist(Ypred_posterior_std,bins=20,weights = np.ones_like(Ypred_posterior_std)/len(Ypred_posterior_std),
        label='Posterior predictive std',histtype='step',range=(0,np.max(MAE_dist)))
#plt.hist((Ypred_posterior-Y_true).flatten(),bins=20,weights = np.ones_like(Ypred_posterior.flatten())/len(Ypred_posterior.flatten()),label='Posterior predictive',histtype='step')
plt.legend()
plt.show()

observed_q = np.quantile(MAE_dist, np.linspace(0,0.97,100))
predicted_q = np.quantile((Ypred_posterior - np.mean(Ypred_posterior, axis=0)).flatten(), np.linspace(0,0.97,100))
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(predicted_q, observed_q, "o", ms=4, color="steelblue", alpha=0.7)
ax.axline((0, 0), slope=1, color="k", lw=1.5, ls="--", label="y = x")
ax.set_xlabel("Posterior Predictive Residual Quantiles")
ax.set_ylabel("Observed Residual Quantiles")
ax.set_title("Residual QQ Plot vs Posterior Predictive")
plt.xlim(-0.1,0.1)
ax.legend()
plt.tight_layout()
plt.show()

Ypred_posterior_mean = np.mean(Ypred_posterior, axis=0)
Ypred_posterior_std = np.std(Ypred_posterior, axis=0)
plt.plot(x,Ypred_posterior_mean,label='Posterior mean')
plt.fill_between(x,Ypred_posterior_mean-Ypred_posterior_std,Ypred_posterior_mean+Ypred_posterior_std,alpha=0.5,label='Posterior std')
plt.plot(x,Y_true,label='True')
plt.plot(x,Y_pred,label='Best fit')
plt.ylim(-1.0,2.0)
plt.legend()
plt.show()
