import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(-5,5,20)
alpha = 0.5
f = alpha*x**2
sigma = 1
y = f + np.random.normal(0,sigma,len(x))
plt.errorbar(x,y,yerr=np.ones_like(x)*sigma, marker='o', linestyle='none', label='Simulated data')
plt.plot(x,f, label='Original function')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
sns.despine()

a = np.linspace(0-1.5,0+.3,50)
c = np.linspace(0.5-.1,.5+.1,50)
A,C = np.meshgrid(a,c)
chi2 = np.zeros_like(A.flatten()) #the log-likehood is the same as chi2
for i, (a, c) in enumerate(zip(A.flatten(),C.flatten())):
    chi2[i] = np.sum(-((y - a - c*x*x)**2)/(sigma**2))
chi2 = chi2.reshape(A.shape)


plt.figure()
cmap = plt.contourf(A,C,np.exp(chi2), levels=10, cmap='Blues')
plt.colorbar(cmap, label = r"$\rho$")
plt.xlabel("a")
plt.ylabel("c")
sns.despine()

import statsmodels.formula.api as sm
df = {'x':x, 'y':y, 'sigma':np.ones_like(x)*sigma}
df = pd.DataFrame(df)
dfs = df.sample(10, replace=True)

a = []
c = []
for i in range(100):
    dfs = df.sample(10, replace=True)
    model = sm.ols('y ~ I(x**2)', data=dfs).fit()
    a.append(model.params['Intercept'])
    c.append(model.params['I(x ** 2)'])
plt.scatter(a,c)
plt.xlabel("a")
plt.ylabel("c")
sns.despine()

df = {'x':x, 'y':y, 'sigma':np.ones_like(x)*sigma}
df = pd.DataFrame(df)

a = []
b = []
c = []
for i in range(1000):
    dfs = df.sample(10, replace=True)
    model = sm.ols('y ~ x + I(x**2)', data=dfs).fit()
    a.append(model.params['Intercept'])
    b.append(model.params['x'])
    c.append(model.params['I(x ** 2)'])
a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)

ensemble_f = a[:,np.newaxis] + c[:,np.newaxis]*x[np.newaxis,:]*x[np.newaxis,:]
f_mean = np.mean(ensemble_f,axis=0)

plt.clf()
plt.errorbar(x,y,yerr=np.ones_like(x)*sigma, marker='o', linestyle='none', label='Simulated data')
plt.plot(x,f_mean, label='ensemble_mean')
plt.plot(x,f,label="true function")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("ensemble_mean_f.png")
plt.clf()

minimum = -b/(2*c)
sns.displot(minimum, kde=True)
plt.xlabel("Minimum")
sns.despine()


plt.figure()
plt.scatter(b, minimum)
plt.xlabel("b")
plt.ylabel("Minimum")
sns.despine()