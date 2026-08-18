import numpy as np 
import matplotlib.pyplot as plt

theta = np.array([0.93,0.99,1.05,1.08,1.12,1.16,1.2,1.47])
aa_sep = np.array([3.5990, 3.5978, 3.5963,3.5956, 3.5630, 3.5628,
                    3.5625, 3.5654])
plt.plot(theta,aa_sep,'o-',label="fitting data: MBD energies only")
plt.xlabel('Theta')
plt.ylabel('AA Separation')
plt.title('AA Separation vs Theta')
plt.show()