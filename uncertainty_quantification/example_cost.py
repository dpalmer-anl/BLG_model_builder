import numpy as np
import matplotlib.pyplot as plt

n = 500
ndpoints = np.arange(10,n,1)
classical_cost = np.exp(-ndpoints/50) + 2.1
tetb_cost = np.exp(-ndpoints/150) + 2

plt.plot(ndpoints,classical_cost,label="classical cost")
plt.plot(ndpoints,tetb_cost,label="TETB cost")
plt.xlabel("num. training data points")
plt.legend()
plt.ylabel("Test Error")
plt.title("Example figure")
plt.savefig("example_tetb_classical_cost.png")
