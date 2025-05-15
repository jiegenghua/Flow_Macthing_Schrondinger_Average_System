import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

n_samples = 1000  # number of samples to generate
noise = 0.1  # noise to add to sample locations
x, _ = datasets.make_moons(n_samples=n_samples, noise=noise)
print(np.shape(x),np.shape(y))

plt.scatter(*x.T, c=y, cmap=plt.cm.Accent)
plt.title("Generated half moons data")
plt.show()