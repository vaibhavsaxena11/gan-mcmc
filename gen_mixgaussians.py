import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

## Params for 2 gaussians
mu1, sigma1 = ([0.2,0.8], [[0.01,0],[0,0.01]])
mu2, sigma2 = ([0.8,0.2], [[0.01,0],[0,0.01]])

samples = np.array([])
num_samples = 4000

for i in range(num_samples):
    p = np.random.uniform(0,1,1)
    
    if p < 0.5:
        val = np.random.multivariate_normal(mu1, sigma1, 1)
    else:
        val = np.random.multivariate_normal(mu2, sigma2, 1)

    if samples.shape[0] is 0:
        samples = val
        continue
    samples = np.append(samples, val, axis=0)

## Normalizing all samples in [0,1]
samples = (samples - np.min(samples, axis=0))/(np.max(samples, axis=0) - np.min(samples, axis=0))
## Saving samples to file
np.savetxt("mixgaussians.csv", samples, delimiter=",")

## Plotting
from mpl_toolkits.mplot3d import Axes3D
x = samples[:,0]
y = samples[:,1]
z = np.ones(samples.shape[0])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()