import tensorflow as tf
import numpy as np
import utils
from gan import GAN
from kl_div import KLD, KDE, plot_density
import matplotlib.pyplot as plt

## importing data
data = np.genfromtxt('mixgaussians.csv', delimiter=',') ## get this from a distribution we want to learn
# data = (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0))

# Plotting data
x = data[:,0]
y = data[:,1]
fig = plt.figure()
plt.scatter(x,y, s=len(x)*[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mixture of Gaussians')
plt.xlim(0,1)
plt.ylim(0,1)
# plt.savefig('./figures/data.png')
plt.show()

## Training and saving / Restoring weights
restore = True
sess = tf.Session()
G = GAN(sess, data)
saver = tf.train.Saver()
if not restore:
    G.train(iters=15000)
    saver.save(sess, './tmp/gan.weights')
else:
    G._connect_D_G()
    saver.restore(sess, './tmp/gan.weights')



### Discriminator Sampling ###

from drs import DRS
print('Generating P_data...')
max_samples = 5000
p_samples, G_samples = DRS(G, sess, length=max_samples)
print('{} out of {} samples from P_g accepted'.format(p_samples.shape[0], max_samples))

# ## Calculating KL Divergence between p_r and p_g
# forward_KL = KLD(p_samples, G_samples)
# reverse_KL = KLD(G_samples, p_samples)
# print('Forward KLD (p_r || p_g): {}'.format(forward_KL))
# print('Reverse KLD (p_g || p_r): {}'.format(reverse_KL))

plot_density(p_samples, title='p_r density', filename='./figures/p_density')
plot_density(G_samples, title='p_g density', filename='./figures/q_density')

# Plotting the retrieved samples
x = p_samples[:,0]
y = p_samples[:,1]
fig = plt.figure()
plt.scatter(x,y, s=len(x)*[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('P_r (Using MCMC)')
plt.xlim(0,1)
plt.ylim(0,1)
# plt.savefig('./figures/p_samples_mcmc.png')
plt.show()


