import tensorflow as tf
import numpy as np
import utils

def random_init(size):
    # return tf.random_normal(shape=size, mean=0.0, stddev=0.25) ## add stddev
	# return tf.random_uniform(shape=size, minval=-1, maxval=1)
	# return utils.xavier_init(size)
	return tf.random.truncated_normal(size, stddev=2.0)

## importing data
data = np.genfromtxt('mixgaussians.csv', delimiter=',') ## get this from a distribution we want to learn


class GAN:
	def __init__(self, sess, data):
		self.sess = sess
		self.data = data

		## Discriminator network graph
		self.D_input_size = data.shape[1]
		self.hidden_size_D = 30

		self.D_pre_labels = tf.placeholder(tf.float32, shape=[None, 1]) # pre-training labels (will contain probability of input x)

		self.X = tf.placeholder(tf.float32, shape=[None, self.D_input_size])

		self.D_W1 = tf.Variable(initial_value = random_init([self.D_input_size, self.hidden_size_D]))
		self.D_b1 = tf.Variable(initial_value = tf.zeros(shape=[self.hidden_size_D]))

		self.D_W2 = tf.Variable(initial_value = random_init([self.hidden_size_D, 1]))
		self.D_b2 = tf.Variable(initial_value = tf.zeros(shape=[1]))

		self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

		# self.D_W2 = tf.Variable(initial_value = random_init([self.hidden_size_D, self.hidden_size_D]))
		# self.D_b2 = tf.Variable(initial_value = tf.zeros(shape=[self.hidden_size_D]))

		# self.D_W3 = tf.Variable(initial_value = random_init([self.hidden_size_D, 1]))
		# self.D_b3 = tf.Variable(initial_value = tf.zeros(shape=[1]))
		
		# self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3]


		## Generator network graph
		self.G_input_size = 2
		self.G_output_size = self.D_input_size
		self.hidden_size_G = 30

		self.Z = tf.placeholder(tf.float32, shape=[None, self.G_input_size])

		self.G_W1 = tf.Variable(initial_value = random_init([self.G_input_size, self.hidden_size_G]))
		self.G_b1 = tf.Variable(initial_value = tf.zeros(shape=[self.hidden_size_G]))

		self.G_W2 = tf.Variable(initial_value = random_init([self.hidden_size_G, self.G_output_size]))
		self.G_b2 = tf.Variable(initial_value = tf.zeros(shape=[self.G_output_size]))

		self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

		# self.G_W2 = tf.Variable(initial_value = random_init([self.hidden_size_G, self.hidden_size_G]))
		# self.G_b2 = tf.Variable(initial_value = tf.zeros(shape=[self.hidden_size_G]))

		# self.G_W3 = tf.Variable(initial_value = random_init([self.hidden_size_G, self.G_output_size]))
		# self.G_b3 = tf.Variable(initial_value = tf.zeros(shape=[self.G_output_size]))

		# self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2, self.G_W3, self.G_b3]

	def _generator(self, z): # Create G graph
		G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
		G_output = tf.matmul(G_h1, self.G_W2) + self.G_b2

		# G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
		# G_output = tf.matmul(G_h2, self.G_W3) + self.G_b3

		return G_output

	def _discriminator(self, x): # Create D graph
		D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
		D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
		D_output = tf.nn.sigmoid(D_logit)

		# D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
		# D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
		# D_output = tf.nn.sigmoid(D_logit)

		return D_output, D_logit

	def _connect_D_G(self):
		## Connecting inputs to network graphs
		self.G_sample = self._generator(self.Z)
		self.D_real, self.D_real_logit = self._discriminator(self.X)
		self.D_fake, self.D_fake_logit = self._discriminator(self.G_sample)

	def train(self, iters=1000):
		self._connect_D_G()

		### Loss functions

		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logit, labels=tf.ones_like(self.D_real_logit)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.zeros_like(self.D_fake_logit)))
		# D wants to o/p 1 for real and 0 for fake
		D_loss = D_loss_real + D_loss_fake
		G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.ones_like(self.D_fake_logit)))
		# G wants D to o/p 1 for fake (it's own o/p's)

		## Pre-training loss
		D_loss_pre = tf.reduce_mean(tf.square(self.D_real - self.D_pre_labels))

		## Optimization
		lr = 0.001 ## HP
		D_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=self.theta_D)
		G_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=self.theta_G)
		# look into other params for AdamOptimizer() (beta1, beta2, epsilon)
		D_pre_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss_pre, var_list=self.theta_D)

		self.sess.run(tf.global_variables_initializer())


		if True:
			## Pre-training discriminator
			
			# Pre-training is done to make discriminator get a good idea of P_data distribution
			# Since P_data is unknown, we get histogram and estimate pdf
			d = self.data
			n_bins = 30
			histc, xedges, yedges = np.histogram2d(d[:,0], d[:,1], bins=n_bins, density=True)

			# Estimated pdf is used as labels after normalization
			max_histc = np.max(histc)
			min_histc = np.min(histc)
			labels = (histc - min_histc) / (max_histc - min_histc)
			d = np.array([[0,0]])
			for x in xedges[1:]:
				for y in yedges[1:]:
					d = np.append(d, [[x,y]], axis=0)
			d = d[1:]
			labels = np.reshape(labels, [n_bins*n_bins, 1])

			num_pretrain_steps = 10000
			for i in range(num_pretrain_steps):
				# Execute one training step
				loss, _ = self.sess.run([D_loss_pre, D_pre_optimizer], {self.X: d, self.D_pre_labels: labels})
				if i%100 ==0:
					print('pre-training: iter {} loss: {}'.format(i + 1, loss))
			print('Pre-training finished!')
			# Simply plotting the histogram
			def plotting(d,labels):
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import Axes3D
				fig = plt.figure()
				ax = Axes3D(fig)
				ax.scatter(d[:,0], d[:,1], labels[:,0])
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('pdf')
				plt.savefig('./figures/hist.png')
			# plotting(d, labels)


		## Adversarial training
		num_batches = 40
		batch_size = int(self.data.shape[0]/num_batches) ## change to smaller value and add loop
		idx_epoch = 0
		for i in range(iters):
			D_curr_loss = None
			G_curr_loss = None
			train_data = self.data[idx_epoch*batch_size:(idx_epoch+1)*batch_size]
			idx_epoch = (idx_epoch+1)%num_batches
			# ## Batch normalization
			# train_data = (train_data - np.min(train_data, axis=0))/(np.max(train_data, axis=0) - np.min(train_data, axis=0))

			if  i<500:
				D_curr_loss, _ = self.sess.run(fetches=[D_loss, D_optimizer], feed_dict={self.X: train_data, self.Z: utils.sample_Z([batch_size, self.G_input_size])}) ## change feed_dict X:data
			else:
				# if i%2==0:
				D_curr_loss, _ = self.sess.run(fetches=[D_loss, D_optimizer], feed_dict={self.X: train_data, self.Z: utils.sample_Z([batch_size, self.G_input_size])}) ## change feed_dict X:data
				G_curr_loss, _ = self.sess.run(fetches=[G_loss, G_optimizer], feed_dict={self.Z: utils.sample_Z([batch_size, self.G_input_size])})

			if i%100 is 0:
				print('Training', i, 'iters. D loss:', D_curr_loss, 'G loss:', G_curr_loss)

		# g_data = self.sess.run(self.G_sample, {self.Z: utils.sample_Z([5000, self.G_input_size])})
		# return g_data

	def __call__(self, z):
		# create duplicate G graph with tensor input (rather than a placeholder)
		_G = self._generator(z)
		return _G

