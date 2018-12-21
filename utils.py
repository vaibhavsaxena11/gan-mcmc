import numpy as np
import tensorflow as tf

def sample_Z(size):
    # return np.random.uniform(low=-1.0, high=1.0, size=size)
    return np.random.multivariate_normal(0.5*np.ones(size[1]), 0.01*np.eye(size[1]), size[0])

def xavier_init(shape):
    """Initializer for the Xavier distribution.
    The output should be sampled uniformly from [-epsilon, epsilon] where
        epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>

    shape:
        Tuple or 1-d array that species the dimensions of the
            requested tensor.
    returns out:
        tf.Tensor of specified shape sampled from the
        Xavier distribution.
    """

    epsilon = np.sqrt(6/np.sum(shape))
    out = tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon+1)

    return out