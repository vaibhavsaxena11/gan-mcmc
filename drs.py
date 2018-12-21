## Code for Metropolis-Hastings and Rejection Sampling
## obtain p_data samples using p_g samples as proposals

import tensorflow as tf
import numpy as np
import utils

def DRS(G, sess, length=1000):
    max_trials = length
    p_samples = np.array([])


    logits = np.array([])
    samples = np.array([[0,0]]) # samples from G
    for i in range(max_trials):
        D_logit, G_sample = sess.run(fetches=[G.D_fake_logit, G.G_sample], feed_dict={G.Z: utils.sample_Z([1, G.G_input_size])})
        logits = np.append(logits, D_logit)
        samples = np.append(samples, G_sample, axis=0)
    samples = samples[1:]

    ratios = np.exp(logits)


    mcmc = True

    if mcmc: # Metropolis-Hastings
        prev_p_by_q = ratios[0]
        indices = np.array([])
        for i in range(1,len(ratios)):
            curr_p_by_q = ratios[i]
            acceptance_prob = curr_p_by_q/prev_p_by_q
            prev_p_by_q = curr_p_by_q

            if np.random.uniform(0,1,1) < acceptance_prob:
                indices = np.append(indices, i)
        indices = indices.astype(int)
        p_samples = samples[indices]
    
    else: # Rejection sampling
        ratios_normalized = ratios/np.max(ratios) # dividing by exp(M)

        indices = np.array([])
        for i in range(len(ratios_normalized)):
            acceptance_prob = ratios_normalized[i]
            if np.random.uniform(0,1,1) < acceptance_prob:
                indices = np.append(indices, i)
        indices = indices.astype(int)
        p_samples = samples[indices]


    return p_samples, samples