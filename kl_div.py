from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def KLD(samples1, samples2):
    """
    samples1: samples from dist1 (2-d array)
    samples2: samples from dist2 (2-d array)

    returns: sum_i [ log(dist1(samples1[i])) - log(dist2(samples1[i])) ]
    """

    dist1 = KDE(samples1)
    dist2 = KDE(samples2)

    query_space = [[1,1]]
    space_pts = 100
    for i in np.linspace(0, 1, space_pts):
        for j in np.linspace(0, 1, space_pts):
            query_space = np.append(query_space, [[i,j]], axis=0)
    query_space = query_space[1:]

    kl_div = 0
    for i in range(len(query_space)):
        query = np.array([ [query_space[i,0]], [query_space[i,1]] ])
        # kl_div = kl_div + np.log(dist1(query)[0]) - np.log(dist2(query)[0])
        kl_div = kl_div + np.log(dist1(query)[0]/dist2(query)[0])

    return kl_div

def KDE(data):
    kde = stats.gaussian_kde(data.T)
    return kde

def plot_density(data, title='pdf', filename='./figures/density'):
    query_space = [[1,1]]
    space_pts = 100
    for i in np.linspace(0, 1, space_pts):
        for j in np.linspace(0, 1, space_pts):
            query_space = np.append(query_space, [[i,j]], axis=0)
    query_space = query_space[1:]

    pdf = np.array([])
    density_fn = KDE(data)
    for i in range(len(query_space)):
        query = np.array([ [query_space[i,0]], [query_space[i,1]] ])
        pdf = np.append(pdf, density_fn(query)[0])

    ## Plotting
    x = query_space[:,0]
    y = query_space[:,1]
    z = pdf
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=len(x)*[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('pdf')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
