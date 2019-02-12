import random
import numpy as np


def get_point(cent, rad, seed):
    """
    A function to generate a random point in an n-ball.
    :param cent: Center of the ball.
    :param rad: Radius of the ball.
    :param seed: Seed for the Mersene twister.
    :return: Random point.
    """
    n = len(cent)
    random.seed(seed)
    a = []
    nrm = 0.
    r = random.uniform(0, rad)
    for el in range(n):
        a.append(random.gauss(0, 1))
        nrm = nrm + a[el]**2

    nrm = r**(1./n)/np.sqrt(nrm)
    for el in range(n):
        a[el] = a[el]*nrm + cent[el]

    return a


def uniformity_test(num_points, cent, rad, nseg):
    """
    Test the radial uniformity of the generator. Returns a list of counts for each individual radial ball segment.
    :param num_points: Number of test points to generate.
    :param cent: The center of the ball.
    :param rad: The radius of the ball.
    :param nseg: The number of radial segments.
    :return: List of counts for each segment.
    """
    seg = []
    dim = len(cent)
    counts = [0]*nseg

    for i in range(nseg):
        seg.append((rad*(i+1)/nseg)**(1./dim))

    for i in range(num_points):
        point = np.subtract(get_point(cent, rad, i), cent)
        dist = 0.
        for j in range(dim):
            dist = dist + point[j]**2
        dist = np.sqrt(dist)
        for j in range(nseg):
            if dist <= seg[j]:
                counts[j] = counts[j] + 1
                break

    return counts


def entropies(data, tree, center, alpha, radius, num_iter):
    """
    Calculates the entropies for given alpha and radii. The zeroth element of alpha is corresponding to H1
    and the first element to H_inf.
    :param data: Array of input data.
    :param tree: The KDTree of the data.
    :param center: An array of center point indices.
    :param alpha: Array of alphas.
    :param radius: Array of radii.
    :param num_iter: Number of iterations.
    :return: 2-D array of tuples d[alpha][radius] = (entropy, k)
    """
    m, _ = data.shape
    ml = np.log(m)
    entrop = np.zeros((len(alpha)+2, len(radius)), dtype=float)
    denom = np.zeros((len(alpha)+2, len(radius)), dtype=int)

    for rad in range(len(radius)):
        for ind in range(num_iter):
            point = get_point(data[center[ind], :], rad, ind)
            neigh = len(tree.query_ball_point(point, rad))+1
            for alp in range(len(alpha)+2):
                # Alpha = 1
                if alp == 0:
                    entrop[alp][rad] = (entrop[alp][rad]*denom[alp][rad]+np.log(neigh))/(denom[alp][rad]+1)
                    denom[alp][rad] = denom[alp][rad] + 1
                # Alpha = infinity
                if alp == 1:
                    entrop[alp][rad] = max(entrop[alp][rad], neigh)
                # Other Alphas
                if alp > 1:
                    entrop[alp][rad] = (denom[alp][rad]*entrop[alp][rad]+neigh**(alpha[alp-2]-1.))/(denom[alp][rad]+1)
                    denom[alp][rad] = denom[alp][rad] + 1

        for alp in range(len(alpha)+2):
            if alp == 0:
                entrop[alp, rad] = ml-entrop[alp, rad]
            if alp == 1:
                entrop[alp, rad] = ml - np.log(entrop[alp, rad])
            else:
                entrop[alp, rad] = ml + np.log(entrop[alp, rad]/(1.-alpha[alp-2]))

    return entrop
