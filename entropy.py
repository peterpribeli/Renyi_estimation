import random
import numpy as np


def get_point(cent, rad):
    """
    A function to generate a random point in an n-ball.
    :param cent: Center of the ball.
    :param rad: Radius of the ball.
    :return: Random point.
    """
    n = len(cent)
    a = []
    nrm = 0.
    r = random.uniform(0, rad)
    for el in range(n):
        a.append(random.gauss(0, 1))
        nrm = nrm + a[el]**2

    if nrm is not 0.:
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
        point = np.subtract(get_point(cent, rad), cent)
        dist = 0.
        for j in range(dim):
            dist = dist + point[j]**2
        dist = np.sqrt(dist)
        for j in range(nseg):
            if dist <= seg[j]:
                counts[j] = counts[j] + 1
                break

    return counts


def entropies(data, tree, alpha, radius, num_iter):
    """
    Calculates the entropies for given alpha and radii. The zeroth element of alpha is corresponding to H_inf
    and the first element to H_1.
    :param data: List of input data.
    :param tree: The KDTree of the data.
    :param alpha: List of alphas.
    :param radius: List of radii.
    :param num_iter: Number of iterations.
    :return: Numpy array entropies[alp][rad]
    """
    m, _ = data.shape
    ml = np.log(m)
    # Initialize an array with zeros
    entrop = np.zeros((len(alpha)+2, len(radius)), dtype=float)

    for ind in range(num_iter):
        for rad in range(len(radius)):
            # Get a random point from an n-ball around a random point
            point = get_point(data[random.randrange(0, m), :], radius[rad])

            # Get the number of nearest neighbours
            neigh = len(tree.query_ball_point(point, radius[rad]))+1
            for alp in range(len(alpha)):
                # Skip alpha = 1
                if alpha[alp] == 1.:
                    continue
                # Calculate the degeneracy
                entrop[alp+2][rad] += neigh**(alpha[alp]-1.)

            # Alpha = Infinity saved at entrop[0][:]
            entrop[0][rad] = max(entrop[0][rad], neigh)

            # Alpha = 1 saved at entrop[1][:]
            entrop[1][rad] = entrop[1][rad] + np.log(neigh)

    # Rescale the degenerations to entropies
    for rad in range(len(radius)):
        for alp in range(len(alpha)+2):
            if alp == 0.:
                entrop[alp][rad] = ml - np.log(entrop[alp][rad])
            elif alp == 1.:
                entrop[alp][rad] = ml - entrop[alp][rad]/num_iter
            else:
                entrop[alp][rad] = ml + np.log(entrop[alp][rad]/num_iter)/(1.-alpha[alp-2])

    return entrop


def get_rms(radius, entrop, pars):
    """
    Calculates the RMS of the fitted data.
    :param radius: List of fitted radii
    :param entrop: List of entropies[alpha][radius]
    :param pars: List of the parameters of the linear fit
    :return: List RMS[alpha]
    """
    m, n = entrop.shape
    rms = []

    for alp in range(m):
        # Sum of squares of (data - prediction)**2
        r = 0
        for rad in range(n):
            r += (entrop[alp][rad]-pars[0]*radius[rad]-pars[1])**2
        rms.append(np.sqrt(r/n))

    return rms
