import random
import numpy as np
import csv
from scipy import spatial


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


if __name__ == "__main__":

    num_iter = 100
    radius = 1.

    # Import data
    data = np.genfromtxt('input.csv', delimiter=' ')
    m, n = data.shape

    for ind in range(num_iter):
        center = data[random.randint(0, m-1), :]
        point = get_point(center, radius, ind)

        print(center)

    # print(uniformity_test(1000, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2., 5))
