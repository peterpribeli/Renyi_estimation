import random
import numpy as np
from scipy import spatial
from time import time
from entropy import entropies


if __name__ == "__main__":

    num_iter = 100
    alpha = [0.1, 0.2, 0.5, 0.7, 2, 3, 4, 5, 7, 10, 15, 50]
    radius = [0.05, 0.1, 0.15, 0.3, 0.6, 0.7]

    output = np.empty([len(alpha), len(rad_range)], dtype=float)

    # Import data
    data = np.genfromtxt('input1.csv', delimiter=' ')
    m, n = data.shape

    # Create k-d tree
    t0 = time()
    tree = spatial.cKDTree(data)
    print("k-d tree construction took: {} s.".format(time()-t0))

    # Generate indices
    center = []
    for i in range(num_iter):
        center.append(random.randint(0, m-1))

    t0 = time()
    ent = entropies(data, tree, center, alpha, radius, num_iter)
    print("Entropies calculation took: {} s.".format(time()-t0))

