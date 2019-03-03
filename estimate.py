import numpy as np
from scipy import spatial
from time import time
from entropy import entropies
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Set input
    num_iter = 10000
    alpha = [0.05, 0.1, 0.8, 1.2, 1.9]
    radius = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]

    # Import data
    data = np.genfromtxt('input1.csv', delimiter=' ')
    m, n = data.shape

    # Create k-d tree
    t0 = time()
    tree = spatial.cKDTree(data)
    print("k-d tree construction took: {} s.".format(time()-t0))

    # Calculate the entropies
    t0 = time()
    ent = entropies(data, tree, alpha, radius, num_iter)
    print("Entropies calculation took: {} s.".format(time()-t0))
    print(ent)

    # Fit the data
    reg = LinearRegression().fit(np.vstack((radius, ent[1, :])).T, np.vstack((radius, ent[1, :])).T)
    print(reg.coef_)

    # Plot the entropy vs radius for alpha=1
    plt.plot(radius, ent[2, :])
    plt.show()

