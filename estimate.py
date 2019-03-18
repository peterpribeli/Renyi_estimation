import numpy as np
from scipy import spatial
from time import time
from entropy import entropies, get_rms
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Set input
    num_iter = 1000
    # alpha = [0.05, 0.1, 0.8, 1.2, 1.9]
    alpha = [0.8, 1.2]
    radius = [1e-1, 2e-1, 3e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1, 1.2, 1.5, 2, 3, 4, 5]
    # radius = [5e-1, 7e-1, 8e-1, 9e-1, 1, 1.2, 1.5]

    # Import data
    data = np.genfromtxt('input1.csv', delimiter=' ')
    m, n = data.shape

    # Create k-d tree
    t0 = time()
    tree = spatial.cKDTree(data)
    print("k-d tree construction took: {:.1f} s.".format(time()-t0))

    # Calculate the entropies
    t0 = time()
    ent = entropies(data, tree, alpha, radius, num_iter)
    print("Entropies calculation took: {:.1f} s.".format(time()-t0))

    # Fit the data
    strt = 4  # Begining of the fit
    stp = 9  # End of the fit
    lograd = [np.log(y) for y in radius]

    coef, cov = np.polyfit(lograd[strt:stp], ent[2, strt:stp], 1, cov=True)
    rms = get_rms(lograd[strt:stp], ent[:, strt:stp], coef)

    print("The dimension is: {:.2f}, RMS: {:.2f}, sigma: {:.2f}".format(-coef[0], rms[2], np.sqrt(cov[0, 0])))

    # Plot the entropy vs radius for alpha=1
    x = np.linspace(radius[strt], radius[stp])
    plt.xscale('log')

    plt.plot(x, coef[0]*np.log(x)+coef[1], linestyle='solid', color='red')
    plt.scatter(radius, ent[2, :])
    plt.show()
