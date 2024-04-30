import pandas as pd
from ete3 import Tree

from bdpn import bd_model
from bdpn.bdpn_model import loglikelihood
import numpy as np
from scipy.stats import binomtest

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

PYBDEI = 'PyBDEI'

REAL_TYPE = 'real'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots likelihoods.")
    parser.add_argument('--nwk', type=str, help="tree", default="/home/azhukova/projects/bdpn/simulations/trees/BDPN/tree.0.nwk")
    parser.add_argument('--log', type=str, default="/home/azhukova/projects/bdpn/simulations/trees/BDPN/tree.0.log")
    params = parser.parse_args()

    tree = Tree(params.nwk)

    df = pd.read_csv(params.log)
    R0, infectious_time, p, pn, rt = df.iloc[0, :5]
    la, psi, psip = R0 / infectious_time, 1 / infectious_time, 1 / rt

    pns = np.arange(pn / 10, max(pn * 10, 1), step=(-pn / 10 + max(pn * 10, 1)) / 10)
    psips = np.arange(psip / 10, psip * 10, step=(-psip / 10 + psip * 10) / 10)
    Z = -np.ones((pns.shape[0], psips.shape[0]), dtype=float) * np.inf
    X = np.zeros(Z.shape, dtype=float)
    Y = np.zeros(Z.shape, dtype=float)
    z_min, z_max = np.inf, -np.inf
    for i in range(pns.shape[0]):
        for j in range(psips.shape[0]):
            X[i, j] = pns[i]
            Y[i, j] = psips[j]
            z = loglikelihood([tree], la, psi, psips[j], p, pns[i])
            Z[i, j] = z
            z_min = min(z_min, z)
            z_max = max(z_max, z)

    ax = plt.figure().add_subplot(projection='3d')
    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, alpha=0.3)
    ax.set(xlim=(pns[0], pns[-1]), ylim=(psips[0], psips[-1]), zlim=(z_min, z_max),
           xlabel='P_n', ylabel='Psi_p', zlabel='logLK')
    true_lk = loglikelihood([tree], la, psi, psip, p, pn)
    print(pn, psip, true_lk)
    # ax.plot_surface(np.array([[pn]]), np.array([[psip]]), np.array([[true_lk]]), edgecolor='red', lw=0.5, alpha=0.5, color='red')

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax.contour(X, Y, Z, zdir='x', offset=pns[0], cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='y', offset=psips[0], cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='z', offset=z_min, cmap='coolwarm')


    plt.show()
