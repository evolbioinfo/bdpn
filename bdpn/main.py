import pandas as pd
from ete3 import Tree

import bdpn.bd as bd
import bdpn.bdpn as bdpn
from bdpn.parameter_estimator import optimize_likelihood_params

nwk = '/home/azhukova/projects/bdpn/trees/bdpn/tree.1.nwk'
tree = Tree(nwk)
df = pd.read_csv(nwk.replace('.nwk', '.log'))
rho = df.loc[0, 'sampling probability']
rho_n = df.loc[0, 'notification probability']
R0 = df.loc[0, 'R0']
it = df.loc[0, 'infectious time']
rt = df.loc[0, 'removal time after notification']
psi = 1 / it
psi_n = 1 / rt
la = R0 * psi
vs = optimize_likelihood_params(tree, input_parameters=[la, None, None, None, None],
                                loglikelihood=bdpn.loglikelihood, get_bounds_start=bdpn.get_bounds_start)

vs_bd = optimize_likelihood_params(tree, input_parameters=[la, None, None],
                                   loglikelihood=bd.loglikelihood, get_bounds_start=bd.get_bounds_start)
print('Real params: {}'.format([la, psi, psi_n, rho, rho_n]))
print('Found BDPN params: {}'.format(vs))
print('Found BD params: {}'.format(vs_bd))