import os

import numpy as np
import pandas as pd
from ete3 import Tree

import bdpn.bd as bd
import bdpn.bdpn as bdpn
from bdpn.parameter_estimator import optimize_likelihood_params, AIC

# DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees', 'bd'))
# nwk = os.path.join(DATA_DIR, 'tree.1.nwk')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulations', 'trees', 'BDPN'))
nwk = os.path.join(DATA_DIR, 'tree.6.nwk')
tree = Tree(nwk)
df = pd.read_csv(nwk.replace('.nwk', '.log'))
rho = df.loc[0, 'sampling probability']
rho_n = df.loc[0, 'notification probability'] if 'notification probability' in df.columns else 0
R0 = df.loc[0, 'R0']
it = df.loc[0, 'infectious time']
rt = df.loc[0, 'removal time after notification'] if 'notification probability' in df.columns else 1
psi = 1 / it
psi_n = 1 / rt
la = R0 * psi


lk_real_bdpn = bdpn.loglikelihood([tree], la, psi, psi_n, rho, rho_n)
lk_real_bd = bd.loglikelihood([tree], la, psi, rho)
print('Real params: {}\t loglikelihood-bdpn: {}\t loglikelihood-bd: {}'.format([la, psi, psi_n, rho, rho_n], lk_real_bdpn, lk_real_bd))

input_parameters = np.array([None, None, None, rho, None])
# bounds = np.array([[la / 10, la * 10], [psi / 10, psi * 10], [psi_n / 10, psi_n * 10], [1e-3, 1], [0, 1]])
bounds = np.array([[0.05, 1], [0.05, 0.2], [0.5, 20], [0.01, 0.99999], [0.1, 0.99999]])
start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
vs_bdpn, cis_bdpn = optimize_likelihood_params([tree], input_parameters=input_parameters,
                                     loglikelihood=bdpn.loglikelihood,
                                     bounds=bounds[input_parameters == None],
                                     start_parameters=start_parameters, cis=True)
lk_bdpn = bdpn.loglikelihood([tree], *vs_bdpn)
print('Found BDPN params: {}\n\t{}\t loglikelihood: {}, AIC: {}'.format(vs_bdpn, cis_bdpn, lk_bdpn, AIC(4, lk_bdpn)))

bd_indices = [0, 1, 3]
input_parameters_bd = input_parameters[bd_indices]
bounds_bd = bounds[bd_indices]
start_parameters_bd = (bounds_bd[:, 0] + bounds_bd[:, 1]) / 2
vs_bd, cis_bd = optimize_likelihood_params([tree], input_parameters=input_parameters_bd,
                                   loglikelihood=bd.loglikelihood, bounds=bounds_bd[input_parameters_bd == None],
                                   start_parameters=start_parameters_bd, cis=True)
lk_bd = bd.loglikelihood([tree], *vs_bd)
print('Found BD params: {}\n\t{}\t loglikelihood: {}, AIC: {}'.format(vs_bd, cis_bd, lk_bd, AIC(2, lk_bd)))
