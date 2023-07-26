import os
from datetime import datetime

import numpy as np
import pandas as pd
from ete3 import Tree
from matplotlib.pyplot import plot, show, title
from scipy.stats import chi2

import bdpn.bd as bd
import bdpn.bdpn as bdpn
from bdpn.parameter_estimator import optimize_likelihood_params, AIC

# DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trees', 'bd'))
# nwk = os.path.join(DATA_DIR, 'tree.1.nwk')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'simulations', 'trees', 'BDPN'))
nwk = os.path.join(DATA_DIR, 'tree.2.nwk')
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

lk_real_bdpn = bdpn.loglikelihood([tree], la, psi, psi_n, rho, rho_n, threads=12)
# lk_real_bd = bd.loglikelihood([tree], la, psi, rho)
real_params = [la, psi, psi_n, rho, rho_n]
# print('Real params: {}\t loglikelihood-bdpn: {}\t loglikelihood-bd: {}'.format(real_params, lk_real_bdpn, lk_real_bd))
print('Real params: {}\t loglikelihood: {}'.format(real_params, lk_real_bdpn))

input_parameters = np.array([None, None, None, rho, None])
# bounds = np.array([[la / 10, la * 10], [psi / 10, psi * 10], [psi_n / 10, psi_n * 10], [1e-3, 1], [0, 1]])
bounds = np.array([[0.05, 1], [0.05, 0.2], [2.5, 100], [0.01, 0.99999], [0.1, 0.99999]])
start_parameters = (bounds[:, 0] + bounds[:, 1]) / 2
print(datetime.now().strftime("%H:%M:%S"))
vs_bdpn, cis_bdpn = optimize_likelihood_params([tree], input_parameters=input_parameters,
                                               loglikelihood=bdpn.loglikelihood,
                                               bounds=bounds[input_parameters == None],
                                               start_parameters=start_parameters, cis=True, threads=12)
lk_bdpn = bdpn.loglikelihood([tree], *vs_bdpn, threads=12)
print(datetime.now().strftime("%H:%M:%S"))
vs_bdpn, cis_bdpn = optimize_likelihood_params([tree], input_parameters=input_parameters,
                                               loglikelihood=bdpn.loglikelihood,
                                               bounds=bounds[input_parameters == None],
                                               start_parameters=start_parameters, cis=True, threads=1)
lk_bdpn = bdpn.loglikelihood([tree], *vs_bdpn, threads=1)
print(datetime.now().strftime("%H:%M:%S"))
print('Found BDPN params: {}\t loglikelihood: {}, AIC: {}'.format(vs_bdpn, lk_bdpn, AIC(4, lk_bdpn)))
print(cis_bdpn)

exit()

lk_threshold = lk_bdpn - chi2.ppf(q=0.95, df=1) / 2
parameter_names = ['lambda', 'psi', 'psi_p', 'rho', 'rho_p']

for i in range(len(real_params)):
    step = (cis_bdpn[i][1] - cis_bdpn[i][0]) / 15
    if step == 0:
        continue
    vs = np.arange(cis_bdpn[i][0], cis_bdpn[i][1] + step, step)
    lks = []
    for _ in vs:
        ps = list(vs_bdpn)
        ps[i] = _
        lks.append(bdpn.loglikelihood([tree], *ps))

    plot(vs, lks, 'b')
    plot(cis_bdpn[i], [lk_threshold, lk_threshold], 'r')
    plot([real_params[i]], [lk_threshold], 'r*')
    plot([vs_bdpn[i]], [lk_bdpn], 'b*')

    vs = np.array(vs)
    lks = np.array(lks)

    # dy = np.diff(lks, 1)
    # dx = np.diff(vs, 1)
    # yfirst = dy / dx
    # xfirst = 0.5 * (vs[:-1] + vs[1:])
    #
    # dyfirst = np.diff(yfirst, 1)
    # dxfirst = np.diff(xfirst, 1)
    # ysecond = dyfirst / dxfirst
    # xsecond = 0.5 * (xfirst[:-1] + xfirst[1:])
    #
    # diff = np.inf
    # the_i = 0
    # for j, x in enumerate(xsecond):
    #     x_diff = np.abs(x - vs_bdpn[i])
    #     if x_diff < diff:
    #         diff = x_diff
    #         the_i = j
    # ci = [vs_bdpn[i] - 1.96 / np.power(-ysecond[the_i], 0.5), vs_bdpn[i] + 1.96 / np.power(-ysecond[the_i], 0.5)]
    # plot(ci, [lk_threshold - 1, lk_threshold - 1], 'g')

    title(parameter_names[i])
    show()
exit()

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
                                           loglikelihood=bd.loglikelihood,
                                           bounds=bounds_bd[input_parameters_bd == None],
                                           start_parameters=start_parameters_bd, cis=True)
lk_bd = bd.loglikelihood([tree], *vs_bd)
print('Found BD params: {}\n\t{}\t loglikelihood: {}, AIC: {}'.format(vs_bd, cis_bd, lk_bd, AIC(2, lk_bd)))
