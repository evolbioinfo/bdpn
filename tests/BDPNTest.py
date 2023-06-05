import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from treesimulator.mtbd_models import BirthDeathModel

from bdpn.bdpn import get_log_p_o, get_log_p_nh
from bdpn.bd import get_log_p, get_u
from bdpn.mtbd import compute_U


def plot_U(get_U, T, LA, PSI, RHO):
    tt = np.linspace(0, T, 1001)
    esol = [get_U(_)[0] for _ in tt]

    y = [get_u(t, LA[0], PSI[0], RHO[0], T) for t in tt]

    plt.plot(tt, esol, 'b', label='U_automatic(t)', alpha=0.5)
    plt.plot(tt, y, 'r', label='U_formulas(t)', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def plot_P(get_U, ti, t0, MU, LA, PSI, RHO, T):
    tt = np.linspace(ti, t0, 1001)

    def pdf_P(P, t):
        U = get_U(t)
        return (MU.sum(axis=1) + LA.dot(1 - U) + PSI) * P - (MU + U * LA).dot(P)

    y0 = np.zeros(len(PSI), np.float64)
    y0[0] = 1
    sol = odeint(pdf_P, y0, tt)

    y = [np.exp(get_log_p(t, LA[0], PSI[0], RHO[0], T, ti)) for t in tt]

    plt.plot(tt, sol[:, 0], 'b', label='P_automatic(t)', alpha=0.5)
    plt.plot(tt, y, 'r:', label='P_formulas(t)', alpha=0.5)

    def pdf_Po(P, t):
        U = get_U(t)
        return (MU.sum(axis=1) + LA.dot(1 - U) + PSI) * P - MU.dot(P)

    y0 = np.zeros(len(PSI), np.float64)
    y0[0] = 1
    sol = odeint(pdf_Po, y0, tt)
    y = [np.exp(get_log_p_o(t, LA[0], PSI[0], RHO[0], T, ti)) for t in tt]

    plt.plot(tt, sol[:, 0], 'g', label='Po_automatic(t)', alpha=0.5)
    plt.plot(tt, y, 'y:', label='Po_formulas(t)', alpha=0.5)

    def pdf_Pnh(P, t):
        return (MU.sum(axis=1) + LA.sum(axis=1) + PSI) * P - MU.dot(P)

    y0 = np.zeros(len(PSI), np.float64)
    y0[0] = 1
    sol = odeint(pdf_Pnh, y0, tt)

    y = [np.exp(get_log_p_nh(t, LA[0], PSI[0], ti)) for t in tt]

    plt.plot(tt, sol[:, 0], 'k', label='Pnh_automatic(t)', alpha=0.5)
    plt.plot(tt, y, 'm:', label='Pnh_formulas(t)', alpha=0.5)

    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def random_bt_0_and_1():
    return 1 - np.random.random(size=1)[0]


if __name__ == '__main__':

    p = random_bt_0_and_1()
    pn = random_bt_0_and_1()
    R0 = random_bt_0_and_1() * 5
    real_psi = random_bt_0_and_1()
    real_la = real_psi * R0
    real_psi_n = real_psi * 100
    T = 20
    model = BirthDeathModel(la=real_la, psi=real_psi, p=p)

    MU, LA, PSI, RHO = model.transition_rates, model.transmission_rates, model.removal_rates, model.ps
    PI = model.state_frequencies
    PSI_RHO = PSI * RHO
    SIGMA = MU.sum(axis=1) + LA.sum(axis=1) + PSI
    get_U = compute_U(T, MU=MU, LA=LA, PSI=PSI, RHO=RHO, SIGMA=SIGMA)

    plot_U(get_U, T, LA, PSI, RHO)

    plot_P(get_U, ti=10, t0=5, MU=MU, LA=LA, PSI=PSI, RHO=RHO, T=T)
