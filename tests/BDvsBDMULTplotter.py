import numpy as np
from matplotlib import pyplot as plt

from bdpn import bd_model, bd_model_mult
from bdpn.formulas import get_c1, get_E, get_c2


def random_bt_0_and_1():
    return 1 - np.random.random(size=1)[0]


if __name__ == '__main__':

    rho = random_bt_0_and_1()
    R0 = random_bt_0_and_1() * 5
    psi = random_bt_0_and_1()
    la = psi * R0
    T = 30
    print(la, psi, rho)

    # dt = T / 1000
    dt = max(bd_model_mult.EPSILON, min(1 / la, 1 / psi) / 10)
    tt = [i * dt for i in range(int(T / dt))] + [T]
    # t = T
    # while t > 0:
    #     t -= dt
    #     tt.append(t)

    c1 = get_c1(la, psi, rho)
    c2 = get_c2(la, psi, c1)
    Us = [bd_model.get_u(la, psi, c1, get_E(c1, c2, t, T)) for t in tt]
    plt.plot(tt, Us, label='U_BD(t)', alpha=0.5)
    Us = bd_model_mult.precalc_u(T, dt, la, psi, rho, r=1.1)
    plt.plot(tt, Us, label='U_BD_{}(t)'.format(1.1), alpha=0.5)
    for r in range(2, 25):
        Us = bd_model_mult.precalc_u(T, dt, la, psi, rho, r)
        plt.plot(tt, Us, label='U_BD_{}(t)'.format(r), alpha=0.5)


    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

    plt.clf()

    ti = random_bt_0_and_1() * T

    E_ti = get_E(c1, c2, ti, T)
    tt = [t for t in tt if t < ti]
    Ps = [bd_model.get_log_p(c1, t, ti, get_E(c1, c2, t, T), E_ti) for t in tt]
    plt.plot(tt, Ps, label='logP_BD(t)', alpha=0.5)
    Us = bd_model_mult.precalc_u(T, dt, la, psi, rho, r=1.1)
    plt.plot(tt, [bd_model_mult.get_log_p(t, ti, dt, la, psi, r=1.1, Us=Us) for t in tt],
             label='logP_BD_{}(t)'.format(1.1), alpha=0.5)
    for r in range(2, 25):
        Us = bd_model_mult.precalc_u(T, dt, la, psi, rho, r=r)
        plt.plot(tt, [bd_model_mult.get_log_p(t, ti, dt, la, psi, r=r, Us=Us) for t in tt],
                 label='logP_BD_{}(t)'.format(r), alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
