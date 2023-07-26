import numpy as np


def get_c1(la, psi, rho):
    return np.power(np.power(la - psi, 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, c1):
    la_minus_psi = la - psi
    return (c1 + la_minus_psi) / (c1 - la_minus_psi)


def get_E1(c1, c2, t, T):
    return c2 * np.exp(c1 * (t - T))


def get_E2(c1, c2, ti, T):
    return c2 * np.exp(c1 * (ti - T))


def get_E3(c1, t, ti):
    return np.exp(c1 * (t - ti))


def get_E4(la, psi, t, ti):
    return np.exp((la + psi) * (t - ti))


def get_u(la, psi, c1, E1):
    two_la = 2 * la
    return (la + psi) / two_la + c1 / two_la * ((E1 - 1) / (E1 + 1))


def get_log_p(c1, t, ti, E1, E2):
    # return np.power((E2 + 1) / (E1 + 1), 2) * E3
    # return 2 * (np.log(E2 + 1) - np.log(E1 + 1)) + np.log(E3)
    return 2 * (np.log(E2 + 1) - np.log(E1 + 1)) + c1 * (t - ti)


def get_log_po(la, psi, c1, t, ti, E1, E2):
    # return (E2 + 1) / (E1 + 1) * np.power(E3 * E4, 0.5)
    # return np.log(E2 + 1) - np.log(E1 + 1) + (np.log(E3) + np.log(E4)) / 2
    return np.log(E2 + 1) - np.log(E1 + 1) + (c1 + la + psi) * (t - ti) / 2


def get_log_po_from_p_pnh(log_p, log_pnh):
    return (log_p + log_pnh) / 2


def get_log_pnh(la, psi, t, ti):
    # return E4
    return (la + psi) * (t - ti)
