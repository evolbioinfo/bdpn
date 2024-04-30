import numpy as np


def get_c1(la, psi, rho):
    """
    Return c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    """
    return np.power(np.power(la - psi, 2) + 4 * la * psi * rho, 0.5)


def get_c2(la, psi, c1):
    """
    Return c2 = (c1 + la - psi) / (c1 - la + psi)
    """
    la_minus_psi = la - psi
    return (c1 + la_minus_psi) / (c1 - la_minus_psi)


def get_E1(c1, c2, t, T):
    """
    Returns E1 = c2 * exp(c1 * (t - T))
    """
    return c2 * np.exp(c1 * (t - T))


def get_E2(c1, c2, ti, T):
    """
    Returns E2 = c2 * exp(c1 * (ti - T))
    """
    return c2 * np.exp(c1 * (ti - T))


def get_E3(c1, t, ti):
    """
    Returns E3 = exp(c1 * (t - ti))
    """
    return np.exp(c1 * (t - ti))


def get_u(la, psi, c1, E1):
    """
    Returns U(t), where
    dU(t)/dt = la + psi * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E1: E1 = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    """
    two_la = 2 * la
    return (la + psi) / two_la + c1 / two_la * ((E1 - 1) / (E1 + 1))


def get_log_p(c1, t, ti, E1, E2):
    """
    Returns log(p(t)), where
    dp(t)/dt = la + psi * p(t) − 2 * la * p(t) * U(t)
    p(ti) = 1
    dU(t)/dt = la + psi * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1.

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E1: E1 = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E2: E2 = c2 * exp(c1 * (ti - T))
    """
    return 2 * (np.log(E2 + 1) - np.log(E1 + 1)) + c1 * (t - ti)


def get_log_po(la, psi, c1, t, ti, E1, E2):
    """
    Returns log(po(t)), where
    dpo(t)/dt = la + psi * po(t) − la * po(t) * U(t)
    po(ti) = 1
    dU(t)/dt = la + psi * U(t) − la U^2(t) − psi * (1 − rho)
    U(T) = 1.

    :param c1: c1 = ((la − psi)^2 + 4 * la * psi * rho)^1/2
    :param E1: E1 = c2 * exp(c1 * (t - T)), where c2 = (c1 + la - psi) / (c1 - la + psi)
    :param E2: E2 = c2 * exp(c1 * (ti - T))
    """
    return np.log(E2 + 1) - np.log(E1 + 1) + (c1 + la + psi) * (t - ti) / 2


def get_log_po_from_p_pn(log_p, log_pn):
    """
    Returns log(po(t)), where
    dpo(t)/dt = la + psi * po(t) − la * po(t) * U(t)
    po(ti) = 1
    if log(p(t)) and log(pn(t)) are known, where
    dpn(t)/dt = la + psi * pnh(t)
    pn(ti) = 1,
    dp(t)/dt = la + psi * p(t) − 2 * la * p(t) * U(t)
    p(ti) = 1
    """
    return (log_p + log_pn) / 2


def get_log_pn(la, psi, t, ti):
    """
    Returns log(pn(t)), where
    dpn(t)/dt = la + psi * pn(t)
    pn(ti) = 1
    """
    return (la + psi) * (t - ti)


def get_log_pp(la, phi, t, ti):
    """
    Returns log(pp(t)), where
    dpp(t)/dt = phi * pp(t)
    pp(ti) = 1
    """
    return phi * (t - ti)
