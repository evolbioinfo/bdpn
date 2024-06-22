import logging

import numpy as np
from treesimulator import save_log, save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import BirthDeathModel, PNModel


def random_float(min_value=0, max_value=1):
    """
    Generate a random float in ]min_value, max_value]
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + (1 - np.random.random(size=1)[0]) * (max_value - min_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates random BDPN tree.")
    parser.add_argument('--log', default='/home/azhukova/projects/bdpn/simulations/trees/BDPN/tree.101.log', type=str, help="Nwk")
    parser.add_argument('--nwk', default='/home/azhukova/projects/bdpn/simulations/trees/BDPN/tree.101.nwk', type=str, help="Log")
    parser.add_argument('--min_tips', default=5000, type=int, help="Min number of tips")
    parser.add_argument('--max_tips', default=10000, type=int, help="Max number of tips")
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=params.nwk.replace('.nwk', '.txt'), filemode='w+')

    R0 = random_float(1, 5)
    psi = random_float(1 / 20, 1 / 5)
    la = psi * R0
    phi = psi * random_float(50, 500)
    rho = random_float(0.1, 0.9)
    upsilon = random_float(0.01 / rho, 0.9)

    print(la, psi, phi, rho, upsilon)

    model = PNModel(model=BirthDeathModel(p=rho, la=la, psi=psi), upsilon=upsilon, partner_removal_rate=phi)

    forest, (total_tips, u, T), _ = generate(model, params.min_tips, params.max_tips)

    save_forest(forest, params.nwk)
    save_log(model, total_tips, T, u, params.log)


