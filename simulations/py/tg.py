from treesimulator.generator import generate
from treesimulator.mtbd_models import BirthDeathModel, PNModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
la, psi, p, psi_p, pn = (0.21666016107187575, 0.06376138425081969, 0.5614842560422901, 8.765449373319328, 0.2)
model = PNModel(model=BirthDeathModel(p=p, la=la, psi=psi), pn=pn, removal_rate=psi_p)
forest, (total_tips, u, T), _ = generate(model, 1000, 5000)