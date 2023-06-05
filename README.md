# bdpn

Estimator of BDPN model parameters from phylogenetic trees. 

## BD-PN
5 parameters:
* λ -- transmission rate
* ψ -- removal rate
* p -- sampling probability upon removal
* p<sub>n</sub> -- probability to notify the last partner upon sampling
* ψ<sub>pn</sub> -- removal rate after notification

Epidemiological parameters:
* R<sub>0</sub>=λ/ψ -- reproduction number
* 1/ψ -- infectious time

## Installation
To install bdpn:
```bash
pip3 install bdpn
```

## Usage
### Command line 

## BDPN
The following command estimated the BDPN parameters for a given tree tree.nwk and a given sampling probability p=0.4, 
and saves the estimated parameters to a comma-separated file estimates.csv:
```bash
bdpn_infer --p 0.4 --nwk tree.nwk --log estimates.csv
```
To seee detailed options, run:
```bash
bdpn_infer --help
```


### Python3
To estimate the BDPN parameters for a given tree tree.nwk and a given sampling probability p=0.4:
```python
from ete3 import Tree
from bdpn.parameter_estimator import optimize_likelihood_params
from bdpn.bdpn import loglikelihood, get_bounds_start

nwk = 'tree.nwk'
p = 0.4
tree = Tree(nwk)
la, psi, psi_n, rho, rho_n = optimize_likelihood_params(tree, input_parameters=[None, None, None, p, None],
                                loglikelihood=loglikelihood, get_bounds_start=get_bounds_start)
print('Found BDPN params: {}'.format([la, psi, psi_n, rho, rho_n]))
```