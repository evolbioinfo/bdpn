# bdpn

Estimator of BDPN model parameters from phylogenetic trees. 

## BD-PN
5 parameters:
* λ -- transmission rate
* ψ -- removal rate
* p -- sampling probability upon removal
* p<sub>n</sub> -- probability to notify the last partner upon sampling
* ψ<sub>p</sub> -- removal rate after notification

Epidemiological parameters:
* R<sub>0</sub>=λ/ψ -- reproduction number
* 1/ψ -- infectious time

## Installation
To install bdpn:
```bash
pip3 install bdpn
```

## Usage in command line 

## BDPN paramer estimation
The following command estimated the BDPN parameters for a given tree tree.nwk and a given sampling probability p=0.4, 
and saves the estimated parameters to a comma-separated file estimates.csv:
```bash
bdpn_infer --p 0.4 --nwk tree.nwk --log estimates.csv
```
To see detailed options, run:
```bash
bdpn_infer --help
```

## PN test
The applies the PN test to a given tree tree.nwk and saves the PN-test value to the file cherry_test.txt:
```bash
pn_test --nwk tree.nwk --log cherry_test.txt --block_size 100
```
To see detailed options, run:
```bash
pn_test --help
```

