# bdpn

Estimator of BDPN model parameters from phylogenetic trees. 

## BD-PN model
BD-PN model has 5 parameters:
* λ -- transmission rate
* ψ -- removal rate
* p -- sampling probability upon removal
* p<sub>n</sub> -- probability to notify the last partner upon sampling
* ψ<sub>p</sub> -- removal rate after notification

As in the basic birth-death model [[Stadler 2009]](https://pubmed.ncbi.nlm.nih.gov/19631666/),
all the individuals are in the same state. 
They can transmit with a constant rate λ, 
get removed with a constant rate ψ, 
and their pathogen can be sampled upon removal 
with a constant probability p. On top of that, in the BD-PN model, 
at the moment of sampling the sampled individual 
might notify their most recent partner with a constant probability p<sub>n</sub>. 
Upon notification, the partner is removed almost instantaneously (modeled via a constant notified
removal rate ψ<sub>p</sub> >> ψ).

BD-PN model makes 4 assumptions:
1. only observed individuals can notify (instead of any removed individual);
2. notified individuals are always observed upon removal;
3. after the notification, notified individuals do not transmit further;
4. only the most recent partner can get notified.

Epidemiological parameters:
* R<sub>0</sub>=λ/ψ -- reproduction number
* 1/ψ -- infectious time

## Installation
To install bdpn:
```bash
python3 setup.py install
```

## Usage in command line 

## BDPN parameter estimation
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

