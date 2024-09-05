# bdpn

Estimator of BD-PN model parameters from phylogenetic trees and a non-parametric PN detection test. 



[//]: # ([![DOI:10.1093/sysbio/syad059]&#40;https://zenodo.org/badge/DOI/10.1093/sysbio/syad059.svg&#41;]&#40;https://doi.org/10.1093/sysbio/syad059&#41;)
[//]: # ([![GitHub release]&#40;https://img.shields.io/github/v/release/evolbioinfo/bdpn.svg&#41;]&#40;https://github.com/evolbioinfo/bdpn/releases&#41;)
[![PyPI version](https://badge.fury.io/py/bdpn.svg)](https://pypi.org/project/bdpn/)
[![PyPI downloads](https://shields.io/pypi/dm/bdpn)](https://pypi.org/project/bdpn/)
[![Docker pulls](https://img.shields.io/docker/pulls/evolbioinfo/bdpn)](https://hub.docker.com/r/evolbioinfo/bdpn/tags)

## BD-PN model

BD-PN model extends the classical birth-death (BD) model with incomplete sampling [[Stadler 2009]](https://pubmed.ncbi.nlm.nih.gov/19631666/), by adding partner notification (PN).
Under this model, infected individuals can transmit their pathogen with a constant rate λ, 
get removed (become non-infectious) with a constant rate ψ, 
and their pathogen can be sampled upon removal 
with a constant probability ρ. On top of that, in the BD-PN model, 
at the moment of sampling the sampled individual 
might notify their most recent partner with a constant probability υ. 
Upon notification, the partner is removed almost instantaneously (modeled via a constant notified
removal rate φ >> ψ) and their pathogen is sampled.

BD-PN model therefore has 5 parameters:
* λ -- transmission rate
* ψ -- removal rate
* ρ -- sampling probability upon removal
* υ -- probability to notify the last partner upon sampling
* φ -- removal (and sampling) rate after notification

These parameters can be expressed in terms of the following epidemiological parameters:
* R<sub>0</sub>=λ/ψ -- reproduction number
* 1/ψ -- infectious time
* 1/φ -- partner removal time

BD-PN model makes 3 assumptions:
1. only observed individuals can notify (instead of any removed individual);
2. notified individuals are always observed upon removal;
3. only the most recent partner can get notified.

For identifiability, BD-PN model requires one of the three BD model parameters (λ, ψ, ρ) to be fixed.

## BD-PN parameter estimator

The bdpn package provides a classical BD and a BD-PN model maximum-likelihood parameter estimator 
from a user-supplied time-scaled phylogenetic tree. 
User must also provide a value for one of the three BD model parameters (λ, ψ, or ρ). 
We recommend providing the sampling probability ρ, 
which could be estimated as the number of tree tips divided by the number of declared cases for the same time period.

## PN test

The bdpn package provides a non-parametric test detecting presence/absence of partner notification in a user-supplied time-scaled phylogenetic tree. 
It outputs a p-value. 

## Input data
One needs to supply a time-scaled phylogenetic tree in newick format. 
In the examples below we will use an HIV tree reconstructed from 200 sequences, 
published in [[Rasmussen _et al._ PLoS Comput. Biol. 2017]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005448), 
which you can find at [PairTree GitHub](https://github.com/davidrasm/PairTree) 
and in [hiv_zurich/Zurich.nwk](hiv_zurich/Zurich.nwk). 

## Installation

There are 4 alternative ways to run __bdpn__ on your computer: 
with [docker](https://www.docker.com/community-edition), 
[apptainer](https://apptainer.org/),
in Python3, or via command line (requires installation with Python3).



### Run in python3 or command-line (for linux systems, recommended Ubuntu 21 or newer versions)

You could either install python (version 3.9 or higher) system-wide and then install bdpn via pip:
```bash
sudo apt install -y python3 python3-pip python3-setuptools python3-distutils
pip3 install bdpn
```

or alternatively, you could install python (version 3.9 or higher) and bdpn via [conda](https://conda.io/docs/) (make sure that conda is installed first). 
Here we will create a conda environment called _bdpnenv_:
```bash
conda create --name bdpnenv python=3.9
conda activate bdpnenv
pip install bdpn
```


#### Basic usage in a command line
If you installed __bdpn__ in a conda environment (here named _bdpnenv_), do not forget to first activate it, e.g.

```bash
conda activate bdpnenv
```

Run the following commands to check for the presence of partner notification and estimate BD-PN model parameters.
The first command applies the PN test to a tree Zurich.nwk and saves the PN-test value to the file cherry_test.txt
The second command estimated the BD-PN parameters and their 95% CIs for this tree, assuming the sampling probability of 0.25, 
and saves the estimated parameters to a comma-separated file estimates_bdpn.csv.
The third command estimated the classical BD parameters and their 95% CIs for this tree, assuming the sampling probability of 0.25, 
and saves the estimated parameters to a comma-separated file estimates_bd.csv.
```bash
pn_test --nwk Zurich.nwk --log cherry_test.txt
bdpn_infer --nwk Zurich.nwk --ci --p 0.25 --log estimates_bdpn.csv
bd_infer --nwk Zurich.nwk --ci --p 0.25 --log estimates_bd.csv
```

#### Help

To see detailed options, run:
```bash
pn_test --help
bdpn_infer --help
bd_infer --help
```


### Run with docker

#### Basic usage
Once [docker](https://www.docker.com/community-edition) is installed, 
run the following command to estimate BD-PN model parameters:
```bash
docker run -v <path_to_the_folder_containing_the_tree>:/data:rw -t evolbioinfo/bdpn --nwk /data/Zurich.nwk --ci --p 0.25 --log /data/estimates.csv
```

This will produce a comma-separated file estimates.csv in the <path_to_the_folder_containing_the_tree> folder,
 containing the estimated parameter values and their 95% CIs (can be viewed with a text editor, Excel or Libre Office Calc).

#### Help

To see advanced options, run
```bash
docker run -t evolbioinfo/bdpn -h
```



### Run with apptainer

#### Basic usage
Once [apptainer](https://apptainer.org/docs/user/latest/quick_start.html#installation) is installed, 
run the following command to estimate BD-PN model parameters (from the folder where the Zurich.nwk tree is contained):

```bash
apptainer run docker://evolbioinfo/bdpn --nwk Zurich.nwk --ci --p 0.25 --log estimates.csv
```

This will produce a comma-separated file estimates.csv,
 containing the estimated parameter values and their 95% CIs (can be viewed with a text editor, Excel or Libre Office Calc).


#### Help

To see advanced options, run
```bash
apptainer run docker://evolbioinfo/bdpn -h
```


