# SLDP (Signed LD Profile) regression

SLDP regression is a method for looking for a directional effect of a signed functional annotation on a heritable trait using GWAS summary statistics. This repository contains code for the SLDP regression method as well as tools required for preprocessing data for use with SLDP regression.

## Installation

First, make sure you have a python distribution installed that includes scientific computing packages like numpy/scipy/pandas as well as the package manager pip; we recommend [Anaconda](https://store.continuum.io/cshop/anaconda/).

To install `sldp`, type the following command.
```  
pip install sldp
```
This should install both sldp as well as any required packages, such as [gprim](https://github.com/yakirr/gprim) and [ypy](https://github.com/yakirr/ypy).

If you prefer to install `sldp` without pip, just clone this repository, together with [gprim](https://github.com/yakirr/gprim) and [ypy](https://github.com/yakirr/ypy), and add an entry for each into your python path.


## Getting started

To verify that the installation went okay, run
```
sldp -h
```
to print a list of all command-line options. If this command fails, there was a problem with the installation.

Once this works, take a look at our [wiki](https://github.com/yakirr/sldp/wiki) for a short tutorial on how to use `sldp`.


## Where can I get signed LD profiles?

You can download signed LD profiles (as well as raw signed functional annotations) for ENCODE ChIP-seq experiments from the [sldp data page](https://data.broadinstitute.org/alkesgroup/SLDP/). These signed LD profiles were created using 1000 Genomes Phase 3 Europeans as the reference panel.

## Where can I get reference panel information such as SVDs of LD blocks and LD scores?

You can download all required reference panel information, computed using 1000 Genomes Phase 3 Europeans, from the [sldp data page](https://data.broadinstitute.org/alkesgroup/SLDP/).


## Citation

If you use `sldp`, please cite

[Reshef, et al. Detecting genome-wide directional effects of transcription factor binding on polygenic disease risk.
BiorXiv, 2017.](https://www.biorxiv.org/content/early/2017/10/17/204685)
