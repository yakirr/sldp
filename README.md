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

## Errata
### Gene-set enrichment method for SLDP results
In the published paper, we described a gene-set enrichment method for assessing whether a genome-wide signed relationship between an annotation and a trait is stronger in areas of the genome that are near a gene set of interest. The method was described in terms of a vector `s` that summarizes the gene-set of interest and a vector `q` that summarizes the SLDP association of interest. Both `s` and `q` have one entry per LD block: the `i`-th entry of `s` contains the number genes from the gene set that lie in the `i`-th LD block, and the `i`-th entry of `q` contains the estimated covariance across SNPs in the `i`-th LD block between the signed LD profile of the annotation in question and summary statistics of the trait in question. There are two errata related to this analysis, which we describe below.

#### Description of gene-set enrichment analysis statistic
In the paper, we stated that statistic we compute is

![equation](https://latex.codecogs.com/png.latex?a%20%3A%3D%20%5Cfrac%7B%5Csum_i%20s_iq_i%7D%7B%5Csum_i%20s_i%7D)

that is, the weighted average of `q` across the LD blocks with non-zero values of `s`. However, the statistic that is used in actuality is

![equation](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Csum_i%20s_iq_i%7D%7B%5Csum_i%20s_i%7D%20-%20%5Cfrac%7B%5Csum_i%20%5Cmathbf%7B1%7D%28s_i%3D0%29%20q_i%7D%7B%5Csum_i%20%5Cmathbf%7B1%7D%28s_i%3D0%29%7D)

that is, we take the difference between the weighted average of `q` across the LD blocks with non-zero values of `s` on the one hand, and the average of `q` across the LD blocks in which `s` is zero on the other hand.

#### Computation of empirical p-values in gene-set enrichment analysis
Our gene-set enrichment procedure computed p-values by shuffling `s` over LD blocks. However, the code that produced our published results computed a simple average of `q` rather than a weighted average when computing the statistic for the null distribution. Fixing the bug led to qualitatively similar but not identical results. For more detail, download the [corrected version of Supplementary Table 10](https://data.broadinstitute.org/alkesgroup/SLDP/errata/corrected_SuppTable10.xlsx) that lists the published and corrected p- and q-values of the gene-set enrichments highlighted in our publication.

## Citation

If you use `sldp`, please cite

[Reshef, et al. Detecting genome-wide directional effects of transcription factor binding on polygenic disease risk.
Nature Genetics, 2018.](https://www.nature.com/articles/s41588-018-0196-7)
