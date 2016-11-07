from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
import statutils.sig as sig
import statutils.vis as vis
import matplotlib.pyplot as plt
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('phenotype')
parser.add_argument('--thresh', type=float, default=9, help='min number of v snps')
args = parser.parse_args()

# read in ENCODE experiment information
meta = pd.read_csv('/groups/price/yakir/data/annot/basset/ids_metadata.dict',
        sep='\t', index_col='id')

blocks = pd.concat([
    pd.read_csv(f, sep='\t')
    for f in glob(args.phenotype + '.*.blocks')
    ], axis=0)
mask = (blocks.suppVo > args.thresh).values
blocks = blocks[mask]
blocks.annot = [a.split(',')[0] for a in blocks.annot]
print(len(blocks), 'blocks after filtering on # of v snps')

blocks['z'] = blocks.vTahat / np.sqrt(blocks.voTRvo_N)
blocks['zf'] = blocks.vfTahat / np.sqrt(blocks.vfTRvf_N)
blocks['p'] = st.chi2.sf(blocks.z.values**2, 1)
blocks['pf'] = st.chi2.sf(blocks.zf.values**2, 1)

p_piv = blocks.pivot('annot','block_num','pf')
for a, row in p_piv.iloc[:5].iterrows():
    print(a, sig.fdr(row.values))
    vis.qqplot(row.values)
plt.show()

# piv_num = blocks.pivot('annot','block_num','vfTahat')
# piv_denom = blocks.pivot('annot','block_num','vfTRvf_N')
# zscores = piv_num.sum(axis=1) / np.sqrt(piv_denom.sum(axis=1))
# pvals = st.chi2.sf(zscores**2, 1)
# print(sig.fdr(pvals))
# vis.qqplot(pvals); plt.show()

thresh, numsig = sig.fdr(blocks.p.values)
threshf, numsigf = sig.fdr(blocks.pf.values)
bothsig = (blocks.p <= thresh) & (blocks.pf <= threshf)
blocks.loc[~(blocks.pf <= threshf), 'zf'] = np.nan
# blocks.loc[~(blocks.p <= thresh), 'z'] = np.nan
# blocks.loc[~bothsig, 'zf'] = np.nan
print(numsig, numsigf, bothsig.sum(), 'passed significance')

piv = blocks.pivot('annot','block_num','zf')

piv['sort_key'] = [meta.experiment[id]+' '+meta.cell_line[id] for id,_ in piv.iterrows()]
piv = piv.sort(columns='sort_key').drop(['sort_key'], axis=1)
plt.matshow(piv.values)
plt.show()

