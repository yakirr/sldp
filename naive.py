from __future__ import print_function, division
import pandas as pd
import numpy as np
import scipy.stats as st
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--sannot-chr', nargs='+')
parser.add_argument('--outfile-stem')
parser.add_argument('--ssjk-chr')
parser.add_argument('--T', default=100000, type=int)
args = parser.parse_args()

print('reading sumstats')
ss = pd.concat([pd.read_csv(args.ssjk_chr+str(c)+'.ss.jk.gz', sep='\t') for c in range(1,23)],
        axis=0)

results = pd.DataFrame()

for sannot in args.sannot_chr:
    annot = pd.concat([pd.read_csv(sannot+str(c)+'.RV.gz', sep='\t') for c in range(1,23)],
            axis=0)
    a = annot.columns[3]
    print(a)
    x = annot[a].values; y = ss.Winv_ahat_I.values
    mask = np.isfinite(x)&np.isfinite(y)
    x = x[mask]; y = y[mask]

    ind = np.concatenate([np.arange(0, len(x), int(len(x)/300)), [len(x)]])
    q = np.array([x[i:j].dot(y[i:j]) for i,j in zip(ind[:-1],ind[1:])])
    score = q.sum()

    s = (-1)**np.random.binomial(1,0.5,size=(args.T, len(q)))
    null = s.dot(q)
    p = ((np.abs(null) >= np.abs(score)).sum()) / float(100000)
    p = min(max(p,1./args.T), 1)
    se = np.abs(score)/np.sqrt(st.chi2.isf(p,1))
    results = results.append({
        'annot':annot.columns[4],
        'pheno':args.ssjk_chr.split('/')[-1],
        'naive_score':score,
        'naive_z':score/se,
        'naive_p':p},
        ignore_index=True)
    print(results.iloc[-1])
    del s; del null; gc.collect()

print('writing results')
print(results)
results.to_csv(args.outfile_stem + '.naiveresults', sep='\t', index=False, na_rep='nan')
print('done')
