from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
from pybedtools import BedTool
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import pyutils.iter as pyit
import gprim.annotation as ga
import gprim.dataset as gd
import pyutils.memo as memo
import gc
import time
import resource

def mem():
    print('memory usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, 'Mb')

def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    mhcmask = None
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.')[0]
    annot = ga.Annotation(args.sannot_chr)
    annotR = ga.Annotation(args.sannotR_chr)
    results = pd.DataFrame()

    mem()
    print('getting h2g')
    info = pd.read_csv(args.bhat_chr+'info', sep='\t')
    h2g = info.h2g[0]
    print('It is:', h2g)

    print('reading maf')
    maf = np.concatenate([refpanel.frq_df(c).MAF.values for c in args.chroms])
    memo.reset(); gc.collect(); mem()

    print('reading sumstats, specifically:', args.use)
    ss = np.concatenate([
        pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t',
            usecols=[args.use])[args.use].values
        for c in args.chroms])
    M = len(ss)
    print('getting typed snps')
    typed = np.isfinite(ss)
    print('restricting to typed snps, of which there are', typed.sum())
    ahat = ss[typed]

    maft = maf[typed]
    mem()

    t0 = time.time()
    names = annot.names(22) # names of annotations
    namesR = annotR.names(22)
    print(time.time()-t0, ': reading annot', annot.filestem())
    a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)
    aR = pd.concat([annotR.sannot_df(c) for c in args.chroms], axis=0)

    print('restricting to typed snps only')
    a = a[typed]
    aR = aR[typed]
    memo.reset(); gc.collect(); mem()

    print('creating mhcmask')
    CHR = a.CHR.values
    BP = a.BP.values
    mhcmask = (CHR == 6)&(BP >= mhc[0])&(BP <= mhc[1])
    maft = maft[~mhcmask]

    print('creating V')
    print('there are', (~np.isfinite(a[names].values)).sum(), 'nans in the annotation')
    V = a[names].values
    RV = aR[namesR].values
    del a; del aR; gc.collect(); mem()

    print('throwing out mhc')
    V = V[~mhcmask,:]
    RV = RV[~mhcmask,:]
    ahat = ahat[~mhcmask]

    if not args.per_norm_genotype:
        print('adjusting for maf')
        V = V*np.sqrt(2*maft*(1-maft))[:,None]

    ## do the math ##
    VTRV = V.T.dot(RV)
    VTRVinv = np.linalg.inv(VTRV)
    VTRbeta = V.T.dot(ahat)
    betaTRbeta = h2g
    R2 = VTRbeta.T.dot(VTRVinv.dot(VTRbeta)) / betaTRbeta

    print('R2 =', R2)
    print('marginal R2 =', VTRbeta**2 / (np.diagonal(VTRV)*betaTRbeta))
    import pdb; pdb.set_trace()


#     for i, name in enumerate(names):
#         print(i, name)
#         v = V[:,i]
#         vjoint = Vjoint[:,i]

#         q = v*ahat
#         qjoint = vjoint*ahat
#         print(name, 'marginal coeff:', v.dot(ahat)/v.dot(v),
#                 'marginal z:', q.sum()/np.linalg.norm(q),
#                 'joint coeff:', vjoint.dot(ahat),
#                 'joint z:', qjoint.sum()/np.linalg.norm(qjoint))

#         import pdb; pdb.set_trace()

#     del V; memo.reset(); gc.collect(); mem()

#     print('writing results')
#     results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False)
#     print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/CD.KG3_50.0/',
            help='one or more paths to .bhat.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', #required=True,
            default='/groups/price/yakir/data/annot/baseline/functional/',
            help='path to gzipped sannot file containing V, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--sannotR-chr', #required=True,
            default='/groups/price/yakir/data/annot/baseline/functional/KG3_50.R.',
            help='one or more paths to gzipped sannot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--use', default='ahat',
            help='which column from the processed sumstats file to correlate with the annot.')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that V is in units of per normalized genotype rather than per ' +\
                    'allele. NOTE: this applies only to sannot-chr and not to sannotR-chr')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
