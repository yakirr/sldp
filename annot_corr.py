from __future__ import print_function, division
import argparse
import numpy as np
import pandas as pd
import pyutils.pretty as pretty
import gprim.annotation as ga
import gprim.dataset as gd
import pyutils.memo as memo
import gc
import time
import resource


def mem():
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

def get_p_empirical(q):
    print(len(q))
    p = 1; p_std = 100; num_per = int(20000000 / len(q))
    stat = q.sum()
    null = np.array([])
    while p_std > 0.3*p and p > 1e-6:
        rademachers = np.random.binomial(1,0.5,(num_per, len(q)))
        rademachers[rademachers == 0] = -1
        null = np.concatenate([null, rademachers.dot(q)])
        p = max((np.abs(null) > np.abs(stat)).sum(), 1) / len(null)
        p_std = np.sqrt(p*(1-p)/len(null))
        print('\t',len(null), p, p_std)
    return p

def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    results = pd.DataFrame()
    mem()

    t0 = time.time()

    for c in args.chroms:
        print('chr', c, 'reading maf')
        mhcmask = None
        maf = refpanel.frq_df(c).MAF.values
        CHR = None
        BP = None
        memo.reset(); gc.collect(); mem()

        print('reading names')
        names = np.concatenate([annot.names(c) for annot in annots]) # names of annotations
        print('allocating V')
        V = np.empty((len(maf), len(names)+1))
        V[:,0] = 1
        for i, (name, annot) in enumerate(zip(names, annots)):
            print(time.time()-t0, ':', i, 'reading annot', annot.filestem())
            if i == 0:
                a = pd.read_csv(annot.filestem()+str(c)+'.sannot.gz', sep='\t')
                BP = a.BP.values
                CHR = a.CHR.values
                V[:,i+1] = a[name].values
            else:
                V[:,i+1] = pd.read_csv(annot.filestem()+str(c)+'.sannot.gz', sep='\t',
                        usecols=[name])[name].values
            mem()
        memo.reset(); gc.collect(); mem()

        if mhcmask is None:
            print('creating mhcmask')
            mhcmask = (CHR == 6)&(BP >= mhc[0])&(BP <= mhc[1])

        print('zeroing out mhc')
        V[mhcmask, :] = 0

        if not args.per_norm_genotype:
            print('adjusting for maf')
            V[:,1:] = V[:,1:]*np.sqrt(2*maf*(1-maf))[:,None]
        mem()

        cols = np.concatenate([['maf'],names])
        print(time.time()-t0, ': computing mean')
        Mmean = V.sum(axis=0).reshape((1,-1))
        pd.DataFrame(Mmean, columns=cols).to_csv(
                args.outfile_stem+str(c)+'.Mmean', sep='\t', index=False)
        print(time.time()-t0, ': computing cov')
        Mcov = V.T.dot(V)
        pd.DataFrame(Mcov, columns=cols).to_csv(
                args.outfile_stem+str(c)+'.Mcov', sep='\t', index=False)
        print(time.time()-t0, ': computing nz')
        nz = (V!=0).astype(int).T.dot((V!=0).astype(int))
        pd.DataFrame(nz, columns=cols).to_csv(
                args.outfile_stem+str(c)+'.nz', sep='\t', index=False)
        print(time.time()-t0, ': done')

        memo.reset(); gc.collect(); mem()

    print('writing results')
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/UwGm12878Ctcf/prod0.lfc.',
                '/groups/price/yakir/data/annot/basset/BroadDnd41Ctcf/prod0.lfc.'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--chroms', nargs='+', default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
