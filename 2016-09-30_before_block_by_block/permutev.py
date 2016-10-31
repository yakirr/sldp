from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import statutils.qnorm as qnorm
import gprim.annotation as ga
import gc


def main(args):
    chroms = range(1,23)
    # read in annotation and determine which column we're interested in
    print('reading annot')
    annot = ga.Annotation(args.sannot_chr)
    a_name = annot.names(1)[0]
    a = pd.concat([annot.sannot_df(c) for c in chroms], axis=0)
    print(len(a), 'snps')

    # read in maf information and adjusting annot
    print('reading maf and adjusting v')
    maf = pd.concat([pd.read_csv(args.bfile_chr+str(c)+'.frq', delim_whitespace=True)
        for c in chroms], axis=0)
    print(len(maf), 'snps')
    a[a_name] = a[a_name]*np.sqrt(2*maf.MAF.values*(1-maf.MAF.values))
    del maf

    # remove MHC
    print('removing mhc')
    mhc = ((a.CHR == 6) & (a.BP >= 25500000) & (a.BP <= 35500000)).values
    a = a[~mhc]
    del mhc
    print(len(a), 'snps')

    # quantile normalize non-zero part of annotation
    print('quantile normalizing')
    mask = (a[a_name] != 0).values
    a.loc[mask, a_name] = qnorm.qnorm(a.loc[mask, a_name].values)

    # read in ld scores
    # TODO

    results = pd.DataFrame(columns=['annot','size_v_typed','score','null_mean','null_std',
                                        'std_gencov','z_perm','p_perm','v2overN'])
    for sumstats_name in args.sumstats:
        nice_name = sumstats_name.split('/')[-1].split('.sumstats.gz')[0]
        print('\n', nice_name)
        # read in sumstats
        print('reading sumstats')
        ss = pd.read_csv(sumstats_name, sep='\t')
        ss['ahat'] = ss.Z / np.sqrt(ss.N)
        N = np.nanmean(ss.N.values)
        print(len(ss), 'snps')
        print(N, 'individuals')

        # reconcile the two
        print('reconciling')
        ss = ga.reconciled_to(a, ss, ['ahat'])

        # filter to where v and alphahat are non-zero
        print('filtering to non-zero v and typed')
        mask = (a[a_name] != 0).values & (ss.ahat != 0).values
        print('done mask')
        a_ = a[mask]
        print('done a')
        ss_ = ss[mask]
        print('done ss')
        # a_ = a
        # ss_ = ss
        results.loc[nice_name,'size_v_typed'] = np.sum(mask)

        results.annot[nice_name] = a_name
        results.score[nice_name] = ss_.ahat.values.dot(a_[a_name].values)

        print('testing nulls')
        nulls = np.array([
            ss_.ahat.values.dot(np.random.permutation(a_[a_name].values))
            for i in range(args.num_perms)])

        results.null_mean[nice_name] = np.mean(nulls)
        results.null_std[nice_name] = np.std(nulls)
        results.v2overN[nice_name] = np.linalg.norm(a_[a_name].values)**2/N
        results.std_gencov[nice_name] = np.sqrt(np.var(nulls) - results.v2overN[nice_name])
        results.z_perm[nice_name] = (results.score[nice_name] - np.mean(nulls))/np.std(nulls)
        results.p_perm[nice_name] = stats.chi2.sf(results.z_perm[nice_name]**2, 1)
        print(results.loc[nice_name])
        del ss; del ss_; del mask; del a_; gc.collect()

    results.to_csv(args.outfile, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', required=True,
            help='path to an output file stem')
    parser.add_argument('--sumstats', nargs='+', required=True,
            help='path to sumstats.gz files, including extension')
    parser.add_argument('--N-thresh', type=float, default=1.0,
            help='this times the 90th percentile N is the sample size threshold')
    parser.add_argument('--num-perms', type=int, default=10000,
            help='the number of permutation tests to run')
    parser.add_argument('--sannot-chr', required=True,
            help='path to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ld-breakpoints',
            default='/groups/price/yakir/data/reference/pickrell_breakpoints.hg19.eur.bed',
            help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                    ' breakpoint')
    parser.add_argument('--mhc-path',
            default='/groups/price/yakir/data/reference/hg19.MHC.bed',
            help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                    ' breakpoint')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--full-ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/allSNPs/'+\
                    '1000G.EUR.QC.',
            help='ld scores to and at all refpanel snps. We assume these are the same snps '+\
                    'in the same order as the reference panel. These ld scores will be '+\
                    'used to weight the quantity being estimated if weight-ld flag is used.')
    parser.add_argument('-weight-ld', action='store_true', default=False,
            help='weight the quantity being estimated by the --full-ldscores-chr data.')

    args = parser.parse_args()
    main(args)


