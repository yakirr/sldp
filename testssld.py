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


def main(args):
    # basic initialization
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.KG3')[0]
    ldblockresults = pd.DataFrame(columns=['pheno','annot','locus_num','chr','start','end'])

    # read in ldblocks and remove ones that overlap mhc
    mhc = [25684587, 35455756]
    ldblocks = pd.read_csv(args.ldblocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]
    print(len(ldblocks), 'ldblocks after removing MHC')

    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading annots for chr', c, 'of', args.chroms)
        # read in all annotations for this chr. we assume that all annots have same snps with
        # same coding as each other and as refpanel, in same order
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in annot', len(snps.columns), 'columns, including metadata')

        # read in sumstats
        ss = pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t')
        snps = ga.smart_merge(snps, ss, fail_if_nonmatching=True)
        snps['typed'] = snps.bhat.notnull()
        snps.loc[~snps.typed,'bhat'] = 0
        snps.loc[~snps.typed,'ahat'] = 0

        # restrict to ld blocks in this chr and process them one by one
        chr_ldblocks = ldblocks[ldblocks.chr=='chr'+str(c)]
        ldblockstarts_ind = np.searchsorted(snps.BP.values, chr_ldblocks.start.values)
        ldblockends_ind = np.searchsorted(snps.BP.values, chr_ldblocks.end.values)
        print('{} : chr {} snps {} - {}'.format(
            time.time()-t0, c, ldblockstarts_ind[0], ldblockends_ind[-1]))
        for i, start_ind, end_ind in zip(
                chr_ldblocks.index, ldblockstarts_ind, ldblockends_ind):
            print('{} : processing block {}, {}:{:.2f}Mb-{:.2f}Mb, {} snps'.format(
                time.time()-t0, i, ldblocks.loc[i].chr,
                ldblocks.loc[i].start/1e6, ldblocks.loc[i].end/1e6, end_ind-start_ind))
            ldblocksnps = snps.iloc[start_ind:end_ind]
            bhat = ldblocksnps.bhat.values
            ahat = ldblocksnps.ahat.values

            # read genotypes
            X = refpanel.stdX(c, (start_ind, end_ind))

            # compute ldblock results
            Xbhat = X.dot(bhat)
            Xahat = X.dot(ahat)
            bhatTRbhat = Xbhat.T.dot(Xbhat) / X.shape[0]
            ahatTRahat = Xahat.T.dot(Xahat) / X.shape[0]

            ldblockresults = ldblockresults.append({
                'pheno':nice_ss_name,
                'locus_num':i,
                'chr':ldblocks.loc[i,'chr'],
                'start':ldblocks.loc[i,'start'],
                'end':ldblocks.loc[i,'end'],
                'bhatTRbhat':bhatTRbhat,
                'ahatTRahat':ahatTRahat,
                'bhat2':bhat.dot(bhat),
                'ahat2':ahat.dot(ahat),
                'numtyped':ldblocksnps.typed.sum()},
                ignore_index=True)
            del ahat; del bhat; del ldblocksnps
            gc.collect()
        del snps; memo.reset(); gc.collect()

    print('writing block-by-block results')
    ldblockresults.to_csv(args.outfile_stem + '.ldblocks', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/UKBB_body_HEIGHTz.KG3_0.1/',
            help='one or more paths to .bhat.gz files, without chr number or extension')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ldblocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per ld block')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--chroms', nargs='+', default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
