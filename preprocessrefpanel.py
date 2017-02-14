from __future__ import print_function, division
import argparse
import os, gzip
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
from pybedtools import BedTool
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import gprim.annotation as ga
import gprim.dataset as gd
import pyutils.memo as memo
import gc
import time


def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)

    # read in ld blocks, remove MHC, read SNPs to print
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]
    print(len(ldblocks), 'loci after removing MHC')
    print_snps = pd.read_csv(args.print_snps, header=None, names=['SNP'])
    print_snps['printsnp'] = True
    print(len(print_snps), 'print snps')

    log = pd.DataFrame()
    for c in args.chroms:
        print('loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        snps = pd.merge(snps, print_snps, on='SNP', how='left')
        snps.printsnp.fillna(False, inplace=True)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        for ldblock, X, meta in refpanel.block_data(ldblocks, c, meta=snps):
            if meta.printsnp.sum() == 0:
                print('no print snps found in this block')
                continue
            mask = meta.printsnp.values
            X_ = X[:,mask]

            print('\tcomputing SVD of R_print')
            U_, svs_, _ = np.linalg.svd(X_.T); svs_ = svs_**2 / X_.shape[0]
            k = np.argmax(np.cumsum(svs_)/svs_.sum() >= args.spectrum_percent / 100.)
            print('\treduced rank of', k, 'out of', meta.printsnp.sum(), 'printed snps')
            np.savez('{}{}.R'.format(args.outstem, ldblock.name), U=U_[:,:k], svs=svs_[:k])

            print('\tcomputing R2_print')
            R2 = X_.T.dot(X.dot(X.T)).dot(X_) / X.shape[0]**2
            print('\tcomputing SVD of R2_print')
            R2_U, R2_svs, _ = np.linalg.svd(R2)
            k = np.argmax(np.cumsum(R2_svs)/R2_svs.sum() >= args.spectrum_percent / 100.)
            print('\treduced rank of', k, 'out of', meta.printsnp.sum(), 'printed snps')
            np.savez('{}{}.R2'.format(args.outstem, ldblock.name),
                    U=R2_U[:,:k], svs=R2_svs[:k])

        del snps; memo.reset(); gc.collect()
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectrum-percent', type=float, default=95)
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--print-snps',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/'+\
                    '1000G_hm3_noMHC.rsid')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--outstem', default='/groups/price/ldsc/reference_files/' + \
            '1000G_EUR_Phase3/svds_95percent/',
            help='stem for output filenames')
    parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int)

    args = parser.parse_args()
    main(args)
