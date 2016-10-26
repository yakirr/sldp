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
import pyutils.iter as pyit
import gprim.annotation as ga
import gprim.dataset as gd
import pyutils.memo as memo
import gc
import time


def main(args):
    # basic initialization
    nrows=args.nrows
    chunksize = 25
    refpanel = gd.Dataset(args.bfile_chr)

    # read in ld blocks, leaving in MHC
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])

    # read sumstats
    print('reading sumstats', args.sumstats_stem)
    ss = pd.read_csv(args.sumstats_stem+'.sumstats.gz', sep='\t', nrows=nrows)
    ss = ss[ss.Z.notnull() & ss.N.notnull()]
    ss['ahat'] = ss.Z / np.sqrt(ss.N)
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    print(len(ss), 'snps after removing outliers')

    # estimate heritability using aggregate estimator
    print('reading in ld scores')
    l2 = pd.concat([pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in range(1,23)], axis=0)
    print(len(l2), 'snps')
    ssl2 = pd.merge(ss, l2, on='SNP', how='inner')
    print(len(ssl2), 'snps after merge')
    mhc = [25684587, 35455756]
    ssl2 = ssl2[~((ssl2.CHR == 6) & (ssl2.BP >= mhc[0]) & (ssl2.BP < mhc[1]))]
    print(len(ssl2), 'snps after removing mhc')
    sample_size_corr = np.sum(1./ssl2.N.values)
    sigma2g = (np.linalg.norm(ssl2.ahat.values)**2 - sample_size_corr) \
            / np.sum(ssl2.L2.values)
    sigma2g = min(max(0,sigma2g), 1/len(ssl2))
    h2g = sigma2g*len(ssl2); sigma2e = 1-h2g
    print('h2g estimated at:', h2g, 'sigma2g:', sigma2g)
    del l2; del ssl2; gc.collect()

    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # merge annot and sumstats, and create Vo := V zeroed out at unobserved snps
        print('reconciling')
        snps = ga.reconciled_to(snps, ss, ['ahat'], othercolnames=['N'], missing_val=np.nan)
        snps['typed'] = snps.N.notnull()
        snps['bhat'] = np.nan

        # restrict to ld blocks in this chr and process them in chunks
        chr_blocks = ldblocks[ldblocks.chr=='chr'+str(c)]
        for block_nums in pyit.grouper(chunksize, range(len(chr_blocks))):
            # get ld blocks in this chunk, and indices of the snps that start and end them
            chunk_blocks = chr_blocks.iloc[block_nums]
            blockstarts_ind = np.searchsorted(snps.BP.values, chunk_blocks.start.values)
            blockends_ind = np.searchsorted(snps.BP.values, chunk_blocks.end.values)
            print('{} : chr {} snps {} - {}'.format(
                time.time()-t0, c, blockstarts_ind[0], blockends_ind[-1]))

            # read in refpanel for this chunk, and find the relevant annotated snps
            Xchunk = refpanel.stdX(c, (blockstarts_ind[0], blockends_ind[-1]))
            snpschunk = snps.iloc[blockstarts_ind[0]:blockends_ind[-1]]
            print('read in chunk')

            # calibrate ld block starts and ends with respect to the start of this chunk
            blockends_ind -= blockstarts_ind[0]
            blockstarts_ind -= blockstarts_ind[0]
            for i, start_ind, end_ind in zip(
                    chunk_blocks.index, blockstarts_ind, blockends_ind):
                print(time.time()-t0, ': processing ld block',
                        i, ',', end_ind-start_ind, 'snps')
                X = Xchunk[:, start_ind:end_ind]
                snpsblock = snpschunk.iloc[start_ind:end_ind]
                ahat = snpsblock.ahat.values
                Nr = X.shape[0]

                def mult_Rinv(A, X_t):
                    B = A / args.Lambda - \
                        1/args.Lambda**2 * X_t.T.dot(
                            np.linalg.solve(
                                Nr*np.eye(Nr) + X_t.dot(X_t.T)/args.Lambda,
                                X_t.dot(A)))
                    return B * (1+args.Lambda)

                # compute betahat
                print('inverting')
                t = snpsblock.typed.values
                X_t = X[:,t]; ahat_t = ahat[t]
                B = mult_Rinv(ahat_t.reshape((-1,1)), X_t)
                snps.loc[snpschunk.iloc[start_ind:end_ind].index[t],'bhat'] = B[:,-1]

                # store results
                del X; del ahat; del snpsblock
                del X_t; del B
            del Xchunk; del snpschunk; gc.collect()

        print('writing finemapped sumstats')
        dirname = args.sumstats_stem + '.' + args.refpanel_name + '_' + str(args.Lambda)
        fs.makedir(dirname)
        with gzip.open('{}/{}.bhat.gz'.format(dirname, c), 'w') as f:
            snps[['SNP','A1','A2','ahat','bhat','N']].to_csv(f, index=False, sep='\t')
        del snps; memo.reset(); gc.collect()

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumstats-stem', #required=True,
            default='/groups/price/yakir/data/sumstats/UKBB_body_HEIGHTz',
            help='path to sumstats.gz files, not including ".sumstats.gz" extension')
    parser.add_argument('--Lambda', type=float, default=0.1)
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to estimate heritability')
    parser.add_argument('--refpanel-name', default='KG3',
            help='suffix added to the directory created for storing output')
    parser.add_argument('--nrows', default=None, type=int)
    parser.add_argument('--chroms', nargs='+', default=range(1,23))

    args = parser.parse_args()
    main(args)
