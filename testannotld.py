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
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    names = np.concatenate([annot.names(22) for annot in annots]) # names of annotations
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
        snps = ga.smart_merge([annot.sannot_df(c) for annot in annots],
                fail_if_nonmatching=True, drop_from_y=['BP','CM','A1','A2','CHR'])
        print(len(snps), 'snps in annot', len(snps.columns), 'columns, including metadata')

        # adjust for maf
        if not args.per_norm_genotype:
            print('adjusting for maf')
            maf = refpanel.frq_df(c)
            snps[names] = snps[names].values*np.sqrt(
                    2*maf.MAF.values*(1-maf.MAF.values))[:,None]
            del maf

        # restrict to ld blocks in this chr and process them in chunks
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
            V = ldblocksnps[names].values
            if (V**2).sum() == 0: # skip it no snps with non-zero v
                continue

            # read genotypes
            X = refpanel.stdX(c, (start_ind, end_ind))

            for j,n in enumerate(names):
                v = V[:,j]
                if np.linalg.norm(v) == 0:
                    continue

                # compute ldblock results
                Xv = X.dot(v)
                vTRv = Xv.T.dot(Xv) / X.shape[0]

                ldblockresults = ldblockresults.append({
                    'annot':n,
                    'locus_num':i,
                    'chr':ldblocks.loc[i,'chr'],
                    'start':ldblocks.loc[i,'start'],
                    'end':ldblocks.loc[i,'end'],
                    'vTRv':vTRv,
                    'v2':v.dot(v),
                    'suppVo':(v!=0).sum(),
                    'normV_4':np.linalg.norm(v, ord=4),
                    'normV_2':np.linalg.norm(v, ord=2),
                    'normV_1':np.linalg.norm(v, ord=1),
                    'normV_2o4':(np.linalg.norm(v, ord=2)**2)/(np.linalg.norm(v, ord=4)**2)},
                    ignore_index=True)
            del V; del ldblocksnps
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
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/UwGm12878Ctcf/prod0.lfc.',
                '/groups/price/yakir/data/annot/basset/BroadDnd41Ctcf/prod0.lfc.'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
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
