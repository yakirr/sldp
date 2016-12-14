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
    nrows=args.nrows
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.KG3')[0]
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    names = np.concatenate([annot.names(22) for annot in annots]) # names of annotations
    locusresults = pd.DataFrame(columns=['pheno','annot','locus_num','chr','start','end'])

    # read in loci and remove ones that overlap mhc
    mhc = [25684587, 35455756]
    loci = pd.read_csv(args.loci, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (loci.chr == 'chr6') & (loci.end > mhc[0]) & (loci.start < mhc[1])
    loci = loci[~mhcblocks]
    print(len(loci), 'loci after removing MHC')

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

        # merge annot and sumstats, and create Vo := V zeroed out at unobserved snps
        print('reconciling')
        ss = pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t')
        snps = ga.smart_merge(snps, ss, fail_if_nonmatching=True)
        snps['typed'] = snps.bhat.notnull()

        # restrict to ld blocks in this chr and process them in chunks
        chr_loci = loci[loci.chr=='chr'+str(c)]
        locusstarts_ind = np.searchsorted(snps.BP.values, chr_loci.start.values)
        locusends_ind = np.searchsorted(snps.BP.values, chr_loci.end.values)
        print('{} : chr {} snps {} - {}'.format(
            time.time()-t0, c, locusstarts_ind[0], locusends_ind[-1]))
        for i, start_ind, end_ind in zip(
                chr_loci.index, locusstarts_ind, locusends_ind):
            print('{} : processing locus {}, {}:{:.2f}Mb-{:.2f}Mb, {} snps'.format(
                time.time()-t0, i, loci.loc[i].chr,
                loci.loc[i].start/1e6, loci.loc[i].end/1e6, end_ind-start_ind))
            locussnps = snps.iloc[start_ind:end_ind]
            t = locussnps.typed.values
            if t.sum() == 0: # skip loci with no typed snps
                continue
            V = locussnps[names].values[t]
            bhat = locussnps.bhat.values[t]
            ahat = locussnps.ahat.values[t]
            if args.no_finemap:
                bhat = ahat
            N = locussnps.N.values[t]

            def var_s(v, b): # variance of v.dot(bhat)
                return (v**2).dot(b**2)
            def var_ss(v, b): # variance of v.dot(bhat)**2 - (v**2).dot(bhat**2)
                return 2*((v**2).dot(b**2)**2 - (v**4).dot(b**4))

            # store locusresults
            VTbhat = V.T.dot(bhat)

            for j,n in enumerate(names):
                v = V[:,j]
                if np.linalg.norm(v) == 0: # skip empty vs
                    continue

                vTbhat = VTbhat[j]
                maxv = np.max(np.abs(v))

                locusresults = locusresults.append({
                    'pheno':nice_ss_name,
                    'annot':n,
                    'locus_num':i,
                    'chr':loci.loc[i,'chr'],
                    'start':loci.loc[i,'start'],
                    'end':loci.loc[i,'end'],
                    'vTbhat':VTbhat[j], # sum statistic
                    'vTbhat_prod':VTbhat[j]**2 - (v**2).dot(bhat**2), # product statistic
                    'vTbhat_s2':(v**2).dot(bhat**2), # variance of sum statistic
                    'vTbhat_s4':(v**4).dot(bhat**4), # used for variance of product stat
                    'vTbhat_vars':var_s(v,bhat), # variance of sum statistic
                    'vTbhat_varss':var_ss(v,bhat), # variance of product statistic
                    'suppVo':(v!=0).sum(),
                    'normahat':ahat.dot(ahat),
                    'normbhat':bhat.dot(bhat),
                    'M_typed':t.sum(),
                    'normVo_inf':maxv,
                    'normVo_4':np.linalg.norm(v, ord=4),
                    'normVo_2':np.linalg.norm(v, ord=2),
                    'normVo_1':np.linalg.norm(v, ord=1),
                    'normVo_info1':maxv/np.linalg.norm(v, ord=1),
                    'normVo_2o4':(np.linalg.norm(v, ord=2)**2)/(np.linalg.norm(v, ord=4)**2)},
                    ignore_index=True)
            del V; del t; del ahat; del bhat; del N; del locussnps
            gc.collect()
        del snps; memo.reset(); gc.collect()

    print('writing locus-by-locus results')
    locusresults.to_csv(args.outfile_stem + '.loci', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/UKBB_body_HEIGHTz.KG3_0.1/',
            help='one or more paths to .bhat.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/UwGm12878Ctcf/prod0.lfc.',
                '/groups/price/yakir/data/annot/basset/BroadDnd41Ctcf/prod0.lfc.'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('-no-finemap', default=False, action='store_true',
            help='dont use the finemapped sumstats, use the normal sumstats')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--loci',
            default='/groups/price/yakir/data/reference/dixon_IMR90.TADs.hg19.bed',
            help='path to UCSC bed file containing one bed interval per locus')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to estimate heritability')
    parser.add_argument('--nrows', default=None, type=int)
    parser.add_argument('--chroms', nargs='+', default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
