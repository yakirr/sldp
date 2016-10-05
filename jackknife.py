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


# import re
# def str_to_np(x):
#     if not type(x)==str:
#         return x
#     result = re.sub(r'[0-9.]\s+[0-9.]', ',', x)
#     result = re.sub(r'\]\s+\[', ',', result)
#     result = result.replace('nan', 'float(\'nan\')')
#     return np.array(eval(result))

# def read_results(outfile):
#     ldblocks = pd.read_csv(outfile, sep='\t')
#     ldblocks.vTahat = ldblocks.vTahat.apply(str_to_np)
#     ldblocks.voTRv = ldblocks.voTRv.apply(str_to_np)
#     ldblocks.voTRvo = ldblocks.voTRvo.apply(str_to_np)
#     ldblocks.voTRvo_N = ldblocks.voTRvo_N.apply(str_to_np)
#     ldblocks.voTR2vo = ldblocks.voTR2vo.apply(str_to_np)
#     return ldblocks
def process_blockresults(blockresults):
    def stat(df):
        z = df.vTahat.values / df.voTRv.values
        mask = np.isfinite(z)
        return df[mask].vTahat.values.dot(df[mask].weights.values) / df[mask].weights.sum()

    def jackknife(df, s):
        loo = np.zeros(len(df))
        B = len(loo)
        for i in range(B):
            loo[i] = s(df[np.arange(B)!=i])
        return s(df), np.sqrt((B-1)*np.var(loo))

    # set weights
    blockresults['weights'] = blockresults.voTRv.values

    results = pd.DataFrame(columns=['pheno','annot','stat','std','p'])
    annots = np.unique(blockresults.annot.values)
    phenos = np.unique(blockresults.pheno.values)
    for p, a in it.product(phenos, annots):
        myresults = blockresults[(blockresults.annot == a)&(blockresults.pheno == p)]
        val, std = jackknife(myresults, stat)
        results = results.append({
            'pheno':p,
            'annot':a,
            'stat':val,
            'std':std,
            'p':st.chi2.sf((val/std)**2, 1)}, ignore_index=True)
    return results

# def multiply(ldblocks='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
#         bfile_chr='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/1000G.EUR.QC.',
#         annot_chr=['/groups/price/yakir/data/annot/basset/UwGm12878Ctcf/prod0.lfc.',
#             '/groups/price/yakir/data/annot/basset/BroadDnd41Ctcf/prod0.lfc.'],
#         ldscores_chr='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/weights.hm3_noMHC.',
#         per_norm_genotype=False,
#         sumstats='/groups/price/yakir/data/sumstats/UKBB_body_HEIGHTz.sumstats.gz',
#         outfile_stem='/groups/price/yakir/temp'):
def multiply(args):
    # basic initialization
    chroms = range(1,23)
    nrows=None
    chunksize = 25
    refpanel = gd.Dataset(args.bfile_chr)
    annots = [ga.Annotation(annot) for annot in args.annot_chr]
    names = np.concatenate([annot.names(22) for annot in annots]) # names of annotations
    namesO = [n + '.O' for n in names] # names of annotations zeroed out at untyped snps
    nice_ss_name = sumstats.split('/')[-1].split('.sumstats.gz')[0]
    blockresults = pd.DataFrame(columns=['pheno','annot','block_num','chr','start','end'])

    # read in ld blocks and remove ones that overlap mhc
    mhc = [25684587, 35455756]
    ldblocks = pd.read_csv(args.ldblocks, delim_whitespace=True, header=False,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]

    # read sumstats, remove outlier snps
    print('reading sumstats', nice_ss_name)
    ss = pd.read_csv(args.sumstats, sep='\t', nrows=nrows)
    ss['ahat'] = ss.Z / np.sqrt(ss.N)
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    ss = ss[ss.ahat**2 <= 8e-4] #TODO: think about whether to remove outliers
    print(len(ss), 'snps after removing outliers')

    # estimate heritability using aggregate estimator
    print('reading in ld scores')
    l2 = pd.concat([pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in chroms], axis=0)
    print(len(l2), 'snps')
    ssl2 = pd.merge(ss, l2, on='SNP', how='inner')
    print(len(ssl2), 'snps after merge')
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
    for c in chroms:
        print(time.time()-t0, ': loading annot for chr', c)
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
        snps = ga.reconciled_to(snps, ss, ['ahat'], othercolnames=['N'])
        snps['typed'] = snps.N.notnull()
        snps[namesO] = snps[names]; snps.loc[~snps.typed, namesO] = 0

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
                X = Xchunk[:, start_ind:end_ind]
                snpsblock = snpschunk.iloc[start_ind:end_ind]
                V = snpsblock[names].values
                Vo = snpsblock[namesO].values
                ahat = snpsblock.ahat.values
                N = np.nanmin(snpsblock.N.values) # TODO: fix this for variable N!

                # compute RVo
                mask = np.unique(np.where(Vo)[0])
                RVo = X.T.dot(X[:,mask].dot(Vo[mask])) / X.shape[0]

                # store blockresults
                VTahat = V.T.dot(ahat)
                VoTRV = RVo.T.dot(V)
                VoTRVo = Vo.T.dot(RVo)
                VoTRVo_N = Vo.T.dot(RVo)/N
                VoTR2Vo = RVo.T.dot(RVo) / (1+1/X.shape[0])
                for j,n in enumerate(names):
                    blockresults = blockresults.append({
                        'pheno':nice_ss_name,
                        'annot':n,
                        'block_num':i,
                        'chr':ldblocks.loc[i,'chr'],
                        'start':ldblocks.loc[i,'start'],
                        'end':ldblocks.loc[i,'end'],
                        'vTahat':VTahat[j],
                        'voTRv':VoTRV[j,j],
                        'voTRvo':VoTRVo[j,j],
                        'voTRvo_N':VoTRVo_N[j,j],
                        'voTR2vo':VoTR2Vo[j,j],
                        'sigma2g':sigma2g}, ignore_index=True)
                del X; del V; del Vo; del ahat; del N; del snpsblock
            del Xchunk; del snpschunk; gc.collect()
        memo.reset(); gc.collect()

    print('writing block-by-block results')
    blockresults.to_csv(args.outfile_stem + '.blocks', sep='\t', index=False)

    results = process_blockresults(blockresults)
    results.to_csv(args.outfile_stem + '.results', sep='\t', index=False)
    print('done')


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
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--mhc-path',
            default='/groups/price/yakir/data/reference/hg19.MHC.bed',
            help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                    ' breakpoint')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    submitmerge_parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to estimate heritability')

    args = parser.parse_args()
    main(args)


