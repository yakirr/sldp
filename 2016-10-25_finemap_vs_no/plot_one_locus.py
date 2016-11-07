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
import matplotlib.pyplot as plt
import statutils.vis as vis


def init(sumstats):
    nrows=None
    nice_ss_name = sumstats.split('/')[-1].split('.sumstats.gz')[0]
    # read sumstats, remove outlier snps
    print('reading sumstats', nice_ss_name)
    ss = pd.read_csv(sumstats, sep='\t', nrows=nrows)
    ss['ahat'] = ss.Z / np.sqrt(ss.N)
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    # ss = ss[ss.ahat**2 <= 8e-4] #TODO: think about whether to remove outliers
    print(len(ss), 'snps after removing outliers')
    return ss

def main(ss, sannot_chr, c, block_num,
        bfile_chr='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/1000G.EUR.QC.',
        ld_blocks='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
        per_norm_genotype=False,
        Lambda=0.1):
    # basic initialization
    refpanel = gd.Dataset(bfile_chr)
    annots = [ga.Annotation(annot) for annot in sannot_chr]
    names = np.concatenate([annot.names(17) for annot in annots]) # names of annotations
    namesO = [n + '.O' for n in names] # names of annotations zeroed out at untyped snps

    # read in ld blocks and remove ones that overlap mhc
    mhc = [25684587, 35455756]
    ldblocks = pd.read_csv(ld_blocks, delim_whitespace=True, header=False,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]

    t0 = time.time()
    print(time.time()-t0, ': loading annot for chr', c)
    # read in all annotations for this chr. we assume that all annots have same snps with
    # same coding as each other and as refpanel, in same order
    snps = ga.smart_merge([annot.sannot_df(c) for annot in annots],
            fail_if_nonmatching=True, drop_from_y=['BP','CM','A1','A2','CHR'])
    print(len(snps), 'snps in annot', len(snps.columns), 'columns, including metadata')
    # adjust for maf
    if not per_norm_genotype:
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
    chunk_blocks = ldblocks.loc[block_num:block_num]
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
        print('processing ld block', i, ',', end_ind-start_ind, 'snps')
        X = Xchunk[:, start_ind:end_ind]
        snpsblock = snpschunk.iloc[start_ind:end_ind]
        V = snpsblock[names].values
        Vo = snpsblock[namesO].values
        ahat = snpsblock.ahat.values
        N = np.nanmin(snpsblock.N.values) # TODO: fix this for variable N!
        Nr = X.shape[0]

        # # compute Vof (fine-mapped V)
        t = snpsblock.typed.values
        # X_t = X[:,t]; Vo_t = Vo[t]
        # Vf = np.zeros(V.shape)
        # Vf[t] = Vo_t / Lambda - \
        #     1/Lambda**2 * X_t.T.dot(
        #         np.linalg.solve(
        #             Nr*np.eye(Nr) + X_t.dot(X_t.T)/Lambda,
        #             X_t.dot(Vo_t)))
        # Vf = Vf * (1+Lambda)

        # compute betahat
        X_t = X[:,t]; ahat_t = ahat[t]
        betahat = np.zeros(ahat.shape)
        betahat[t] = ahat_t / Lambda - \
            1/Lambda**2 * X_t.T.dot(
                np.linalg.solve(
                    Nr*np.eye(Nr) + X_t.dot(X_t.T)/Lambda,
                    X_t.dot(ahat_t)))
        betahat = betahat * (1+Lambda)

        return Vo[t].flatten(), betahat[t], ahat[t], X_t

        plt.scatter(Vo[t].flatten, betahat[t]); plt.show()

        # compute RVo and RVof
        small_ref_l = 0.1
        print('small_ref_l = ', small_ref_l)
        mask = np.unique(np.where(Vo)[0])
        RVo = X.T.dot(X[:,mask].dot(Vo[mask])) / Nr
        RVo = RVo/(1+small_ref_l) + small_ref_l/(1+small_ref_l)*Vo
        RVf = X.T.dot(X.dot(Vf)) / Nr
        RVf = RVf/(1+small_ref_l) + small_ref_l/(1+small_ref_l)*Vf
        import pdb; pdb.set_trace()

        # store blockresults
        VTahat = V.T.dot(ahat)
        VfTahat = Vf.T.dot(ahat)
        VoTRV = RVo.T.dot(V)
        VfTRV = RVf.T.dot(V)
        VoTRVo = Vo.T.dot(RVo)
        VfTRVf = Vf.T.dot(RVf)
        VoTRVo_N = Vo.T.dot(RVo)/N
        VfTRVf_N = Vf.T.dot(RVf)/N
        VoTR2Vo = RVo.T.dot(RVo) / (1+1/Nr)
        VfTR2Vf = RVf.T.dot(RVf) / (1+1/Nr)
        for j,n in enumerate(names):
            blockresults = blockresults.append({
                'pheno':nice_ss_name,
                'annot':n,
                'block_num':i,
                'chr':ldblocks.loc[i,'chr'],
                'start':ldblocks.loc[i,'start'],
                'end':ldblocks.loc[i,'end'],
                'Lambda':args.Lambda,
                'vTahat':VTahat[j],
                'vfTahat':VfTahat[j],
                'voTRv':VoTRV[j,j],
                'vfTRv':VfTRV[j,j],
                'voTRvo':VoTRVo[j,j],
                'vfTRvf':VfTRVf[j,j],
                'voTRvo_N':VoTRVo_N[j,j],
                'vfTRvf_N':VfTRVf_N[j,j],
                'voTR2vo':VoTR2Vo[j,j],
                'vfTR2vf':VfTR2Vf[j,j],
                'suppVo':(Vo[:,j]!=0).sum(),
                'normVo':Vo[:,j].dot(Vo[:,j]),
                'normVf':Vf[:,j].dot(Vf[:,j]),
                'maxVo2':np.nanmax(Vo[:,j]**2),
                'maxVf2':np.nanmax(Vf[:,j]**2),
                'sigma2g':sigma2g}, ignore_index=True)
        del X; del V; del Vo; del ahat; del N; del snpsblock
    del Xchunk; del snpschunk; gc.collect()
    del snps; memo.reset(); gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--sumstats', required=True,
            default='/groups/price/yakir/data/sumstats/UKBB_body_HEIGHTz.sumstats.gz',
            help='path to sumstats.gz files, including extension')
    parser.add_argument('--sannot-chr', nargs='+', required=True,
            default=['/groups/price/yakir/data/annot/basset/UwGm12878Ctcf/prod0.lfc.',
                '/groups/price/yakir/data/annot/basset/BroadDnd41Ctcf/prod0.lfc.'],
            help='path to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--Lambda', type=float, default=0.1)
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to estimate heritability')
    parser.add_argument('--chroms', nargs='+', default=range(1,23))

    args = parser.parse_args()
    main(args)
