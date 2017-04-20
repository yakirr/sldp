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
    chunksize = 25
    refpanel = gd.Dataset(args.bfile_chr)

    # read in ld blocks, leaving in MHC
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])

    # read sumstats
    print('reading sumstats', args.sumstats_stem)
    ss = pd.read_csv(args.sumstats_stem+'.sumstats.gz', sep='\t', nrows=args.nrows)
    ss = ss[ss.Z.notnull() & ss.N.notnull()]
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    print(len(ss), 'snps after removing outliers')

    # read ld scores
    print('reading in ld scores')
    ld = pd.concat([pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in range(1,23)], axis=0)
    M_5_50 = sum([int(open(args.ldscores_chr+str(c)+'.l2.M_5_50').next())
                    for c in range(1,23)])
    wld = pd.concat([pd.read_csv(args.ldscores_weights_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in range(1,23)], axis=0).rename(columns={'L2':'wL2'})[['SNP','wL2']]
    ld = pd.merge(ld, wld, on='SNP', how='inner')
    print(len(ld), 'snps')
    ssld = pd.merge(ss, ld, on='SNP', how='left')
    ssld['hm3'] = ssld.L2.notnull()
    print(ssld.hm3.sum(), 'hm3 snps with sumstats after merge')
    mhc = [25684587, 35455756]
    ssld = ssld[~((ssld.CHR == 6) & (ssld.BP >= mhc[0]) & (ssld.BP < mhc[1]))]
    print(ssld.hm3.sum(), 'snps after removing mhc')

    # estimate heritability using aggregate estimator
    totalchi2 = (ssld[ssld.hm3].Z**2).sum()
    Mo = ssld.hm3.sum()
    suml2 = (ssld[ssld.hm3].N*ssld[ssld.hm3].L2).sum()
    h2g = (totalchi2 - Mo)/(suml2/M_5_50)
    sigma2g = h2g / M_5_50
    print('h2g estimated at:', h2g, 'sigma2g =', sigma2g)

    # write results to file
    dirname = args.sumstats_stem + '.' + args.refpanel_name + '_' + str(args.NLambda)
    fs.makedir(dirname)
    if 1 in args.chroms:
        print('writing info file')
        info = pd.DataFrame(); info=info.append(
                {'pheno':args.sumstats_stem.split('/')[-1],'h2g':h2g,
                    'Nbar':ss.N.mean()},ignore_index=True)
        info.to_csv(dirname+'/info', sep='\t', index=False)

    t0 = time.time()
    log = pd.DataFrame()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # merge annot and sumstats, and create Vo := V zeroed out at unobserved snps
        print('reconciling')
        snps = ga.reconciled_to(snps, ssld, ['Z'],
                othercolnames=['N','hm3','L2','wL2'], missing_val=np.nan)
        snps['typed'] = snps.Z.notnull()
        snps['ahat'] = snps.Z / np.sqrt(snps.N)
        snps['weights'] = 1./(snps.wL2 * (1./snps.N + sigma2g * snps.L2))
        snps['bhat'] = np.nan # bhat = Ro^{-1} ahat
        snps['dhat'] = np.nan # dhat = R{*o} Ro^{-1} Ro^{-1} ahat
        snps['ahat_hm3'] = np.nan
        snps['bhat_hm3'] = np.nan # bhat = Ro^{-1} ahat, o=HM3
        snps['ghat_hm3'] = np.nan # ghat = R{*o} ahat, o=HM3
        snps['dhat_hm3'] = np.nan # dhat = R{*o} Ro^{-1} Ro^{-1} ahat, o=HM3

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
                w = snpsblock.weights.values
                t = snpsblock.typed.values
                h = snpsblock.hm3.fillna(False).astype(bool).values
                Nr = X.shape[0]

                def mult_Rinv(A, X_t):
                    lam = args.NLambda / Nr
                    B = A / lam - \
                        1/lam**2 * X_t.T.dot(
                            np.linalg.solve(
                                Nr*np.eye(Nr) + X_t.dot(X_t.T)/lam,
                                X_t.dot(A)))
                    return B * (1+lam)
                def mult_Rinv_SVD(A, X_t, R=None):
                    if R is None:
                        R = X_t.T.dot(X_t) / Nr
                    U, svs, VT = np.linalg.svd(R)
                    k = np.argmax(np.cumsum(svs)/svs.sum() >= 0.99)
                    print('using', k-1, 'of', len(svs), 'singular values')
                    svsinv = 1./svs
                    svsinv[k:] = 0
                    return U.dot(svsinv*VT.dot(A)), R
                def mult_R(A, X, X_t):
                    return X.T.dot(X_t.dot(A)) / Nr

                monomorphic = (np.var(X, axis=0) == 0)
                use = t&(~monomorphic)

                #### all observed snps
                if use.sum() > 0:
                    print('\t',use.sum(), 'observed snps')
                    # compute bhat
                    print('\t','computing bhat')
                    X_u = X[:,use]; ahat_u = ahat[use]
                    bhat = mult_Rinv(ahat_u, X_u)
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],'bhat'] = bhat

                    # compute dhat
                    print('\t','computing dhat')
                    dhat = mult_Rinv(bhat, X_u)
                    dhat = mult_R(dhat, X, X_u)
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[:],'dhat'] = dhat

                ### hm3 snps
                use = use&h
                if use.sum() > 0:
                    print('\t',use.sum(), 'observed hm3 snps')
                    # computa ahat
                    print('\t','computing ahat_hm3')
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],'ahat_hm3'] = ahat[use]

                    # compute bhat
                    print('\t','computing bhat_hm3')
                    X_u = X[:,use]; ahat_u = ahat[use]
                    bhat_hm3, R = mult_Rinv_SVD(ahat_u, X_u)
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],'bhat_hm3'] = bhat_hm3

                    # compute dhat
                    print('\t','computing dhat_hm3')
                    dhat_hm3, _ = mult_Rinv_SVD(bhat_hm3, X_u, R=R)
                    dhat_hm3 = mult_R(dhat_hm3, X, X_u)
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[:],'dhat_hm3'] = dhat_hm3

                    # compute ghat
                    print('\t','computing ghat_hm3')
                    w_u = w[use]
                    ghat_hm3 = mult_R(w_u*ahat_u, X, X_u)
                    snps.loc[snpschunk.iloc[start_ind:end_ind].index[:],'ghat_hm3'] = ghat_hm3

                    # do qc
                    print('\t','**testing bhat_hm3, ghat_hm3, and ahat_u')
                    print(bhat_hm3.dot(ghat_hm3[use]), 'should be close to', ahat_u.dot(w_u*ahat_u))

                    print('\t','**testing bhat_hm3')
                    temp = mult_R(bhat_hm3, X_u, X_u)
                    print(temp[:10])
                    print('\t','should be similar to')
                    print(ahat_u[:10])
                    print('\t','their correlation is', np.corrcoef(temp, ahat_u)[0,1])

                    print('\t','**testing dhat_hm3 and bhat_hm3')
                    print(dhat_hm3[use][:10])
                    print('\t','should be similar to')
                    print(bhat_hm3[:10])
                    print('\t','their correlation is', np.corrcoef(dhat_hm3[use],bhat_hm3)[0,1])

                    log = log.append({
                        'CHR':c,
                        'blocknum':i,
                        'start':ldblocks.iloc[i].start,
                        'end':ldblocks.iloc[i].end,
                        'q1':bhat_hm3.dot(ghat_hm3[use]),
                        'q2':ahat_u.dot(w_u*ahat_u),
                        'corr1':np.corrcoef(temp, ahat_u)[0,1],
                        'corr2':np.corrcoef(dhat_hm3[use],bhat_hm3)[0,1]},
                        ignore_index=True)

                    del X; del ahat; del t; del h; del snpsblock
                    del X_u; del bhat; del dhat; del bhat_hm3; del dhat_hm3; del ghat_hm3
            del Xchunk; del snpschunk; gc.collect()

        print('writing finemapped sumstats')
        with gzip.open('{}/{}.bhat.gz'.format(dirname, c), 'w') as f:
            snps[['SNP','A1','A2','N','ahat','bhat','dhat',
                'ahat_hm3','bhat_hm3','ghat_hm3','dhat_hm3']].to_csv(
                    f, index=False, sep='\t', na_rep='nan')

        print('writing log file')
        log[['blocknum','CHR','start','end','q1','q2','corr1','corr2']].to_csv(
                '{}/{}.log'.format(dirname, c), index=False, sep='\t')

        del snps; memo.reset(); gc.collect()

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumstats-stem', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/UKBB_body_HEIGHTz',
            help='path to sumstats.gz files, not including ".sumstats.gz" extension')
    parser.add_argument('--NLambda', type=float, default=500*0.1)

    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/LDscore/LDscore.',
            help='path to LD scores at a smallish set of SNPs. LD should be computed '+\
                    'to all potentially causal snps. This is used for weighting and for '+\
                    'heritability estimation')
    parser.add_argument('--ldscores-weights-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to weight the ld-ified summary '+\
                    'statistics.')
    parser.add_argument('--refpanel-name', default='KG3',
            help='suffix added to the directory created for storing output')
    parser.add_argument('--nrows', default=None, type=int)
    parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int)

    args = parser.parse_args()
    main(args)
