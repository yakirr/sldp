from __future__ import print_function, division
import argparse
import os, gzip
import numpy as np
import pandas as pd
import itertools as it
from pybedtools import BedTool
import pyutils.pretty as pretty
import pyutils.fs as fs
import pyutils.iter as pyit
import gprim.annotation as ga
import gprim.dataset as gd
import pyutils.memo as memo
import gc
import time


def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    chunksize = 25
    refpanel = gd.Dataset(args.bfile_chr)
    annot = ga.Annotation(args.sannot_chr)
    outstem = args.sannot_chr + args.refpanel_name + '_' + str(args.NLambda)
    Rpath = outstem + '.R.'
    Rlipath = outstem + '.Rli.'
    RRlipath = outstem + '.RRli.'
    print('Will store output in', Rpath, 'and', Rlipath)

    # read in ld blocks, leaving in MHC
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]
    print(len(ldblocks), 'loci after removing MHC')

    t0 = time.time()
    log = pd.DataFrame()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)

        # read in annotation
        print('reading annot', annot.filestem())
        names = annot.names(c) # names of annotations
        namesRli = [n+'.Rli' for n in names]
        namesR = [n+'.R' for n in names]
        namesRRli = [n+'.RRli' for n in names]
        maf = refpanel.frq_df(c).MAF.values
        a = annot.sannot_df(c)
        V = a[names].values

        # put on per-normalized-genotype scale
        if not args.per_norm_genotype:
            print('adjusting for maf')
            V = V*np.sqrt(2*maf*(1-maf))[:,None]

        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')
        if any(a.SNP != snps.SNP):
            print('ERROR: refpanel and annot need to have same SNPs in same order')
            exit()

        # reading in typed snps
        typed = np.isfinite(pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t').ahat)
        # typed = np.ones(len(snps)).astype(bool)

        # merge annot and refpanel
        print('merging')
        snps = pd.concat([snps, pd.DataFrame(V, columns=names)], axis=1)
        snps = pd.concat([snps, pd.DataFrame(V, columns=namesR)], axis=1)
        snps = pd.concat([snps, pd.DataFrame(V, columns=namesRli)], axis=1)
        snps = pd.concat([snps, pd.DataFrame(V, columns=namesRRli)], axis=1)
        snps['typed'] = typed

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
            # process blocks one at a time
            for i, start_ind, end_ind in zip(
                    chunk_blocks.index, blockstarts_ind, blockends_ind):
                print(time.time()-t0, ': processing ld block',
                        i, ',', end_ind-start_ind, 'snps')
                X = Xchunk[:, start_ind:end_ind]
                snpsblock = snpschunk.iloc[start_ind:end_ind]

                ### processing is here ###
                V = snpsblock[names].values
                Nr = X.shape[0]

                def mult_Rli(A, X_t):
                    lam = args.NLambda / Nr
                    B = A / lam - \
                        1/lam**2 * X_t.T.dot(
                            np.linalg.solve(
                                Nr*np.eye(Nr) + X_t.dot(X_t.T)/lam,
                                X_t.dot(A)))
                    return B * (1+lam)
                def mult_R(A, X, X_t):
                    return X.T.dot(X_t.dot(A)) / Nr

                monomorphic = (np.var(X, axis=0) == 0)
                use = ~monomorphic

                # compute RV
                print('\t','computing RV')
                print('\t',use.sum(), 'snps being used')
                X_u = X[:,use]; V_u = V[use]
                RV = mult_R(V_u, X_u, X_u)
                snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],namesR] = RV

                # compute RliV
                print('\t','computing RliV')
                use = use & snpsblock.typed.values
                print('\t',use.sum(), 'snps being used')
                X_u = X[:,use]; V_u = V[use]
                RliV = mult_Rli(V_u, X_u)
                snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],namesRli] = RliV

                print('\t','computing RRliV')
                RRliV = mult_R(RliV, X_u, X_u)
                snps.loc[snpschunk.iloc[start_ind:end_ind].index[use],namesRRli] = RRliV

                v = V[:,0][use]
                Rv = RV[:,0][use]
                Rliv = RliV[:,0]
                RRliv = RRliV[:,0]
                vTv = v.dot(v); vTRliRv = Rliv.dot(Rv)
                vTRliv = v.dot(Rliv); vTRliRRliv = RRliv.dot(Rliv)

                log = log.append({
                    'blocknum':i,
                    'vTv':vTv, 'vTRliRv':vTRliRv,
                    'vTRliv':vTRliv, 'vTRliRRliv':vTRliRRliv,
                    'vTv/vTRliRv':v.dot(v)/Rliv.dot(Rv),
                    'vTRliv/vTRliRRliv':v.dot(Rliv)/RRliv.dot(Rliv)},
                    ignore_index=True)
                print(log.iloc[-1])

                del X; del V; del snpsblock
                del X_u; del RliV; del RV;
            del Xchunk; del snpschunk; gc.collect()

        print('writing output')
        with gzip.open('{}{}.sannot.gz'.format(Rpath, c),'w') as f:
            snps[['CHR','BP','SNP','CM','A1','A2']+namesR].to_csv(
                    f, index=False, sep='\t', na_rep='nan')
        with gzip.open('{}{}.sannot.gz'.format(Rlipath, c),'w') as f:
            snps[['CHR','BP','SNP','CM','A1','A2']+namesRli].to_csv(
                    f, index=False, sep='\t', na_rep='nan')
        with gzip.open('{}{}.sannot.gz'.format(RRlipath, c),'w') as f:
            snps[['CHR','BP','SNP','CM','A1','A2']+namesRRli].to_csv(
                    f, index=False, sep='\t', na_rep='nan')
        with open('{}{}.log'.format(args.sannot_chr, c),'w') as f:
            log.to_csv(f, index=False, sep='\t')

        del snps; memo.reset(); gc.collect()

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sannot-chr', #required=True,
            default='/groups/price/yakir/data/annot/baseline/functional/',
            help='path to sannot.gz files, not including chromosome')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/CD.KG3_50.0/',
            help='the snps to consider as typed when inverting the LD matrix')
    parser.add_argument('--NLambda', type=int, default=int(500*0.1))

    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--refpanel-name', default='KG3',
            help='suffix added to the directory created for storing output')
    parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int)

    args = parser.parse_args()
    main(args)
