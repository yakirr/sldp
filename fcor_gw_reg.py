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
import resource

import statutils.vis as vis


def mem():
    print('memory usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, 'Mb')

def get_locus_inds(snps, loci, chroms):
    loci.sort(columns='CHR', inplace=True)
    locusstarts = np.array([])
    locusends = np.array([])
    for c in chroms:
        chrsnps = snps[snps.CHR == c]
        mask = (loci.CHR == 'chr'+str(c))
        chrloci = loci[mask]
        offset = np.where(snps.CHR.values == c)[0][0]
        newlocusstarts = offset + np.searchsorted(chrsnps.BP.values, chrloci.start.values)
        newlocusends = offset + np.searchsorted(chrsnps.BP.values, chrloci.end.values)
        # locusstarts = np.append(locusstarts,
        #         newlocusstarts)
        # locusends = np.append(locusends,
        #         newlocusends)
        loci.loc[mask,'start_ind'] = newlocusstarts.astype(int)
        loci.loc[mask,'end_ind'] = newlocusends.astype(int)
    return zip(loci.start_ind.values, loci.end_ind.values)


def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    mhcmask = None
    locus_inds = None
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.')[0]
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    results = pd.DataFrame()

    # read in loci and remove ones that overlap mhc
    loci = pd.read_csv(args.loci, delim_whitespace=True, header=None,
            names=['CHR','start', 'end'])
    mhcblocks = (loci.CHR == 'chr6') & (loci.end > mhc[0]) & (loci.start < mhc[1])
    loci = loci[~mhcblocks]
    print(len(loci), 'loci after removing MHC')

    mem()

    print('reading maf')
    maf = np.concatenate([refpanel.frq_df(c).MAF.values for c in args.chroms])
    memo.reset(); gc.collect(); mem()

    print('reading sumstats, specifically:', args.use)
    ss = np.concatenate([
        pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t',
            usecols=[args.use])[args.use].values
        for c in args.chroms])
    print('getting typed snps')
    typed = np.isfinite(ss)
    print('restricting to typed snps, of which there are', typed.sum())
    bhat = ss[typed]

    maft = maf[typed]
    mem()

    t0 = time.time()
    results = pd.DataFrame()
    for annot in annots:
        names = annot.names(22) # names of annotations
        print(time.time()-t0, ': reading annot', annot.filestem())
        a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)

        print('restricting to typed snps only')
        a = a[typed]
        memo.reset(); gc.collect(); mem()

        if mhcmask is None:
            print('creating mhcmask')
            CHR = a.CHR.values
            BP = a.BP.values
            mhcmask = (CHR == 6)&(BP >= mhc[0])&(BP <= mhc[1])
            print('getting locus indices')
            locus_inds = get_locus_inds(a, loci, args.chroms)
            maft = maft[~mhcmask]

        print('creating V')
        # TODO: this line should be a warning instead, because if V has nan's we need
        # to regenerate it
        print('there are', (~np.isfinite(a[names].values)).sum(), 'nans in the annotation')
        V = a[names].fillna(0).values

        print('throwing out mhc')
        V = V[~mhcmask,:]
        bhat = bhat[~mhcmask]

        if not args.per_norm_genotype:
            print('adjusting for maf')
            V = V*np.sqrt(2*maft*(1-maft))[:,None]

        Vjoint = np.linalg.solve(V.T.dot(V), V.T).T

        print(V.T.dot(V))


        if args.dump is not None:
            import pickle
            ss = np.concatenate([
                pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t',
                    usecols=['ahat','bhat'])[['ahat','bhat']].values
                for c in args.chroms])
            ss = ss[typed]
            ss = ss[~mhcmask]
            pickle.dump({'V':V,'bhat':ss[:,1],'names':names,'locus_inds':locus_inds,'loci':loci,
                'ahat':ss[:,0], 'maf':maf,'chr':a.CHR.values,'bp':a.BP.values},
                open(args.dump,'w'), 2)
        del a; gc.collect(); mem()

        for i, name in enumerate(names):
            print(i, name)
            v = V[:,i]
            vjoint = Vjoint[:,i]

            q = v*bhat
            qloci = np.array([
                q[start:end].sum()
                for start, end in locus_inds])
            qjoint = vjoint*bhat
            qlocijoint = np.array([
                qjoint[start:end].sum()
                for start, end in locus_inds])
            print(name, 'marginal coeff:', v.dot(bhat)/v.dot(v),
                    'marginal z:', q.sum()/np.linalg.norm(q),
                    'joint coeff:', vjoint.dot(bhat),
                    'joint z:', qjoint.sum()/np.linalg.norm(qjoint))
            results = results.append({
                'annot':name,
                'pheno':nice_ss_name,
                'use':args.use,
                'marginal_corr':v.dot(bhat)**2 / (v.dot(v)*bhat.dot(bhat)),
                'marginal':v.dot(bhat)/v.dot(v),
                'marginal_z':q.sum()/np.linalg.norm(q),
                'marginal_loci_z':qloci.sum()/np.linalg.norm(qloci),
                'joint':vjoint.dot(bhat),
                'joint_z':qjoint.sum()/np.linalg.norm(qjoint),
                'joint_loci_z':qlocijoint.sum()/np.linalg.norm(qlocijoint)},
                ignore_index=True)

        del V; memo.reset(); gc.collect(); mem()

    print('writing results')
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/CD.KG3_0.1/',
            help='one or more paths to .bhat.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/HaibK562MaxV0416102/prod0.lfc.',
                '/groups/price/yakir/data/annot/basset/HaibK562Atf3V0416101/prod0.lfc.'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--use', default='bhat',
            help='which column from the processed sumstats file.')
    parser.add_argument('-clumpv', default=False, action='store_true',
            help='clump togther close-by snps in the null')
    parser.add_argument('-center-on-v', default=False, action='store_true',
            help='mean-center based on magnitude of v rather than maf')
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only. this is used to estimate heritability')
    # parser.add_argument('--loci',
    #         default='/groups/price/yakir/data/reference/dixon_IMR90.TADs.hg19.bed',
    #         help='path to UCSC bed file containing one bed interval per locus')
    parser.add_argument('--loci',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per locus')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))
    parser.add_argument('--dump', default=None)

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
