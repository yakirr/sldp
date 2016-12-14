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


def mem():
    print('memory usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, 'Mb')

def get_p_empirical(q):
    print(len(q))
    p = 1; p_std = 100; num_per = int(20000000 / len(q))
    stat = q.sum()
    null = np.array([])
    while p_std > 0.3*p and p > 1e-6:
        rademachers = np.random.binomial(1,0.5,(num_per, len(q)))
        rademachers[rademachers == 0] = -1
        null = np.concatenate([null, rademachers.dot(q)])
        p = max((np.abs(null) > np.abs(stat)).sum(), 1) / len(null)
        p_std = np.sqrt(p*(1-p)/len(null))
        print('\t',len(null), p, p_std)
    return p

def get_locus_inds(snps, loci, chroms):
    loci.sort(columns='CHR', inplace=True)
    locusstarts = np.array([])
    locusends = np.array([])
    for c in chroms:
        chrsnps = snps[snps.CHR == c]
        chrloci = loci[loci.CHR == 'chr'+str(c)]
        offset = np.where(snps.CHR.values == c)[0][0]
        locusstarts = np.append(locusstarts,
                offset + np.searchsorted(chrsnps.BP.values, chrloci.start.values))
        locusends = np.append(locusends,
                offset + np.searchsorted(chrsnps.BP.values, chrloci.end.values))
    return zip(locusstarts.astype(int), locusends.astype(int))


def main(args):
    # basic initialization
    mhc = [25684587, 35455756]
    mhcmask = None
    locus_inds = None
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.KG3')[0]
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

    col = ('ahat' if args.no_finemap else 'bhat')
    print('reading sumstats, specifically:', col)
    ss = np.concatenate([
        pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t', usecols=[col])[col].values
        for c in args.chroms])
    print('getting typed snps')
    typed = np.isfinite(ss)
    print('restricting to typed snps')
    bhat = ss[typed]

    mem()

    t0 = time.time()
    for annot in annots:
        names = annot.names(22) # names of annotations
        print(time.time()-t0, ': reading annot', annot.filestem())
        a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)

        print('restricting to typed snps only')
        a = a[typed]
        maft = maf[typed]
        memo.reset(); gc.collect(); mem()

        if mhcmask is None:
            print('creating mhcmask')
            CHR = a.CHR.values
            BP = a.BP.values
            mhcmask = (CHR == 6)&(BP >= mhc[0])&(BP <= mhc[1])
            print('getting locus indices')
            locus_inds = get_locus_inds(a, loci, args.chroms)

        print('creating V')
        V = a[names].values
        del a; gc.collect(); mem()

        print('zeroing out mhc')
        V[mhcmask, :] = 0

        if not args.per_norm_genotype:
            print('adjusting for maf')
            V = V*np.sqrt(2*maft*(1-maft))[:,None]

        for i, name in enumerate(names):
            print(i, name)
            v = V[:,i]
            nz = (v!=0)

            if nz.sum() == 0:
                continue

            # center bhat (this code can do it in maf bins if necessary)
            print('centering betahat and v')
            bhatc = bhat.copy()
            vc = v.copy()
            cvec = (-np.abs(v) if args.center_on_v else maft)
            cutoffs = np.percentile(cvec[nz], [0,5,10,20,50,100])
            for a, b in zip(cutoffs[:-1],cutoffs[1:]):
                print('\t',a,b,((cvec>=a)&(cvec<b)&nz).sum())
                bhatc[(cvec >= a)&(cvec<b)&nz] -= bhat[(cvec >=a)&(cvec<b)&nz].mean()
                vc[(cvec >= a)&(cvec<b)&nz] -= vc[(cvec >=a)&(cvec<b)&nz].mean()

            # compute q
            qall = v*bhat
            qcall = vc*bhatc
            qpall = np.abs(v)*bhat
            qpcall = np.abs(vc)*bhat

            #go through and for (q1,q2,...) that should be merged, replace them
            # with (q1+q2+...,0)
            if args.clumpv:
                closetonext = (BP[1:] - BP[:-1] <= 150)
                for i in np.where(closetonext)[0]:
                    qall[i] += qall[i+1]; qall[i+1] = 0
                    qcall[i] += qcall[i+1]; qcall[i+1] = 0
                    qpall[i] += qpall[i+1]; qpall[i+1] = 0
                    qpcall[i] += qpcall[i+1]; qpcall[i+1] = 0
                nummerged = nz.sum() - (qcall!=0).sum()
                print('merged', nummerged, 'elements of q')
            else:
                nummerged = 0

            # compute the statistics
            q = qall[nz]
            qc = qcall[nz]
            qp = qpall[nz]
            qpc = qpcall[nz]

            std = np.linalg.norm(q, ord=2)
            stdc = np.linalg.norm(qc, ord=2)
            stdp = np.linalg.norm(qp, ord=2)
            stdpc = np.linalg.norm(qpc, ord=2)

            qloci = np.array([
                qall[start:end].sum()
                for start, end in locus_inds])
            q2loci = np.array([
                (qall[start:end]**2).sum()
                for start, end in locus_inds])
            q4loci = np.array([
                (qall[start:end]**4).sum()
                for start, end in locus_inds])

            prod = (qloci**2).sum() - q2loci.sum()
            prod_std = np.sqrt(
                    2*((q2loci**2).sum() - q4loci.sum()))

            qcloci = np.array([
                qcall[start:end].sum()
                for start, end in locus_inds])
            qc2loci = np.array([
                (qcall[start:end]**2).sum()
                for start, end in locus_inds])
            qc4loci = np.array([
                (qcall[start:end]**4).sum()
                for start, end in locus_inds])

            prodc = (qcloci**2).sum() - qc2loci.sum()
            prodc_std = np.sqrt(
                    2*((qc2loci**2).sum() - qc4loci.sum()))

            results = results.append({
                'pheno':nice_ss_name,
                'annot':name,
                'v_num_merged':nummerged,
                'v_mean':v[nz].mean(),
                'v_std':v[nz].std(),
                'v_norm0':nz.sum(),
                'v_norm2':np.linalg.norm(v[nz], ord=2),
                'v_norm4':np.linalg.norm(v[nz], ord=4),
                'v_norm2o4':np.linalg.norm(v[nz], ord=2)/np.linalg.norm(v[nz], ord=4),
                'bhat_mean':bhat[nz].mean(),
                'bhat_std':bhat[nz].std(),

                'prod':prod,
                'prod_std':prod_std,
                'prod_z':prod/prod_std,
                'prod_p':st.norm.sf(prod/prod_std, 0, 1),

                'prodc':prodc,
                'prodc_std':prodc_std,
                'prodc_z':prodc/prodc_std,
                'prodc_p':st.norm.sf(prodc/prodc_std, 0, 1),

                'sum':q.sum(),
                'sum_std':std,
                'sum_z':q.sum()/std,
                'sum_p':st.chi2.sf((q.sum()/std)**2,1),
                'sum_corr':q.mean()/(v[nz].std()*bhat[nz].std()),
                'sum_corr_se':(q.std()/np.sqrt(nz.sum()))/(v[nz].std()*bhat[nz].std()),

                'sumc':qc.sum(),
                'sumc_z':qc.sum()/stdc,
                'sumc_p':st.chi2.sf((qc.sum()/stdc)**2,1),
                'sumc_corr':qc.mean()/(vc[nz].std()*bhatc[nz].std()),
                'sumc_corr_se':(qc.std()/np.sqrt(nz.sum()))/(vc[nz].std()*bhatc[nz].std()),

                'sump':qp.sum(),
                'sump_z':qp.sum()/stdp,
                'sump_p':st.chi2.sf((qp.sum()/stdp)**2,1),

                'sumpc':qpc.sum(),
                'sumpc_z':qpc.sum()/stdpc,
                'sumpc_p':st.chi2.sf((qpc.sum()/stdpc)**2,1)},
                ignore_index=True)
            print(results.iloc[-1])
            del v, vc, q, qc, qp, qpc

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
    parser.add_argument('-no-finemap', default=False, action='store_true',
            help='dont use the finemapped sumstats, use the normal sumstats')
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
    parser.add_argument('--loci',
            default='/groups/price/yakir/data/reference/dixon_IMR90.TADs.hg19.bed',
            help='path to UCSC bed file containing one bed interval per locus')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
