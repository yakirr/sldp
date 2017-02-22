from __future__ import print_function, division
import argparse, os, gc, time, resource
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


def mem():
    print('memory usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, 'Mb')


def get_locus_inds(snps, loci, chroms):
    loci.sort_values(by='CHR', inplace=True)
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
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.sssf_chr.split('/')[-2].split('.')[0]
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    baselineannots = [ga.Annotation(annot) for annot in args.baseline_sannot_chr]
    baseline_names = sum([a.names(22) for a in baselineannots], [])
    print('baseline annotations:', baseline_names)

    # read in ldblocks and remove ones that overlap mhc
    loci = pd.read_csv(args.loci, delim_whitespace=True, header=None,
            names=['CHR','start', 'end'])
    mhcblocks = (loci.CHR == 'chr6') & (loci.end > mhc[0]) & (loci.start < mhc[1])
    loci = loci[~mhcblocks]
    print(len(loci), 'loci after removing MHC')

    print('getting locus indices')
    snps = pd.concat([refpanel.bim_df(c) for c in args.chroms], axis=0)
    locus_inds = get_locus_inds(snps, loci, args.chroms)
    maf = pd.concat([refpanel.frq_df(c)[['MAF']] for c in args.chroms], axis=0).MAF.values
    del snps; gc.collect()

    print('reading sumstats')
    ss = np.concatenate([
        pd.read_csv(args.sssf_chr+str(c)+'.ss.sf.gz', sep='\t',
            usecols=['R_'+args.weightedss])['R_'+args.weightedss].values
        for c in args.chroms])
    print(np.isnan(ss).sum(), 'nans out of', len(ss))

    # read in baseline annotations
    baseline = []
    for bannot in baselineannots:
        baseline.append(pd.concat([bannot.sannot_df(c) for c in args.chroms], axis=0))
    if len(baseline) > 0:
        baseline = pd.concat(baseline, axis=1)
        B = baseline[baseline_names].values
    else:
        B = None

    results = pd.DataFrame()
    t0 = time.time()
    for annot in annots:
        names = annot.names(22) # names of annotations
        print(time.time()-t0, ': reading annot', annot.filestem())
        a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)

        print('creating V')
        print('there are', (~np.isfinite(a[names].values)).sum(), 'nans in the annotation.',
                'This number should be 0.')
        V = a[names].values

        for i, name in enumerate(names):
            print(i, name)
            v = V[:,i]
            nz = (v!=0) & np.isfinite(ss)

            if nz.sum() == 0:
                continue

            # residualize out baseline model covariates
            print('accounting for baseline model')
            if B is not None:
                vc = v.copy()
                coeffs = np.linalg.solve(B[nz].T.dot(B[nz]), B[nz].T.dot(v[nz]))
                print(coeffs)
                vc[nz] = v[nz] - B[nz].dot(coeffs)
            else:
                vc = v
            v[~nz] = 0; vc[~nz] = 0

            # compute q
            qall = v*ss
            qcall = vc*ss

            # compute the statistics
            q = qall[nz]
            qc = qcall[nz]

            std = np.linalg.norm(q, ord=2)
            stdc = np.linalg.norm(qc, ord=2)

            qloci = np.array([
                np.nan_to_num(qall[start:end]).sum()
                for start, end in locus_inds])
            q2loci = np.array([
                np.nan_to_num(qall[start:end]**2).sum()
                for start, end in locus_inds])
            q4loci = np.array([
                np.nan_to_num(qall[start:end]**4).sum()
                for start, end in locus_inds])
            prod = (qloci**2).sum() - q2loci.sum()
            prod_std = np.sqrt(
                    2*((q2loci**2).sum() - q4loci.sum()))

            qcloci = np.array([
                np.nan_to_num(qcall[start:end]).sum()
                for start, end in locus_inds])
            qc2loci = np.array([
                np.nan_to_num(qcall[start:end]**2).sum()
                for start, end in locus_inds])
            qc4loci = np.array([
                np.nan_to_num(qcall[start:end]**4).sum()
                for start, end in locus_inds])
            prodc = (qcloci**2).sum() - qc2loci.sum()
            prodc_std = np.sqrt(
                    2*((qc2loci**2).sum() - qc4loci.sum()))

            results = results.append({
                'pheno':nice_ss_name,
                'annot':name,
                'v_mean':v[nz].mean(),
                'v_std':v[nz].std(),
                'v_norm0':nz.sum(),
                'v_norm2':np.linalg.norm(v[nz], ord=2),
                'v_norm4':np.linalg.norm(v[nz], ord=4),
                'v_norm2o4':np.linalg.norm(v[nz], ord=2)/np.linalg.norm(v[nz], ord=4),

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

                'sumc':qc.sum(),
                'sumc_std':stdc,
                'sumc_z':qc.sum()/stdc,
                'sumc_p':st.chi2.sf((qc.sum()/stdc)**2,1)},
                ignore_index=True)
            print(results.iloc[-1])
            del v, vc, q, qc

        del V; memo.reset(); gc.collect(); mem()

    print('writing results')
    results.to_csv(args.outfile_stem + '.sf.gwresults', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--sssf-chr', #required=True,
            default='/groups/price/yakir/data/simsumstats/GERAimp.wim5unm/Sp1_varenrichment/'+\
                    '1/all.KG3.95/',
            help='one or more paths to .ss.sf.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/processed.a8/8988T/',
                '/groups/price/yakir/data/annot/basset/processed.a8/A549/'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--weightedss', default='Winv_ahat_h',
            help='which set of processed sumstats to use')
    parser.add_argument('--baseline-sannot-chr', nargs='+',
            default=[])
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--loci',
            default='/groups/price/yakir/data/reference/dixon_IMR90.TADs.hg19.bed',
            help='path to UCSC bed file containing one bed interval per locus')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
