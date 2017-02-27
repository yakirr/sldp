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
import gprim.dataset as gd; reload(gd)
import pyutils.memo as memo
import weights; reload(weights)


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
    nice_ss_name = args.ssjk_chr.split('/')[-2].split('.')[0]
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    marginal_names = sum([a.names(22, True) for a in annots], [])
    baselineannots = [ga.Annotation(annot) for annot in args.baseline_sannot_chr]
    baseline_names = sum([a.names(22, True) for a in baselineannots], [])
    bias_names = sum([[n+'.vTRv' for n in a.names(22)] for a in baselineannots+annots], [])
    print('baseline annotations:', baseline_names)
    print('marginal annotations:', marginal_names)

    # read in ldblocks and remove ones that overlap mhc
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]
    ldblocks['filename'] = ldblocks.index.values
    print(len(ldblocks), 'ld blocks after removing MHC')
    for annot in baselineannots+annots:
        if os.path.exists(annot.filestem()+'ldblocks'):
            myldblocks = pd.read_csv(annot.filestem()+'ldblocks', sep='\t')
            ldblocks = pd.merge(ldblocks, myldblocks, how='left',
                    on=['chr','start','end']).sort_values(by='filename')
    ldblocks.set_index('filename', inplace=True)

    # read in sumstats and annots, and compute numerator and denominator of regression for
    # each ldblock. These will later be jackknifed
    numerators = dict(); denominators = dict(); olddenominators = dict()
    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata and sumstats for this chromosome
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')
        print('reading sumstats')
        ss = pd.read_csv(args.ssjk_chr+str(c)+'.ss.jk.gz', sep='\t',
                usecols=['N',args.weightedss])
        sigma2g = pd.read_csv(args.ssjk_chr+'info', sep='\t').sigma2g.values[0]
        print(np.isnan(ss[args.weightedss]).sum(), 'nans out of', len(ss))

        # merge annot and sumstats
        print('merging')
        snps['Winv_ahat'] = ss[args.weightedss]
        snps['N'] = ss.N
        snps['typed'] = snps.Winv_ahat.notnull()

        # read in annotations
        print('reading annotations')
        for annot in baselineannots+annots:
            mynames = annot.names(22, True) # names of annotations
            print(time.time()-t0, ': reading annot', annot.filestem())
            snps = pd.concat([snps, annot.RV_df(c)], axis=1)

            print('there are', (~np.isfinite(snps[mynames].values)).sum(),
                    'nans in the annotation. This number should be 0.')

        # perform computations
        for ldblock, X, meta, ind in refpanel.block_data(
                ldblocks, c, meta=snps, genos=False, verbose=0):
            # print(meta.typed.sum(), 'typed snps')
            if meta.typed.sum() == 0 or \
                    not os.path.exists(args.svd_stem+str(ldblock.name)+'.R.npz'):
                print('no typed snps/hm3 snps in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            if (meta[baseline_names+marginal_names] == 0).values.all():
                print('annotations are all 0 in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            ldblocks.loc[ldblock.name, 'M_H'] = len(meta)

            R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
            R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
            meta_t = meta[meta.typed]
            N = meta_t.N.mean()
            biases = ldblocks.loc[ldblock.name, bias_names].values.astype(float)

            # multiply ahat by the weights
            Winv_RV_h = weights.invert_weights(
                    R, R2, sigma2g, N, meta[baseline_names+marginal_names].values,
                    typed=meta.typed, mode=args.weightedss)

            numerators[ldblock.name] = \
                (meta_t[baseline_names+marginal_names].T.dot(meta_t.Winv_ahat)/1e6).values
            D = meta_t[baseline_names+marginal_names].T.dot(Winv_RV_h[meta.typed]).values
            olddenominators[ldblock.name] = D/1e6
            if args.no_bc:
                tr = 0
            else:
                tr = weights.trace_inv(R, R2, sigma2g, N,
                        meta[baseline_names+marginal_names].values,
                        typed=meta.typed, mode=args.weightedss)
            D = D - tr * np.diag(biases) / refpanel.N()
            D /= 1e6
            denominators[ldblock.name] = D

    # start jackknifing
    print('jackknifing')
    ## create jk blocks
    ldblocks.M_H.fillna(0, inplace=True)
    totalM = ldblocks.M_H.sum()
    jkblocksize = totalM / args.jk_blocks
    avgldblocksize = totalM / (ldblocks.M_H != 0).sum()
    blockendpoints = [0]
    currldblock = 0; currsize = 0
    while currldblock < len(ldblocks):
        while currsize <= max(1,jkblocksize-avgldblocksize/2) and currldblock < len(ldblocks):
            currsize += ldblocks.iloc[currldblock].M_H
            currldblock += 1
        currsize = 0
        blockendpoints += [currldblock]
    ## collapse data within jk blocks
    jknumerators = []; jkdenominators = []; jksizes = []; jkweights = []
    for n, (i,j) in enumerate(zip(blockendpoints[:-1], blockendpoints[1:])):
        ldblock_ind = [l for l in ldblocks.iloc[i:j].index if l in numerators.keys()]
        if len(ldblock_ind) > 0:
            jknumerators.append(sum(
                [numerators[l] for l in ldblock_ind]))
            jkdenominators.append(sum(
                [denominators[l] for l in ldblock_ind]))
            jksizes.append(np.sum(ldblocks.iloc[i:j].M_H.values))
            jkweights.append(np.diagonal(jkdenominators[-1]))
    print('jk sizes:', jksizes)
    print('jk weights:', jkweights)
    ## compute LOO sufficient statistics
    loonumerators = []; loodenominators = []
    for i in range(len(jknumerators)):
        loonumerators.append(sum(jknumerators[:i]+jknumerators[(i+1):]))
        loodenominators.append(sum(jkdenominators[:i]+jkdenominators[(i+1):]))
    ## produce estimates and SE's for each marginal annotation
    def get_est(num, denom, name):
        k = len(baseline_names); i = marginal_names.index(name)
        ind = range(k)+[k+i]
        num = num[ind]
        denom = denom[ind][:,ind]
        return np.linalg.solve(denom, num)[-1]
    def jackknife_se(est, loonumerators, loodenominators, jkweights):
        m = np.array(jkweights)
        theta_notj = [get_est(nu, de, name) for nu, de in zip(loonumerators, loodenominators)]
        g = len(jkweights)
        n = m.sum()
        h = n/m
        theta_J = g*est - ((n-m)/n*theta_notj).sum()
        tau = est*h - (h-1)*theta_notj
        return np.sqrt(np.mean((tau - theta_J)**2/(h-1)))

    results = pd.DataFrame()
    for i, name in enumerate(marginal_names):
        total_est = get_est(sum(jknumerators), sum(jkdenominators), name)
        loo_est = [get_est(n, d, name) for n, d in zip(loonumerators, loodenominators)]
        if args.weight_jk == 'none':
            se = jackknife_se(total_est, loonumerators, loodenominators, [1]*len(jkweights))
        elif args.weight_jk == 'snps':
            se = jackknife_se(total_est, loonumerators, loodenominators, jksizes)
        elif args.weight_jk == 'annot':
            se = jackknife_se(total_est, loonumerators, loodenominators,
                    [j[len(baseline_names)+i] for j in jkweights])
        else: # annotsqrt
            se = jackknife_se(total_est, loonumerators, loodenominators,
                    [np.sqrt(j[len(baseline_names)+i]) for j in jkweights])
        results = results.append({
            'pheno':nice_ss_name,
            'annot':name,
            'est':total_est,
            'se':se,
            'z':total_est/se,
            'p':st.chi2.sf((total_est/se)**2,1)},
            ignore_index=True)

    print('writing results')
    print(results)
    results.to_csv(args.outfile_stem + '.jk.gwresults', sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--ssjk-chr', #required=True,
            default='/groups/price/yakir/data/simsumstats/GERAimp.wim5unm/Sp1_varenrichment/'+\
                    '1/all.KG3.95/',
            help='one or more paths to .ss.jk.gz files, without chr number or extension')
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
                '1000G.EUR.QC.hm3_noMHC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--svd-stem',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/svds_95percent/',
            help='path to truncated svds of reference panel, by LD block')
    parser.add_argument('--jk-blocks', type=int, default=100,
            help='number of jackknife blocks to use')
    parser.add_argument('--weight-jk', default='none',
            help='what to weight the jackknife by. Options are none, snps, annot')
    parser.add_argument('-no-bc', default=False, action='store_true',
            help='dont use bias correction for denominator of regression')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per ld block')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
