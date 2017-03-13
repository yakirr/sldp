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
    marginal_names = [n for n in marginal_names if '.R' in n]
    baselineannots = [ga.Annotation(annot) for annot in args.baseline_sannot_chr]
    baseline_names = sum([a.names(22, True) for a in baselineannots], [])
    baseline_names = [n for n in baseline_names if True or '.R' in n] #TODO: remove the "not"
        # it's currently this way so that v gets added into the regression
    bias_names = sum([[n for n in a.names(22) if '.R' not in n] for a in baselineannots+annots], [])
    print('baseline annotations:', baseline_names)
    print('marginal annotations:', marginal_names)
    print('bias names:', bias_names)

    # read in ldblocks and remove ones that overlap mhc
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & (ldblocks.end > mhc[0]) & (ldblocks.start < mhc[1])
    ldblocks = ldblocks[~mhcblocks]

    # read in sumstats and annots, and compute numerator and denominator of regression for
    # each ldblock. These will later be jackknifed
    numerators = dict(); denominators = dict(); olddenominators = dict()
    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata and sumstats for this chromosome
        snps = refpanel.bim_df(c)
        if args.ldscores_chr is not None:
            l2 = pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz', sep='\t')
        snps = pd.merge(snps, l2[['SNP','L2']], on='SNP', how='left')
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')
        print('reading sumstats')
        ss = pd.read_csv(args.ssjk_chr+str(c)+'.ss.jk.gz', sep='\t',
                usecols=['N',args.weightedss])
        sigma2g = pd.read_csv(args.ssjk_chr+'info', sep='\t').sigma2g.values[0]
        print(np.isnan(ss[args.weightedss]).sum(), 'sumstats nans out of', len(ss))

        # merge annot and sumstats
        print('merging')
        snps['Winv_ahat'] = ss[args.weightedss]
        snps['N'] = ss.N
        snps['typed'] = snps.Winv_ahat.notnull()
        snps['MAF'] = refpanel.frq_df(c).MAF.values
        snps.loc[snps.typed & snps.L2.isnull()] = 1

        # read in annotations
        print('reading annotations')
        for annot in baselineannots+annots:
            mynames_ = annot.names(22, True) # names of annotations
            mynames = []
            if annot in baselineannots:
                mynames += [n for n in mynames_ if n in baseline_names]
            else:
                mynames += [n for n in mynames_ if n in marginal_names]
            print(time.time()-t0, ': reading annot', annot.filestem())
            print('adding', mynames)
            snps = pd.concat([snps, annot.RV_df(c)[mynames]], axis=1)

            print('there are', (~np.isfinite(snps[mynames].values)).sum(),
                    'nans in the annotation. This number should be 0.')

        # ignore some regression snps if necessary
        if args.ldscore_percentile is not None:
            to_threshold = 'L2'
            thresh = np.percentile(np.nan_to_num(snps[snps.typed][to_threshold]),
                    args.ldscore_percentile)
            print('thresholding', to_threshold, 'at', thresh)
            print('going from', snps.typed.sum(), 'regression snps')
            snps.loc[snps[to_threshold] <= thresh, 'typed'] = False
            print('to', snps.typed.sum(), 'regression snps')

        # perform computations
        for ldblock, X, meta, ind in refpanel.block_data(
                ldblocks, c, meta=snps, genos=False, verbose=0):
            # print(ldblock)
            # print(len(meta), 'snps, of which', meta.typed.sum(), 'are typed snps')
            # print(len(ind), 'is length of ind')
            if meta.typed.sum() == 0 or \
                    not os.path.exists(args.svd_stem+str(ldblock.name)+'.R.npz'):
                print('no typed snps/hm3 snps in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            if (meta[baseline_names+marginal_names] == 0).values.all():
                print('annotations are all 0 in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            ldblocks.loc[ldblock.name, 'M_H'] = meta.typed.sum()

            if args.weightedss != "Winv_ahat_I":
                R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
                R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
            else:
                R = None; R2 = None
            meta_t = meta[meta.typed]
            N = meta_t.N.mean()

            # multiply ahat by the weights
            Winv_RV_h = weights.invert_weights(
                    R, R2, sigma2g, N, meta[baseline_names+marginal_names].values,
                    typed=meta.typed, mode=args.weightedss)

            numerators[ldblock.name] = \
                (meta_t[baseline_names+marginal_names].T.dot(
                    meta_t.Winv_ahat)/1e6).values
            D = meta_t[baseline_names+marginal_names].T.dot(
                    Winv_RV_h[meta.typed]).values

            olddenominators[ldblock.name] = D/1e6
            if not args.no_bc:
                biases = pd.read_csv(annots[-1].filestem()+'VTRV.'+str(ldblock.name), sep='\t',
                        index_col=0)
                biases = biases.loc[bias_names, bias_names].values
                tr = weights.trace_inv(R, R2, sigma2g, N,
                        meta[baseline_names+marginal_names].values,
                        typed=meta.typed, mode=args.weightedss)
                D = D - tr * biases / refpanel.N()
            denominators[ldblock.name] = D / 1e6

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
    # print('jk sizes:', jksizes)
    # print('jk weights:', jkweights)
    ## compute LOO sufficient statistics
    loonumerators = []; loodenominators = []
    for i in range(len(jknumerators)):
        loonumerators.append(sum(jknumerators[:i]+jknumerators[(i+1):]))
        loodenominators.append(sum(jkdenominators[:i]+jkdenominators[(i+1):]))
    ## produce estimates and SE's for each marginal annotation
    def get_est(num, denom, name):
        k = len(baseline_names); i = marginal_names.index(name)
        avoid = baseline_names.index(name) if name in baseline_names else -1
        ind = [j for j in range(k) if j != avoid] + [k+i]
        num = num[ind]
        denom = denom[ind][:,ind]
        try:
            return np.linalg.solve(denom, num)[-1]
        except np.linalg.linalg.LinAlgError:
            return np.nan
    def jackknife_se(est, loonumerators, loodenominators, jkweights, name):
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
        if args.sf or args.sf_approx:
            k = marginal_names.index(name)
            q = np.array([num[len(baseline_names)+k] for num in jknumerators])
            score = q.sum()
            if args.sf_approx:
                se = np.sqrt(np.sum(q**2))
            else:
                null = []
                T = 100000
                for i in range(T):
                    s = (-1)**np.random.binomial(1,0.5,size=len(q))
                    null.append(s.dot(q))
                p = ((np.abs(null) >= np.abs(score)).sum() + 1) / float(T)
                se = np.abs(score)/np.sqrt(st.chi2.isf(p,1))
        else:
            score = total_est
            loo_est = [get_est(n, d, name) for n, d in zip(loonumerators, loodenominators)]
            if args.weight_jk == 'none':
                se = jackknife_se(total_est, loonumerators, loodenominators, [1]*len(jkweights),
                        name)
            elif args.weight_jk == 'snps':
                se = jackknife_se(total_est, loonumerators, loodenominators, jksizes, name)
            elif args.weight_jk == 'annot':
                se = jackknife_se(total_est, loonumerators, loodenominators,
                        [j[len(baseline_names)+i] for j in jkweights], name)
            else: # annotsqrt
                se = jackknife_se(total_est, loonumerators, loodenominators,
                        [np.sqrt(j[len(baseline_names)+i]) for j in jkweights], name)
        results = results.append({
            'pheno':nice_ss_name,
            'annot':name,
            'est':score,
            'se':se,
            'z':score/se,
            'p':st.chi2.sf((score/se)**2,1)},
            ignore_index=True)

    print('writing results')
    print(results)
    results.to_csv(args.outfile_stem + '.jk.gwresults', sep='\t', index=False, na_rep='nan')
    print('done')
    # print('==== point estimate of full joint fit ===')
    # print(baseline_names + marginal_names)
    # N = sum(numerators.values())
    # D = sum(denominators.values())
    # oD = sum(olddenominators.values())
    # print(N)
    # print(D)
    # print(oD)
    # print()
    # try:
    #     print(np.linalg.solve(D,N))
    #     print(np.linalg.solve(oD,N))
    # except np.linalg.linalg.LinAlgError:
    #     print('singular matrix, could not get joint fit')


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
    parser.add_argument('-sf', default=False, action='store_true',
            help='print z score for marginal sign-flip test')
    parser.add_argument('-sf-approx', default=False, action='store_true',
            help='print z score for marginal sign-flip test')
    parser.add_argument('--baseline-sannot-chr', nargs='+',
            default=[])
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.hm3_noMHC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ldscores-chr', default=None,
            help='path to ldcores computed to all potentially causal snps. ld scores are '+\
                    'only needed at regression snps')
    parser.add_argument('--ldscore-percentile', default=None,
            help='snps with ld score below this threshold wont be used for regression')
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
