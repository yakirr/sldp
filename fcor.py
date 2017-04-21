from __future__ import print_function, division
import argparse, os, gc, time, resource, sys
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
    baseline_names = [n for n in baseline_names if n=='minor1.R']

    print('baseline annotations:', baseline_names)
    print('marginal annotations:', marginal_names)

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
        # get refpanel snp metadata 
        snps = refpanel.bim_df(c)
        l_all = pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz', sep='\t').rename(
                columns={'L2':'l_all'})
        l_reg = pd.read_csv(args.ldscores_reg_chr+str(c)+'.l2.ldscore.gz', sep='\t').rename(
                columns={'L2':'l_reg'})
        snps = ga.smart_merge(snps, l_all, drop_from_y=['CHR','BP'], how='left')
        snps = ga.smart_merge(snps, l_reg, drop_from_y=['CHR','BP'], how='left')
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # read sumstats
        print('reading sumstats')
        ss = pd.read_csv(args.ssjk_chr+str(c)+'.ss.jk.gz', sep='\t',
                usecols=['N',args.weightedss])
        sigma2g = pd.read_csv(args.ssjk_chr+'info', sep='\t').sigma2g.values[0]
        print(np.isnan(ss[args.weightedss]).sum(), 'sumstats nans out of', len(ss))
        snps['Winv_ahat'] = ss[args.weightedss]
        snps['N'] = ss.N
        snps['typed'] = snps.Winv_ahat.notnull()
        snps['MAF'] = refpanel.frq_df(c).MAF.values

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

        # sanity check
        if (np.array(baseline_names+marginal_names) !=
                snps.columns.values[-len(baseline_names+marginal_names):]).any():
            print('ERROR')
            sys.exit(1)

        # ignore some regression snps if necessary
        if args.ldscore_percentile is not None:
            to_threshold = 'l_all'
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
                # print('no typed snps/hm3 snps in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            if (meta[baseline_names+marginal_names] == 0).values.all():
                # print('annotations are all 0 in this block')
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            ldblocks.loc[ldblock.name, 'M_H'] = meta.typed.sum()

            meta_t = meta[meta.typed]
            N = meta_t.N.mean()
            if args.weightedss == "Winv_ahat_h":
                R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
                R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
                if len(R['U']) != len(meta) or len(R2['U']) != len(meta):
                    print('theres a mismatch')
                    sys.exit(1)
            else:
                R = None; R2 = None

            # multiply ahat by the weights
            Winv_RV = weights.invert_weights(
                    R, R2, meta.l_all.values, meta.l_reg.values,
                    sigma2g, N, meta[baseline_names+marginal_names].values,
                    typed=meta.typed, mode=args.weightedss)

            numerators[ldblock.name] = \
                (meta_t[baseline_names+marginal_names].T.dot(
                    meta_t.Winv_ahat)).values/1e6
            denominators[ldblock.name] = meta_t[baseline_names+marginal_names].T.dot(
                    Winv_RV[meta.typed]).values/1e6
        memo.reset()

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
        print(i, name)
        total_est = get_est(sum(jknumerators), sum(jkdenominators), name)
        results = results.append({
            'pheno':nice_ss_name,
            'annot':name},
            ignore_index=True)

        # sf approx
        k = marginal_names.index(name)
        q = np.array([num[len(baseline_names)+k] for num in jknumerators])

        # (find the bias of the coin flips if there's a baseline annotation)
        if len(baseline_names) > 0:
            u = np.array([denom[0,len(baseline_names)+k] for denom in jkdenominators])
            w = np.array([denom[0,0] for denom in jkdenominators])
            mu = u.sum()/np.linalg.norm(u, ord=1)
        else:
            u = np.zeros(len(q))
            mu = 0
        mu = mu/2 + 0.5

        score = q.sum()
        optscore = np.abs(q).sum()
        se = np.sqrt(np.sum(q**2))
        results.loc[i,'sfapprox_score'] = score
        results.loc[i,'sfapprox_se'] = se
        results.loc[i,'sfapprox_z'] = score/se
        results.loc[i,'sfapprox_p'] = st.chi2.sf((score/se)**2,1)
        results.loc[i,'opt_z'] = optscore/se

        # sf exact
        null = []
        test = []
        T = 100000
        print(((np.sign(u) == 0)&(q != 0)).sum())
        print((np.sign(u)==0).sum())
        print((np.sign(w)==0).sum())

        import pdb; pdb.set_trace()
        for t in range(T):
            s = (-1)**np.random.binomial(1,mu,size=len(q))
            s = -s*np.sign(u)
            null.append(s.dot(q))
            test.append(s.dot(u))
        p = ((np.abs(null) >= np.abs(score)).sum()) / float(T)
        p = min(max(p,1./T), 1)
        se = np.abs(score)/np.sqrt(st.chi2.isf(p,1))
        results.loc[i,'sf_score'] = score
        results.loc[i,'sf_se'] = se
        results.loc[i,'sf_z'] = score/se
        results.loc[i,'sf_p'] = p

        # jk
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
        results.loc[i,'jk_score'] = score
        results.loc[i,'jk_se'] = se
        results.loc[i,'jk_z'] = score/se
        results.loc[i,'jk_p'] = st.chi2.sf((score/se)**2,1)

    print('writing results')
    print(results)
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False, na_rep='nan')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--ssjk-chr', #required=True,
            default='/groups/price/yakir/data/simsumstats/GERAimp.wim9nm/'+\
                    'no_enrichment_3pcs_affy/1/all.KG3.95/',
            help='one or more paths to .ss.jk.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/simannot/KG3.wim9nm/HaibGm12878Sp1Pcr1x/KG3/',
                    '/groups/price/yakir/data/simannot/KG3.wim9nm/HaibH1hescUsf1Pcr1x/KG3/'],
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
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/LDscore/LDscore.',
            help='path to LD scores at a smallish set of SNPs. LD should be computed '+\
                    'to all potentially causal snps. This is used for '+\
                    'heritability estimation')
    parser.add_argument('--ldscores-reg-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to LD scores, where LD is computed to regression SNPs only.')
    parser.add_argument('--ldscore-percentile', default=None,
            help='snps with ld score below this threshold wont be used for regression')
    parser.add_argument('--svd-stem',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/svds_95percent/',
            help='path to truncated svds of reference panel, by LD block')
    parser.add_argument('--jk-blocks', type=int, default=100,
            help='number of jackknife blocks to use')
    parser.add_argument('--weight-jk', default='none',
            help='what to weight the jackknife by. Options are none, snps, annot')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per ld block')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
