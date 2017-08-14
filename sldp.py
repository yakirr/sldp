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
import gprim.dataset as gd
import pyutils.memo as memo
import weights
import chunkstats as cs


def main(args):
    # basic initialization
    mhc_bp = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)
    pheno_name = args.pss_chr.split('/')[-2].split('.')[0]
    if args.seed is not None:
        np.random.seed(args.seed)

    # read in names of background annotations and marginal annotations
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    marginal_names = sum([a.names(22, RV=True) for a in annots], [])
    marginal_names = [n for n in marginal_names if '.R' in n]
    marginal_infos = pd.concat([a.info_df(args.chroms) for a in annots], axis=0)
    backgroundannots = [ga.Annotation(annot) for annot in args.background_sannot_chr]
    background_names = sum([a.names(22, True) for a in backgroundannots], [])
    background_names = [n for n in background_names if '.R' in n]
    print('background annotations:', background_names)
    print('marginal annotations:', marginal_names)
    if len(set(background_names) & set(marginal_names)) > 0:
        raise ValueError('the background annotation names and the marginal annotation '+\
                'names must be disjoint sets')

    # read in ldblocks and remove ones that overlap mhc
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & \
            (ldblocks.end > mhc_bp[0]) & \
            (ldblocks.start < mhc_bp[1])
    ldblocks = ldblocks[~mhcblocks]

    # read information about sumstats
    sumstats_info = pd.read_csv(args.pss_chr+'info', sep='\t')
    sigma2g = sumstats_info.loc[0].sigma2g
    h2g = sumstats_info.loc[0].h2g

    # read in sumstats and annots, and compute numerator and denominator of regression for
    # each ldblock. These will later be processed and aggregated
    numerators = dict(); denominators = dict()
    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)

        # get refpanel snp metadata 
        snps = refpanel.bim_df(c)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # read sumstats
        print('reading sumstats')
        ss = pd.read_csv(args.pss_chr+str(c)+'.pss.gz', sep='\t',
                usecols=['N',args.weights])
        print(np.isnan(ss[args.weights]).sum(), 'sumstats nans out of', len(ss))
        snps['Winv_ahat'] = ss[args.weights]
        snps['N'] = ss.N
        snps['typed'] = snps.Winv_ahat.notnull()

        # read annotations
        print('reading annotations')
        for annot in backgroundannots+annots:
            mynames = [n for n in annot.names(22, RV=True) if '.R' in n] #names of annotations
            print(time.time()-t0, ': reading annot', annot.filestem())
            print('adding', mynames)
            snps = pd.concat([snps, annot.RV_df(c)[mynames]], axis=1)
            if (~np.isfinite(snps[mynames].values)).sum() > 0:
                raise ValueError('There should be no nans in the postprocessed annotation. '+\
                        'But there are '+str((~np.isfinite(snps[mynames].values)).sum()))

        # make sure things are in the order we think they are
        if (np.array(background_names+marginal_names) !=
                snps.columns.values[-len(background_names+marginal_names):]).any():
            raise ValueError('Merged annotations are not in the right order')

        # perform computations
        for ldblock, X, meta, ind in refpanel.block_data(
                ldblocks, c, meta=snps, genos=False, verbose=0):
            if meta.typed.sum() == 0 or \
                    not os.path.exists(args.svd_stem+str(ldblock.name)+'.R.npz'):
                # no typed snps/hm3 snps in this block. set num snps to 0
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            if (meta[background_names+marginal_names] == 0).values.all():
                # annotations are all 0 in this block
                ldblocks.loc[ldblock.name, 'M_H'] = 0
                continue
            # record the number of typed snps in this block
            ldblocks.loc[ldblock.name, 'M_H'] = meta.typed.sum()

            # load regression weights and prepare for regression computation
            meta_t = meta[meta.typed]
            N = meta_t.N.mean()
            if args.weights == 'Winv_ahat_h' or args.weights == 'Winv_ahat_hlN':
                R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
                R2 = None
                if len(R['U']) != len(meta):
                    raise ValueError('regression wgts dimension must match regression snps')
            elif args.weights == 'Winv_ahat_h2' or args.weights == 'Winv_ahat':
                R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
                R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
                if len(R['U']) != len(meta) or len(R2['U']) != len(meta):
                    raise ValueError('regression wgts dimension must match regression snps')
            else:
                R = None; R2 = None

            # multiply ahat by the weights
            Winv_RV = weights.invert_weights(
                    R, R2, sigma2g, N, meta[background_names+marginal_names].values,
                    typed=meta.typed, mode=args.weights)

            numerators[ldblock.name] = \
                (meta_t[background_names+marginal_names].T.dot(
                    meta_t.Winv_ahat)).values/1e6
            denominators[ldblock.name] = meta_t[background_names+marginal_names].T.dot(
                    Winv_RV[meta.typed]).values/1e6
        memo.reset()

    # gete data for jackknifing
    print('jackknifing')
    chunk_nums, chunk_denoms, loo_nums, loo_denoms = cs.collapse_to_chunks(
            ldblocks,
            numerators,
            denominators,
            args.jk_blocks)

    # compute final results
    results = pd.DataFrame()
    for i, name in enumerate(marginal_names):
        print(i, name)
        k = marginal_names.index(name)
        mu = cs.get_est(sum(chunk_nums), sum(chunk_denoms), k, len(background_names))
        results = results.append({
            'pheno':pheno_name,
            'annot':name},
            ignore_index=True)

        # compute q
        q = cs.getq(chunk_nums, chunk_denoms, len(background_names), k)

        # exact sign-flipping
        score = q.sum()
        p, z = cs.signflip(q, args.T, printmem=True)
        results.loc[i,'sf_z'] = z
        results.loc[i,'sf_p'] = p

        # jackknife
        se = cs.jackknife_se(mu, loo_nums, loo_denoms, k, len(background_names))
        results.loc[i,'mu'] = mu
        results.loc[i,'jk_se'] = se
        results.loc[i,'jk_z'] = mu/se
        results.loc[i,'jk_p'] = st.chi2.sf((mu/se)**2,1)

        # add metadata about v and scale point estimate to get r_f
        for prop in ['supp', 'sqnorm', 'supp_5_50', 'sqnorm_5_50']:
            results.loc[i,prop] = marginal_infos.loc[name[:-2], prop]

        # add estimates of r_f and h2v
        M = marginal_infos.loc[name[:-2],'M']
        results.loc[i,'h2g'] = h2g
        results.loc[i,'r_f'] = mu * np.sqrt(
                results.loc[i].sqnorm / (M*sigma2g))
        results.loc[i,'h2v_h2g'] = results.loc[i].r_f**2 - \
                results.loc[i].jk_se**2 * results.loc[i].sqnorm / (M*sigma2g)
        results.loc[i,'h2v'] = results.loc[i].h2v_h2g * (M*sigma2g)

    print(results)
    print('writing results to', args.outfile_stem + '.gwresults')
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False, na_rep='nan')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--pss-chr', #required=True,
            default='/groups/price/yakir/data/sumstats.hm3/processed/CD.KG3.95/',
            help='one or more paths to .pss.gz files, without chr number or extension')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=[
                '/groups/price/yakir/data/annot/basset/processed.a9/HaibGm12878Sp1Pcr1x/',
                '/groups/price/yakir/data/annot/basset/processed.a9/HaibH1hescUsf1Pcr1x/'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--weights', default='Winv_ahat_h',
            help='which set of processed sumstats to use')
    parser.add_argument('--background-sannot-chr', nargs='+',
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
    parser.add_argument('--T', type=int, default=1000000,
            help='number of times to sign flip for empirical p-values')
    parser.add_argument('--seed', default=None, type=int,
            help='Seed random number generator to a certain value. Off by default')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per ld block')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    print('=====')
    print(' '.join(sys.argv))
    print('=====')
    args = parser.parse_args()
    pretty.print_namespace(args)
    print('=====')

    main(args)
