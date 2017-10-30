from __future__ import print_function, division
import argparse, os, time, sys
import numpy as np
import pandas as pd
import gprim.annotation as ga
import gprim.dataset as gd
import ypy.pretty as pretty
import ypy.memo as memo
import weights
import chunkstats as cs
import config


def main(args):
    # basic initialization
    mhc_bp = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)
    pheno_name = args.pss_chr.split('/')[-2].replace('.KG3.95','')
    if args.seed is not None:
        np.random.seed(args.seed)
        print('random seed:', args.seed)

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
            # record the number of typed snps in this block, and start- and end- snp indices
            ldblocks.loc[ldblock.name, 'M_H'] = meta.typed.sum()
            ldblocks.loc[ldblock.name, 'snpind_begin'] = min(meta.index)
            ldblocks.loc[ldblock.name, 'snpind_end'] = max(meta.index)+1

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

    # get data for jackknifing
    print('jackknifing')
    chunk_nums, chunk_denoms, loo_nums, loo_denoms, chunkinfo = cs.collapse_to_chunks(
            ldblocks,
            numerators,
            denominators,
            args.jk_blocks)

    # compute final results
    global q, results
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
        q, r, mux, muy = cs.residualize(chunk_nums, chunk_denoms, len(background_names), k)
        if args.verbose:
            print('writing verbose results to', args.outfile_stem+'.'+name)
            chunkinfo['q'] = q
            chunkinfo['r'] = r
            chunkinfo.to_csv(args.outfile_stem+'.'+name+'.chunks', sep='\t', index=False)
            coeffs = pd.DataFrame()
            coeffs['annot'] = background_names
            coeffs['mux'] = mux
            coeffs['muy'] = muy
            coeffs.to_csv(args.outfile_stem+'.'+name+'.coeffs', sep='\t', index=False)


        # exact sign-flipping
        p, z = cs.signflip(q, args.T, printmem=True, mode=args.stat)
        results.loc[i,'z'] = z
        results.loc[i,'p'] = p
        if args.more_stats:
            import scipy.stats as st
            results.loc[i,'qkurtosis'] = st.kurtosis(q)
            results.loc[i,'qstd'] = np.std(q)
            results.loc[i,'p_jk'] = st.chi2.sf((mu/se)**2,1)

        # jackknife
        se = cs.jackknife_se(mu, loo_nums, loo_denoms, k, len(background_names))
        results.loc[i,'mu'] = mu
        results.loc[i,'se(mu)'] = se

        # add metadata about v and scale point estimate to get r_f
        for prop in ['supp', 'sqnorm']:
            results.loc[i,prop] = marginal_infos.loc[name[:-2], prop]

        # add estimates of r_f and h2v
        M = marginal_infos.loc[name[:-2],'M']
        results.loc[i,'h2g'] = h2g
        results.loc[i,'r_f'] = mu * np.sqrt(
                results.loc[i].sqnorm / h2g)
        results.loc[i,'h2v_h2g'] = results.loc[i].r_f**2 - \
                results.loc[i].jk_se**2 * results.loc[i].sqnorm / (M*sigma2g)
        results.loc[i,'h2v'] = results.loc[i].h2v_h2g * h2g
        results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False, na_rep='nan')

    print(results)
    print('writing results to', args.outfile_stem + '.gwresults')
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False, na_rep='nan')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--outfile-stem', required=True,
            help='Path to an output file stem.')
    parser.add_argument('--pss-chr', required=True,
            help='Path to .pss.gz file, without chromosome number or .pss.gz extension. '+\
                    'This is the phenotype that SLDP will analyze.')
    parser.add_argument('--sannot-chr', nargs='+', required=True,
            help='One or more (space-delimited) paths to gzipped annot files, without '+\
                    'chromosome number or .sannot.gz extension. These are the annotations '+\
                    'that SLDP will analyze against the phenotype.')

    # optional arguments
    parser.add_argument('--background-sannot-chr', nargs='+', default=[],
            help='One or more (space-delimited) paths to gzipped annot files, without '+\
                    'chromosome number or .sannot.gz extension. These are the annotations '+\
                    'that SLDP will control for.')
    parser.add_argument('-verbose', default=False, action='store_true',
            help='Print additional information about each association studied. This '+\
                    'includes: '+\
                    'the covariance in each independent block of genome (.chunks files), '+\
                    'and the coefficients required to residualize any background '+\
                        'annotations out of the other annotations being analyzed.')
    parser.add_argument('-tell-me-stories', default=False, action='store_true',
            help='COMING SOON. '+\
                    'Print information about SNPs or loci that may be promising to study. '+\
                    'This includes: '+\
                    'Genome-wide significant SNPs with large values of the signed LD '+\
                        ' profile, and '+\
                    'lists of loci where the signed LD profile is highly correlated with '+\
                        'the GWAS signal in a direction consistent with the global effect.')
    parser.add_argument('-more-stats', default=False, action='store_true',
            help='Print additional statistis about q in results file')
    parser.add_argument('--T', type=int, default=1000000,
            help='number of times to sign flip for empirical p-values. Default is 10^6.')
    parser.add_argument('--jk-blocks', type=int, default=300,
            help='Number of jackknife blocks to use. Default is 300.')
    parser.add_argument('--weights', default='Winv_ahat_h',
            help='which set of regression weights to use. Default is Winv_ahat_h, '+\
                    'corresponding to weights described in Reshef et al. 2017.')
    parser.add_argument('--seed', default=None, type=int,
            help='Seed random number generator to a certain value. Off by default.')
    parser.add_argument('--stat', default='sum',
            help='*experimental* Which statistic to use for hypothesis testing. Options '+\
                    'are: sum, medrank, or thresh.')
    parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int,
            help='Space-delimited list of chromosomes to analyze. Default is 1..22')

    # configurable arguments
    parser.add_argument('--config', default=None,
            help='Path to a json file with values for other parameters. ' +\
                    'Values in this file will be overridden by any values passed ' +\
                    'explicitly via the command line.')
    parser.add_argument('--svd-stem', default=None,
            help='Path to directory in which output files will be stored. ' +\
                    'If not supplied, will be read from config file.')
    parser.add_argument('--bfile-chr', default=None,
            help='Path to plink bfile of reference panel to use, not including ' +\
                    'chromosome number. If not supplied, will be read from config file.')
    parser.add_argument('--ld-blocks', default=None,
            help='Path to UCSC bed file containing one bed interval per LD block. If '+\
                    'not supplied, will be read from config file.')


    print('=====')
    print(' '.join(sys.argv))
    print('=====')
    args = parser.parse_args()
    config.add_default_params(args)
    pretty.print_namespace(args)
    print('=====')

    main(args)
