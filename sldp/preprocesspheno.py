from __future__ import print_function, division

def main(args):
    print('initializing...')
    import gzip, os, gc, time
    import numpy as np
    import pandas as pd
    import gprim.annotation as ga
    import gprim.dataset as gd
    import ypy.fs as fs
    import ypy.memo as memo
    import sldp.weights as weights

    # read in refpanel, ld blocks, and svd snps
    refpanel = gd.Dataset(args.bfile_chr)
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    print_snps = pd.read_csv(args.print_snps, header=None, names=['SNP'])
    print_snps['printsnp'] = True
    print(len(print_snps), 'svd snps')

    # read sumstats
    print('reading sumstats', args.sumstats_stem)
    ss = pd.read_csv(args.sumstats_stem+'.sumstats.gz', sep='\t')
    ss = ss[ss.Z.notnull() & ss.N.notnull()]
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    ss = pd.merge(ss, print_snps[['SNP']], on='SNP', how='inner')
    print(len(ss), 'snps typed')

    # read ld scores
    print('reading in ld scores')
    ld = pd.concat([pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in range(1,23)], axis=0)
    if args.no_M_5_50:
        M = sum([int(open(args.ldscores_chr+str(c)+'.l2.M').next())
                        for c in range(1,23)])
    else:
        M = sum([int(open(args.ldscores_chr+str(c)+'.l2.M_5_50').next())
                        for c in range(1,23)])
    print(len(ld), 'snps with ld scores')
    ssld = pd.merge(ss, ld, on='SNP', how='left')
    print(len(ssld), 'hm3 snps with sumstats after merge.')

    # estimate heritability using aggregate estimator
    def esth2g(ssld):
        meanchi2 = (ssld.Z**2).mean()
        meanNl2 = (ssld.N*ssld.L2).mean()
        sigma2g = (meanchi2 - 1)/meanNl2
        h2g = sigma2g * M
        K = M/meanNl2 # h2g = K (meanchi2 - 1)
        return h2g, sigma2g, meanchi2, K
    h2g, sigma2g, meanchi2, K = esth2g(ssld)
    h2g = max(h2g, 0.03) #0.03 is an arbitrarily chosen minimum
    print('mean chi2:', meanchi2)
    print('h2g estimated at:', h2g, 'sigma2g =', sigma2g)
    if args.set_h2g:
        print('scaling Z-scores to achieve h2g of', args.set_h2g)
        norm = meanchi2 / (1 + args.set_h2g/ K)
        print('dividing all z-scores by', np.sqrt(norm))
        ssld.Z /= np.sqrt(norm)
        h2g, sigma2g, _, _ = esth2g(ssld)
        print('h2g is now', h2g)

    # write h2g results to file
    dirname = args.sumstats_stem + '.' + args.refpanel_name
    fs.makedir(dirname)
    if 1 in args.chroms:
        print('writing info file')
        info = pd.DataFrame(); info=info.append(
                {'pheno':args.sumstats_stem.split('/')[-1],
                    'h2g':h2g,
                    'sigma2g':sigma2g,
                    'Nbar':ss.N.mean()},ignore_index=True)
        info.to_csv(dirname+'/info', sep='\t', index=False)

    # preprocess ld blocks
    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        snps = pd.merge(snps, print_snps, on='SNP', how='left')
        snps.printsnp.fillna(False, inplace=True)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # merge annot and sumstats
        print('reconciling')
        snps = ga.reconciled_to(snps, ss, ['Z'],
                othercolnames=['N'], missing_val=np.nan)
        snps['typed'] = snps.Z.notnull()
        snps['ahat'] = snps.Z / np.sqrt(snps.N)

        # initialize result dataframe
        # I = no weights
        # h = heuristic weights, using R_o
        snps['Winv_ahat_I'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat_I'] = np.nan # = R_{*o} W_o^{-1} ahat_o
        snps['Winv_ahat_h'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat_h'] = np.nan # = R_{*o} W_o^{-1} ahat_o

        # restrict to ld blocks in this chr and process them in chunks
        for ldblock, X, meta, ind in refpanel.block_data(ldblocks, c, meta=snps):
            if meta.printsnp.sum() == 0 or \
                    not os.path.exists(args.svd_stem+str(ldblock.name)+'.R.npz'):
                print('no svd snps found in this block')
                continue
            print(meta.printsnp.sum(), 'svd snps', meta.typed.sum(), 'typed snps')
            if meta.typed.sum() == 0:
                print('no typed snps found in this block')
                snps.loc[ind, [
                    'R_Winv_ahat_I',
                    'R_Winv_ahat_h'
                    ]] = 0
                continue
            R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
            R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
            N = meta[meta.typed.values].N.mean()
            meta_svd = meta[meta.printsnp.values]

            # multiply ahat by the weights
            x_I = snps.loc[ind[meta.printsnp],'Winv_ahat_I'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat_I')
            x_h = snps.loc[ind[meta.printsnp],'Winv_ahat_h'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat_h')

        print('writing processed sumstats')
        with gzip.open('{}/{}.pss.gz'.format(dirname, c), 'w') as f:
            snps.loc[snps.printsnp,['N',
                'Winv_ahat_I',
                'Winv_ahat_h'
                ]].to_csv(
                    f, index=False, sep='\t')

        del snps; memo.reset(); gc.collect()

    print('done')
