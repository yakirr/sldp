from __future__ import print_function, division
import argparse, gzip, os, gc, time
import numpy as np
import pandas as pd
import pyutils.pretty as pretty
import pyutils.fs as fs
import gprim.annotation as ga
import gprim.dataset as gd
import weights
import pyutils.memo as memo


def main(args):
    # read in refpanel, ld blocks, and svd snps
    refpanel = gd.Dataset(args.bfile_chr)
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    svd_snps = pd.read_csv(args.svd_snps, header=None, names=['SNP'])
    svd_snps['svdsnp'] = True
    print(len(svd_snps), 'svd snps')

    # read sumstats
    print('reading sumstats', args.sumstats_stem)
    ss = pd.read_csv(args.sumstats_stem+'.sumstats.gz', sep='\t')
    ss = ss[ss.Z.notnull() & ss.N.notnull()]
    print('{} snps, {}-{} individuals (avg: {})'.format(
        len(ss), np.min(ss.N), np.max(ss.N), np.mean(ss.N)))
    ss = pd.merge(ss, svd_snps[['SNP']], on='SNP', how='inner')
    print(len(ss), 'snps typed after removing MHC')

    # read ld scores
    print('reading in ld scores')
    ld = pd.concat([pd.read_csv(args.ldscores_chr+str(c)+'.l2.ldscore.gz',
                        delim_whitespace=True)
                    for c in range(1,23)], axis=0)
    M_5_50 = sum([int(open(args.ldscores_chr+str(c)+'.l2.M_5_50').next())
                    for c in range(1,23)])
    print(len(ld), 'snps with ld scores')
    ssld = pd.merge(ss, ld, on='SNP', how='left')
    print(len(ssld), 'hm3 snps with sumstats after merge. Assuming MHC snps arent typed')

    # estimate heritability using aggregate estimator
    meanchi2 = (ssld.Z**2).mean()
    meanNl2 = (ssld.N*ssld.L2).mean()
    h2g = (meanchi2 - 1)/(meanNl2/M_5_50)
    h2g = max(h2g, 0.03) #0.03 is the minimum of stephens estimates using real methods
    sigma2g = h2g / M_5_50
    print('h2g estimated at:', h2g, 'sigma2g =', sigma2g)

    # write results to file
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

    t0 = time.time()
    for c in args.chroms:
        print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
        # get refpanel snp metadata for this chromosome
        snps = refpanel.bim_df(c)
        snps = pd.merge(snps, svd_snps, on='SNP', how='left')
        snps.svdsnp.fillna(False, inplace=True)
        print(len(snps), 'snps in refpanel', len(snps.columns), 'columns, including metadata')

        # merge annot and sumstats
        print('reconciling')
        snps = ga.reconciled_to(snps, ss, ['Z'],
                othercolnames=['N'], missing_val=np.nan)
        snps['typed'] = snps.Z.notnull()
        snps['ahat'] = snps.Z / np.sqrt(snps.N)
        # hlN = heuristic weights with large N approximation
        # h = heuristic weights, using R_o
        # h2 = heuristic weights, using (R2)_o
        # no suffix = exact weights
        snps['Winv_ahat_hlN'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat_hlN'] = np.nan # = R_{*o} W_o^{-1} ahat_o
        snps['Winv_ahat_h'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat_h'] = np.nan # = R_{*o} W_o^{-1} ahat_o
        snps['Winv_ahat_h2'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat_h2'] = np.nan # = R_{*o} W_o^{-1} ahat_o
        snps['Winv_ahat'] = np.nan # = W_o^{-1} ahat_o
        snps['R_Winv_ahat'] = np.nan # = R_{*o} W_o^{-1} ahat_o

        # restrict to ld blocks in this chr and process them in chunks
        for ldblock, X, meta, ind in refpanel.block_data(ldblocks, c, meta=snps):
            if meta.svdsnp.sum() == 0 or \
                    not os.path.exists(args.svd_stem+str(ldblock.name)+'.R.npz'):
                print('no svd snps found in this block')
                continue
            print(meta.svdsnp.sum(), 'svd snps', meta.typed.sum(), 'typed snps')
            if meta.typed.sum() == 0:
                print('no typed snps found in this block')
                snps.loc[ind, ['R_Winv_ahat_hlN','R_Winv_ahat_h','R_Winv_ahat']] = 0
                continue
            R = np.load(args.svd_stem+str(ldblock.name)+'.R.npz')
            R2 = np.load(args.svd_stem+str(ldblock.name)+'.R2.npz')
            N = meta[meta.typed].N.mean()
            meta_svd = meta[meta.svdsnp]

            # multiply ahat by the weights
            x_hlN = snps.loc[ind[meta.svdsnp],'Winv_ahat_hlN'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat_hlN')
            x_h = snps.loc[ind[meta.svdsnp],'Winv_ahat_h'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat_h')
            x_h2 = snps.loc[ind[meta.svdsnp],'Winv_ahat_h2'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat_h2')
            x = snps.loc[ind[meta.svdsnp],'Winv_ahat'] = weights.invert_weights(
                    R, R2, sigma2g, N, meta_svd.ahat.values, mode='Winv_ahat')
            print(np.corrcoef([x_hlN[meta_svd.typed],
                x_h[meta_svd.typed],
                x_h2[meta_svd.typed],
                x[meta_svd.typed]]))

            # multiply the results by R
            X_typed = X[:,meta.typed]
            snps.loc[ind, 'R_Winv_ahat_hlN'] = X.T.dot(X_typed.dot(
                x_hlN[meta_svd.typed]))/X.shape[0]
            snps.loc[ind, 'R_Winv_ahat_h'] = X.T.dot(X_typed.dot(
                x_h[meta_svd.typed]))/X.shape[0]
            snps.loc[ind, 'R_Winv_ahat_h2'] = X.T.dot(X_typed.dot(
                x_h2[meta_svd.typed]))/X.shape[0]
            snps.loc[ind, 'R_Winv_ahat'] = X.T.dot(X_typed.dot(
                x[meta_svd.typed]))/X.shape[0]

        print('writing finemapped sumstats')
        with gzip.open('{}/{}.ss.sf.gz'.format(dirname, c), 'w') as f:
            snps[['N','R_Winv_ahat_hlN','R_Winv_ahat_h',
                'R_Winv_ahat_h2','R_Winv_ahat']].to_csv(
                    f, index=False, sep='\t')
        with gzip.open('{}/{}.ss.jk.gz'.format(dirname, c), 'w') as f:
            snps.loc[snps.svdsnp,['N','Winv_ahat_hlN','Winv_ahat_h',
                'Winv_ahat_h2','Winv_ahat']].to_csv(
                    f, index=False, sep='\t')

        del snps; memo.reset(); gc.collect()

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumstats-stem', #required=True,
            default='/groups/price/yakir/data/sumstats.hm3/processed/UKBiobank_Height3',
            help='path to sumstats.gz files, not including ".sumstats.gz" extension')

    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num.')
    parser.add_argument('--svd-stem',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/svds_95percent/',
            help='path to truncated svds of reference panel, by LD block')
    parser.add_argument('--svd-snps',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/'+\
                    '1000G_hm3_noMHC.rsid',
            help='The set of snps for which the svds are computed')
    parser.add_argument('--ld-blocks',
            default='/groups/price/yakir/data/reference/pickrell_ldblocks.hg19.eur.bed',
            help='path to UCSC bed file containing one bed interval per LD' + \
                    ' block')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/LDscore/LDscore.',
            help='path to LD scores at a smallish set of SNPs. LD should be computed '+\
                    'to all potentially causal snps. This is used for '+\
                    'heritability estimation')
    parser.add_argument('--refpanel-name', default='KG3.95',
            help='suffix added to the directory created for storing output')
    parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int)

    args = parser.parse_args()
    pretty.print_namespace(args)
    main(args)
