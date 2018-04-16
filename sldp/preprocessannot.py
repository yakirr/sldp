from __future__ import print_function, division

def main(args):
    print('initializing...')
    import gzip, gc, time
    import numpy as np
    import pandas as pd
    import gprim.annotation as ga
    import gprim.dataset as gd
    import ypy.memo as memo

    # basic initialization
    mhc_bp = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]

    # read in ld blocks, remove MHC, read SNPs to print
    ldblocks = pd.read_csv(args.ld_blocks, delim_whitespace=True, header=None,
            names=['chr','start', 'end'])
    mhcblocks = (ldblocks.chr == 'chr6') & \
            (ldblocks.end > mhc_bp[0]) & \
            (ldblocks.start < mhc_bp[1])
    ldblocks = ldblocks[~mhcblocks]
    print(len(ldblocks), 'loci after removing MHC')
    print_snps = pd.read_csv(args.print_snps, header=None, names=['SNP'])
    print_snps['printsnp'] = True
    print(len(print_snps), 'print snps')

    # process annotations
    for annot in annots:
        t0 = time.time()
        for c in args.chroms:
            print(time.time()-t0, ': loading chr', c, 'of', args.chroms)
            # get refpanel snp metadata for this chromosome
            snps = refpanel.bim_df(c)
            snps = ga.smart_merge(snps, refpanel.frq_df(c)[['SNP','MAF']])
            print(len(snps), 'snps in refpanel',
                    len(snps.columns), 'columns, including metadata')

            # read in annotation
            print('reading annot', annot.filestem())
            names = annot.names(c) # names of annotations
            namesR = [n+'.R' for n in names] # names of results
            a = annot.sannot_df(c)
            if 'SNP' in a.columns:
                print('not a thinannot => doing full reconciliation of snps and allele coding')
                snps = ga.reconciled_to(snps, a, names, missing_val=0)
            else:
                print('detected thinannot, so assuming that annotation is synched to refpanel')
                snps = pd.concat([snps, a[names]], axis=1)

            # add information on which snps to print
            print('merging in print_snps')
            snps = pd.merge(snps, print_snps, how='left', on='SNP')
            snps.printsnp.fillna(False, inplace=True)
            snps.printsnp.astype(bool)

            # put on per-normalized-genotype scale
            if args.alpha != -1:
                print('scaling by maf according to alpha=', args.alpha)
                snps[names] = snps[names].values*\
                        np.power(2*snps.MAF.values*(1-snps.MAF.values),
                                (1.+args.alpha)/2)[:,None]

            # make room for RV
            snps = pd.concat(
                    [snps, pd.DataFrame(np.zeros(snps[names].shape), columns=namesR)],
                    axis=1)

            # compute simple statistics about annotation
            print('computing basic statistics and writing')
            info = pd.DataFrame(
                    columns=['M', 'M_5_50', 'sqnorm', 'sqnorm_5_50', 'supp', 'supp_5_50'])
            info['name'] = names
            info.set_index('name', inplace=True)
            info['M'] = len(snps)
            info['sqnorm'] = np.linalg.norm(snps[names], axis=0)**2
            info['supp'] = np.linalg.norm(snps[names], ord=0, axis=0)
            M_5_50 = (snps.MAF >= 0.05).values
            info['M_5_50'] = M_5_50.sum()
            info['sqnorm_5_50'] = np.linalg.norm(snps.loc[M_5_50, names], axis=0)**2
            info['supp_5_50'] = np.linalg.norm(snps.loc[M_5_50, names], ord=0, axis=0)
            info.to_csv(annot.info_filename(c), sep='\t')

            # process ldblocks one by one
            for ldblock, X, meta, ind in refpanel.block_data(ldblocks, c, meta=snps):
                if meta.printsnp.sum() == 0:
                    print('no print-snps in this block')
                    continue
                print(meta.printsnp.sum(), 'print-snps')
                if (meta[names] == 0).values.all():
                    print('annotations are all 0 in this block')
                    snps.loc[ind, namesR] = 0
                else:
                    mask = meta.printsnp.values
                    V = meta[names].values
                    XV = X.dot(V)
                    snps.loc[ind[mask], namesR] = \
                            X[:,mask].T.dot(XV[:,-len(names):]) / X.shape[0]

            # write
            print('writing output')
            with gzip.open(annot.RV_filename(c), 'w') as f:
                snps.loc[snps.printsnp,['SNP','A1','A2']+names+namesR].to_csv(
                        f, index=False, sep='\t')

            del snps; memo.reset(); gc.collect()

    print('done')
