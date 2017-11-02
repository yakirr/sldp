from __future__ import print_function, division
import numpy as np
import pandas as pd
import gprim.annotation as ga
import gprim.dataset as gd
import ypy.fs as fs
import matplotlib.pyplot as plt

# find windows with genome-wide significant SNPs that are consistent with the global signal
def write(folder, args, name, background_names, mux, muy, z, corr_thresh=0.8):
    print('STORYTELLING for ', name, 'z=', z)
    refpanel = gd.Dataset(args.bfile_reg_chr)
    annot = [ga.Annotation(a) for a in args.sannot_chr
            if name in ga.Annotation(a).names(22, RV=True)][0]

    backgroundannots = [ga.Annotation(a) for a in args.background_sannot_chr]
    print('focal annotation columns:', annot.names(22, True))
    print('background annotations:', background_names)

    # get refpanel snp metadata 
    print('re-reading snps')
    snps = pd.concat([refpanel.bim_df(c) for c in args.chroms], axis=0)

    # read sumstats
    print('re-reading sumstats')
    ss = pd.concat([
        pd.read_csv(args.pss_chr+str(c)+'.pss.gz', sep='\t', usecols=['N','Winv_ahat_I'])
        for c in args.chroms])
    snps['ahat'] = ss['Winv_ahat_I']
    snps['N'] = ss['N']
    del ss

    # read annotations
    print('re-reading background annotations')
    for a in backgroundannots:
        mynames = [n for n in a.names(22, RV=True) if '.R' in n] #names of annotations
        snps = pd.concat([snps,
            pd.concat([a.RV_df(c)[mynames] for c in args.chroms], axis=0)
            ], axis=1)

    print('reading focal annotation')
    snps = pd.concat([snps,
        pd.concat([annot.RV_df(c)[name] for c in args.chroms], axis=0)
        ], axis=1)

    print('residualizing background out of focal')
    A = snps[background_names]
    snps['chi2'] = snps.N * snps.ahat**2
    snps['Rv'] = snps[name]
    snps['ahat_resid'] = snps.ahat - A.values.dot(muy)
    snps['Rv_resid'] = snps.Rv - A.values.dot(mux)
    snps['typed'] = snps.ahat_resid.notnull()
    snps = snps[snps.typed].reset_index(drop=True)
    snps['significant'] = snps.chi2 > 29.716785
    print(snps.significant.sum(), 'genome-wide significant SNPs')

    print('searching for good windows')
    # get endpoints of windows
    stride = 20
    windowsize_in_strides = 5
    windowsize = stride * windowsize_in_strides

    # find all starting points of windows containing GWAS-sig SNPs
    starts = np.concatenate([
                [ int(i/stride)*stride - k*stride
                    for k in range(0, windowsize_in_strides)]
                for i in np.where(snps.significant)[0]])
    starts = np.array(sorted(list(set(starts))))
    # compute corresponding endpoints
    ends = starts + windowsize

    # truncate any windows that extend past the ends of the genome
    starts = starts[ends < len(snps)]
    ends = ends[ends < len(snps)]
    ends = ends[starts >= 0]
    starts = starts[starts >= 0]

    print(len(starts), 'windows with GWAS hits found')

    # compute correlations
    numbers = pd.DataFrame(
            np.array([[i,j,
                np.max(snps.iloc[i:j].chi2),
                np.corrcoef(snps.iloc[i:j].Rv_resid, snps.iloc[i:j].ahat_resid)[0,1]]
                for i,j in zip(starts, ends)]),
            columns=['start','end','maxchi2','corr'])

    # keep only cases with strong correlations in the right direction
    numbers = numbers[numbers['corr']**2 >= corr_thresh]
    numbers = numbers[np.sign(numbers['corr']) == np.sign(z)]
    print(len(numbers), 'windows with GWAS hits and squared correlation with Rv >=',
            corr_thresh, 'in the right direction')
    for i,j in zip(numbers.start, numbers.end):
        i = int(i); j = int(j)
        print('saving', i,j)
        c = snps.iloc[i].CHR
        start = snps.iloc[i].BP
        end = snps.iloc[j].BP
        plt.figure()
        plt.scatter(snps.iloc[i:j].Rv_resid,
                snps.iloc[i:j].ahat_resid * np.sqrt(snps.iloc[i:j].N))
        plt.title('chr{}:{}-{}'.format(c, start, end))
        plt.xlabel(r'residual $Rv$')
        plt.ylabel(r'residual $Z$')

        filename = '{}/chr{}:{}-{}.pdf'.format(folder, c, start, end)
        fs.makedir_for_file(filename)
        plt.savefig(filename)
        plt.close()
