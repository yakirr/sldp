from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
import glob
import statutils.sig as sig


def compare_dists(dist, nulldist):
    # ds = np.sort(dist, ascending=False)
    # nds = np.sort(nulldist, ascending=False)
    p = np.zeros(dist.shape)
    for i, x in enumerate(dist):
        p[i] = (nulldist >= x).sum() / len(nulldist)
    return p


def aggregate(args):
    for pheno in args.phenos:
        results = pd.DataFrame(columns=[
            'pheno','annot','sum','sum_std','sum_p','rsum','rsum_p',
                'ss', 'rss', 'sprod', 'rsprod'])
        print('reading {}'.format(pheno))

        lociresults = pd.read_csv(args.folder+pheno+'.nzloci', sep='\t')

        print('pivoting')
        VTbhat = lociresults.pivot('locus_num','annot','vTbhat')
        VTbhat_stds = lociresults.pivot('locus_num','annot','vTbhat_stds')
        VTbhat_ss = lociresults.pivot('locus_num','annot','vTbhat_ss')
        VrTbhat = lociresults.pivot('locus_num','annot','vrTbhat')

        for annot in lociresults.annot.unique():
            print(annot)
            vTbhat = np.nan_to_num(VTbhat[annot].values)
            stds = np.nan_to_num(VTbhat_stds[annot].values)
            vrTbhat = np.nan_to_num(VrTbhat[annot].values)
            vTbhat_ss = np.nan_to_num(VTbhat_ss[annot].values)

            results=results.append({
                'pheno':pheno,
                'annot':annot,
                'sum':vTbhat.sum(),
                'sum_std':np.sqrt(stds.dot(stds)),
                'sum_p':st.chi2.sf(vTbhat.sum()**2/stds.dot(stds),1),
                'rsum':vrTbhat.sum(),
                'rsum_p':st.chi2.sf(vrTbhat.sum()**2/stds.dot(stds),1),
                'ss':vTbhat.dot(vTbhat),
                'rss':vrTbhat.dot(vrTbhat),
                'sprod':vTbhat.dot(vTbhat) - vTbhat_ss.sum(),
                'rsprod':vrTbhat.dot(vrTbhat) - vTbhat_ss.sum()},
                ignore_index=True)
        outfilename = args.folder+pheno+'.meta'
        print('writing', outfilename)
        results.to_csv(outfilename, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
            default='/groups/price/yakir/data/annot/basset/results_tads/')
    parser.add_argument('--phenos', nargs='+',
            help='array of phenotypes to analyze')

    args = parser.parse_args()
    aggregate(args)
