from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
import glob
import statutils.sig as sig


def aggregate(args):
    for pheno in args.phenos:
        results = pd.DataFrame(columns=[
            'pheno','annot','sum','sum_std','sum_p',
                'ss', 'ss_std',
                'prod', 'prod_std', 'prod_p',
                'norm_1','norm_2','norm_2o4','norm_4','norm_inf','norm_info1',
                'suppVo'])
        print('reading {}'.format(pheno))

        lociresults = pd.read_csv(args.folder+pheno+'.nzloci', sep='\t')

        print('pivoting')
        piv = lociresults.pivot('locus_num','annot')
        piv.fillna(value=0, inplace=True)

        for annot in lociresults.annot.unique():
            print(annot)
            _sum = piv['vTbhat',annot].sum()
            _sum_std = np.sqrt(piv['vTbhat_vars',annot].sum())
            nz = (piv['vTbhat_prod',annot] != 0)
            _prod = piv['vTbhat_prod',annot][nz].sum()
            _prod_std = np.sqrt(piv['vTbhat_varss',annot][nz].sum())
            _ss = (piv['vTbhat',annot]**2).sum()
            norm1 = np.linalg.norm(piv['normVo_1',annot], ord=1)
            norm2 = np.linalg.norm(piv['normVo_2',annot], ord=2)
            norm4 = np.linalg.norm(piv['normVo_4',annot], ord=4)
            norminf = np.nanmax(piv['normVo_inf',annot].values)

            results=results.append({
                'pheno':pheno,
                'annot':annot,
                'sum':_sum,
                'sum_std':_sum_std,
                'sum_p':st.chi2.sf((_sum/_sum_std)**2,1),
                'ss':_ss,
                'ss_std':_prod_std,
                'prod':_prod,
                'prod_std':_prod_std,
                'prod_p':st.chi2.sf((_prod/_prod_std)**2,1),
                'norm_1':norm1,
                'norm_2':norm2,
                'norm_4':norm4,
                'norm_inf':norminf,
                'norm_2o4':norm2/norm4,
                'norm_info1':norminf/norm1,
                'supp':piv['suppVo',annot][nz].sum()},
                ignore_index=True)
        outfilename = args.folder+pheno+'.meta'
        print('writing', outfilename)
        results.to_csv(outfilename, sep='\t', index=False)
        print(sig.fdr(results.sum_p))
        print(sig.fdr(results.prod_p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
            default='/groups/price/yakir/data/annot/basset/results_tads.2/')
    parser.add_argument('--phenos', nargs='+',
            help='array of phenotypes to analyze')

    args = parser.parse_args()
    aggregate(args)
