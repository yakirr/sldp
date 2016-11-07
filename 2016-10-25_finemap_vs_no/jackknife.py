from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
import glob
import statutils.sig as sig


def spearman_stat(df):
    val, pval = st.spearmanr(df.voTRv.values, df.vTahat.values)
    return val, -1, pval

def spearman_sflip(df):
    def sp(x,y,w):
        x = np.argsort(np.argsort(x))
        y = np.argsort(np.argsort(y))
        xc = x-np.mean(x)
        yc = y-np.mean(y)
        xcw = xc*w
        ycw = yc*w
        return xc.dot(ycw)/np.sqrt(xc.dot(xcw)*yc.dot(ycw))

    x = df.vTahat.values; y = df.voTRv.values; w = df.weights.values
    nz = (w != 0)
    x = x[nz]; y = y[nz]; w = w[nz]
    val = sp(x, y, w)
    rand_signs = (-1)**np.random.binomial(1,0.5,size=(1000,len(y)))
    null = [sp(x, rand_signs[i]*y, w) for i in range(1000)]
    std = np.std(null)
    pval = st.chi2.sf((val/std)**2, 1)
    return val, std, pval

def lin_reg(df):
    y = df.vTahat.values
    x = df.voTRv.values
    w = df.weights.values
    w[w<0] = 0
    conc = np.sort(w)[-40:].sum() / w.sum()
    numbig = (x >= 0.5).sum()
    quant = np.sort(w)[-40]
    nz = (w!=0)&(x!=0)
    x = x[nz]; y = y[nz]; w = w[nz]

    # adjust weights to reduce influence of outliers
    w /= w.sum()
    # TODO: make 10 biggest weights of w equal to the 10-th biggest one?
    w[w>0.05] = 0.05
    # try:
    #     w[w > 0.05] = w[w<=0.05].max()
    # except:
    #     print('exception')
    #     w[w > 0.05] = 0.05

    x *= (np.sqrt(w)/x); y *= (np.sqrt(w)/x)
    h = (x*x)/x.dot(x)

    val = x.dot(y)/x.dot(x)
    resid2 = (y - val*x)**2 / (1-h)**2
    std = np.sqrt(x.dot(x*resid2)) / x.dot(x)
    pval = st.chi2.sf((val/std)**2, 1)
    return val, std, pval, conc, numbig, quant

def correlation_stat(df):
    return np.corrcoef(df.voTRv.values, df.vTahat.values)[0,1]

def rand_sign(df):
    z = np.nan_to_num(df.vTahat.values / df.voTRv.values)
    numerator = z.dot(df.weights.values)
    denominator = df.weights.sum()

    val = numerator/denominator
    std = np.sqrt((z**2).dot(df.weights.values**2) / df.weights.sum()**2)
    pval = st.chi2.sf((val/std)**2, 1)
    return val, std, pval

def generic_jackknife(df, stat):
    loo = np.zeros(len(df))
    for i in range(len(loo)):
        loo[i] = stat(df[np.arange(len(loo))!=i])
    val = stat(df)
    std = np.sqrt((len(loo)-1)*np.var(loo))
    pval = st.chi2.sf((val/std)**2, 1)
    return val, std, pval

# NOTE: I think it doesn't make sense to use the function below,
# because it assumes the jackknife weights equals the regression weights, which
# doesn't quite seem right
def weighted_jackknife(df):
    z = np.nan_to_num(df.vTahat.values / df.voTRv.values)
    m = df.weights.values
    nz = (df.weights != 0)
    z = z[nz]; m=m[nz]; g = len(z)
    n = m.sum()
    numerator = z.dot(m)
    denominator = m.sum()
    phi = numerator/denominator
    numerators = numerator - z*m
    denominators = denominator - m
    loo = numerators / denominators
    phiJ = g*phi - ((n-m)/n).dot(loo)
    h = n / m
    tau = h*phi - (h-1)*loo
    var = ((tau - phiJ)**2/(h-1)).sum() / g
    std = np.sqrt(var)
    pval = st.chi2.sf((phi/std)**2, 1)
    return phi, std, pval

def jackknife(df):
    z = df.vTahat.values / df.voTRv.values
    w = df.weights.values
    nz = ((df.weights != 0) & np.isfinite(z)).values
    z = z[nz]; w=w[nz]; g = len(z)
    numerator = z.dot(w)
    denominator = w.sum()

    phi = numerator/denominator
    numerators = numerator - z*w
    denominators = denominator - w
    loo = numerators / denominators

    std = np.sqrt((g-1)*np.var(loo))
    pval = st.chi2.sf((phi/std)**2, 1)
    return phi, std, pval

def get_weights(df, scheme):
    if scheme=='const': # constant weights
        return np.ones(len(df))
    elif scheme=='constnz':
        result = np.ones(len(df))
        result[df.voTRv.values == 0] = 0
        return result
    elif scheme=='top':
        result = np.ones(len(df))
        thresh = np.sort(df.voTRv.values)[-30]
        result[result < thresh] = 0
        return result
    elif scheme=='sqrt':
        return np.nan_to_num(np.sqrt(df.voTRv.values))
    elif scheme=='fixed': # weights resulting in fixed-beta statistic
        return np.nan_to_num(df.voTRv.values)
    elif scheme=='rank':
        result = np.argsort(np.argsort(df.voTRv.values))
        result[df.voTRv.values == 0] = 0
        return result
    elif scheme=='sqrtrank':
        result = np.sqrt(np.argsort(np.argsort(df.voTRv.values)))
        result[df.voTRv.values == 0] = 0
        return result
    elif scheme=='naive': # naively regress vTahat against voTRv
        return np.nan_to_num(df.voTRv.values**2)
    elif scheme=='cubic': # extreme up-weighting of points with large vTRv
        return np.nan_to_num(df.voTRv.values**3)
    elif scheme=='xtreme':
        return np.nan_to_num(df.voTRv.values**100)
    elif scheme=='optimal': # theoretically optimal random beta weights
        return np.nan_to_num(df.voTRv.values**2 / \
            (df.voTRvo_N.values + df.sigma2g.values*df.voTR2vo.values))
    elif scheme=='simple_random': # simplest weights that incl. both v and ld to other stuff
        return np.nan_to_num(df.voTRv.values**2 / df.voTR2vo.values)

def set_weights(df, scheme):
    df.loc[:,'weights'] = get_weights(df, scheme)

def aggregate(args):
    results = pd.DataFrame(columns=['pheno','annot','stat','std','p'])
    for filename in args.files:
        print('aggregating {} using statistic "{}" and weights "{}" with Rinv="{}"'.format(
            filename, args.statistic, args.weight_scheme, args.Rinv))

        blockresults = pd.read_csv(filename, sep='\t')
        if args.Rinv:
            blockresults.voTRv = blockresults.normVo
            blockresults.vTahat = blockresults.vfTahat
            blockresults.voTRvo_N = blockresults.vfTRvf_N
            blockresults.voTR2vo = blockresults.normVo

        set_weights(blockresults, scheme=args.weight_scheme)

        annots = blockresults.annot.unique()
        phenos = blockresults.pheno.unique()
        print('computing pvals')
        for p, a in it.product(phenos, annots):
            # print(len(results), p, a)
            myresults = blockresults[(blockresults.annot == a)&(blockresults.pheno == p)]
            conc = numbig = quant = 0
            if args.statistic=='linear':
                val, std, pval = jackknife(myresults)
            elif args.statistic=='spearman':
                val, std, pval = spearman_stat(myresults)
            elif args.statistic=='spearman_sflip':
                val, std, pval = spearman_sflip(myresults)
            elif args.statistic=='correlation':
                val, std, pval = generic_jackknife(myresults, correlation_stat)
            elif args.statistic=='rand_sign':
                val, std, pval = rand_sign(myresults)
            elif args.statistic=='lin_reg':
                val, std, pval, conc, numbig, quant = lin_reg(myresults)
            # print(val, std, pval, conc, numbig, quant)
            results = results.append({
                'pheno':p,
                'annot':a,
                'stat':val,
                'std':std,
                'p':pval,
                'conc':conc,
                'numbig':numbig,
                'quant':quant}, ignore_index=True)
        print(sig.fdr(results.p))
    outfilename = '{}.{}.{}'.format(
        args.outfile_stem, args.statistic, args.weight_scheme)
    if args.Rinv:
        outfilename = outfilename + '.Rinv'
    print('writing', outfilename)
    results.to_csv(outfilename, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+',
            help='filename pattern describing a bunch of files in *.blocks format')
    parser.add_argument('--outfile-stem', required=True,
            help='path to output file, stat type and weight scheme will be appended to name')
    parser.add_argument('--statistic', default='linear',
            help='the type of statistic to use for aggregating across loci')
    parser.add_argument('--weight-scheme', default='fixed',
            help='the weighting scheme to use for the statistic')
    parser.add_argument('-Rinv', default=False, action='store_true',
            help='whether to use the fine-mapped statistics')

    args = parser.parse_args()
    aggregate(args)
