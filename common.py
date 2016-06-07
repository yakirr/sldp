from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import gzip
from pybedtools import BedTool
from pyutils import iter as it
import itertools
import primitives.dataset as prd
import primitives.genome as prg
import primitives.annotation as pa


def get_ldscores(ldscoresfile):
    print('reading ldscores')
    return pd.read_csv(ldscoresfile, header=0, sep='\t', compression='gzip')

def get_ldscores_allchr(ldscoresfile_chr, chromosomes):
    print('reading ldscores')
    ldscores = []
    for chrnum in chromosomes:
        print('\t', chrnum)
        ldscores.append(get_ldscores(ldscoresfile_chr + str(chrnum) + '.l2.ldscore.gz'))
    return pd.concat(ldscores).reset_index(drop=True)

def get_sumstats(sumstatsfile, thresh_factor):
    print('reading sumstats and filtering by sample size')
    sumstats = pd.read_csv(sumstatsfile, header=0, sep='\t',
            compression='gzip')
    print('\tremoving', np.sum(np.isnan(sumstats['N'])), 'snps with nans')
    sumstats = sumstats.loc[~np.isnan(sumstats['N'])]
    N90 = np.percentile(sumstats['N'], 90)
    Nthresh = thresh_factor * N90
    print('\tthreshold sample size:', Nthresh)
    print('\toriginally at', len(sumstats), 'SNPs')
    sumstats = sumstats.loc[
            sumstats['N'] <= Nthresh]
    print('\tafter filtering by N, now at', len(sumstats), 'SNPs')
    print('\tmin, mean (std), max of N is:',
            '{}, {} ({}), {}'.format(
                np.min(sumstats.N),
                np.mean(sumstats.N),
                np.std(sumstats.N),
                np.max(sumstats.N)))
    return sumstats

def get_annot(annotfilename):
    annot = pd.read_csv(annotfilename, header=0, sep='\t', compression='gzip')
    return annot, annot.columns[6:].values.tolist()

def get_annots(annotfilenames):
    print('reading annot files', annotfilenames)
    annot, annot_names = get_annot(annotfilenames[0])
    for annotfilename in annotfilenames[1:]:
        newannot, newannot_names = get_annot(annotfilename)
        toflip = annot['A1'] == newannot['A2']
        if np.sum(toflip) > 0:
            raise Exception('all annotations must have the same allele coding')
        annot[newannot_names] = newannot[newannot_names]
        annot_names += newannot_names

    print('\tannotation names:', annot_names)
    print('\tannotation contains', len(annot), 'SNPs')
    print('\tannotation supported on',
            np.sum(annot[annot_names].values != 0, axis=0), 'SNPs')
    print('\tsquared norm of annotation is',
            np.linalg.norm(annot[annot_names].values, axis=0)**2)
    return annot, annot_names

# TODO: in the two functions below, it needs to be the case that singletons contribute
# a single 1 along the diagonal of R. As currently implemented, they contribute 0
# NB: in real life, the refpanel won't have singletons so this may not be an issue
def mult_by_R_ldblocks(V, (refpanel, chrnum), ld_breakpoints, mhcpath):
    print('\tloading ld breakpoints and MHC')
    breakpoints = BedTool(ld_breakpoints)
    mhc = BedTool(mhcpath)
    print('\tconstructing SNP partition')
    blocks = prg.SnpPartition(refpanel.ucscbed(chrnum), breakpoints, mhc)

    print('\tdoing multiplication')
    result = np.zeros(V.shape)
    for r in it.show_progress(blocks.ranges()):
        # print('\tXTXV', r[0], r[1], 'of', refpanel.M(chrnum))
        X = refpanel.stdX(chrnum, r)
        result[r[0]:r[1],:] = X.T.dot(X.dot(V[r[0]:r[1],:]))
    return result / refpanel.N()

def get_biascorrection(V, RV, Lambda, (refpanel, chrnum), breakpoints, mhcpath):
    print('\tloading ld breakpoints and MHC')
    breakpoints = BedTool(breakpoints)
    mhc = BedTool(mhcpath)
    print('\tconstructing SNP partition')
    blocks = prg.SnpPartition(refpanel.ucscbed(chrnum), breakpoints, mhc)

    print('\tcomputing bias term')
    result = 0
    for r in blocks.ranges():
        result += V[r[0]:r[1]].T.dot(RV[r[0]:r[1]]) * \
                np.sum(Lambda[r[0]:r[1]]) / refpanel.N()
    return result

def mult_by_R_noldblocks(V, (refpanel, chrnum)):
    r = 0
    XV = 0
    while r < refpanel.M(chrnum):
        s = (r, min(r+1000, refpanel.M(chrnum)))
        print(s, 'of', refpanel.M(chrnum))
        X = refpanel.stdX(chrnum, s)
        XV += X.dot(V[s[0]:s[1]])
        r += 1000
    r = 0
    XTXV = np.zeros(V.shape)
    while r < refpanel.M(chrnum):
        s = (r, min(r+1000, refpanel.M(chrnum)))
        print(s, 'of', refpanel.M(chrnum))
        X = refpanel.stdX(chrnum, s)
        XTXV[s[0]:s[1]] = X.T.dot(XV)
        r += 1000
    return XTXV / refpanel.N()

def sparse_QF(v1, v2, (refpanel, chrnum)):
    v1 = v1.reshape((-1,))
    v2 = v2.reshape((-1,))
    nz1 = np.nonzero(v1)[0]
    nz2 = np.nonzero(v2)[0]
    ind = np.sort(np.unique(np.concatenate([nz1, nz2])))

    r = 0
    Xv1 = 0
    Xv2 = 0
    while r < len(ind):
        s = (r, min(r+500, len(ind)))
        print(s, 'of', len(ind))
        X = refpanel.stdX_it(chrnum, ind[s[0]:s[1]])
        print('done reading')
        Xv1 += X.dot(v1[ind[s[0]:s[1]]])
        Xv2 += X.dot(v2[ind[s[0]:s[1]]])
        r += 500
    return Xv1.dot(Xv2)/refpanel.N()

def convolve(df, cols_to_convolve, (refpanel, chrnum), ld_breakpoints, mhcpath,
        fullconv=False, newnames=None):
    print('\trefpanel contains', refpanel.M(chrnum), 'SNPs')
    print('\tmerging df and refpanel')
    if len(df) != len(refpanel.bim_df(chrnum)):
        print('df SNPs and refpanel SNPs must be the same')
        raise Exception()
    refwithdf = refpanel.bim_df(chrnum).merge(df, how='left', on=['SNP'])

    print('\tconvolving')
    if fullconv:
        RV = mult_by_R_noldblocks(refwithdf[cols_to_convolve].values, (refpanel, chrnum))
    else:
        RV = mult_by_R_ldblocks(refwithdf[cols_to_convolve].values, (refpanel, chrnum),
                ld_breakpoints, mhcpath)

    if newnames is None:
        newnames = [n+'.conv1' for n in cols_to_convolve]
    for i, n1 in enumerate(newnames):
        df[n1] = RV[:,i]

    return df, newnames

def get_conv(convfilename, annotfilename, (refpanel, chrnum), ld_breakpoints, mhcpath):
    if not os.path.exists(convfilename):
        print('\tconv file', convfilename, 'not found. creating...')

        print('\t\treading annot', annotfilename)
        annot, annot_names = get_annot(annotfilename)

        convolved, newnames = convolve(annot, annot_names, (refpanel, chrnum),
                ld_breakpoints, mhcpath, fullconv=pa.Annotation.isfullconv(convfilename))

        def save(convolved, newnames, out):
            print('\t\tsaving')
            if out[-3:] != '.gz':
                print('\t\toutfile must end in .gz. file not saved.')
                return
            with gzip.open(out, 'w') as f:
                convolved[['CHR','BP','SNP','CM','A1','A2'] + newnames].to_csv(
                        f, index=False, sep='\t')
        save(convolved, newnames, convfilename)

    conv = pd.read_csv(convfilename, header=0, sep='\t', compression='gzip')
    return conv, conv.columns.values[6:]

def get_convs(convfilenames, annotfilenames, (refpanel, chrnum), ld_breakpoints,
        mhcpath):
    print('loading conv files', convfilenames)
    if len(convfilenames) != len(annotfilenames):
        raise Exception('\tERROR: the list of annot files and conv files must match')

    conv, conv_names = get_conv(convfilenames[0], annotfilenames[0],
            (refpanel, chrnum), ld_breakpoints, mhcpath)
    for convfilename, annotfilename in zip(convfilenames[1:], annotfilenames[1:]):
        newconv, newconv_names = get_conv(convfilename, annotfilename,
                (refpanel, chrnum), ld_breakpoints, mhcpath)
        toflip = conv['A1'] == newconv['A2']
        if np.sum(toflip) > 0:
            raise Exception('all conv files must have the same allele coding')
        conv[newconv_names] = newconv[newconv_names]
        conv_names += newconv_names

    return conv, conv_names

def get_ambiguous(alleles, A1name, A2name):
    allele = [''.join((a1, a2)) for (a1, a2) in zip(alleles[A1name], alleles[A2name])]
    COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    BASES = COMPLEMENT.keys()
    STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
        for x in itertools.product(BASES, BASES)
        if x[0] != x[1]}
    return np.array([STRAND_AMBIGUOUS[x] for x in allele])

def check_for_strand_ambiguity(data, names, A1name='A1_x', A2name='A2_x'):
    nonzero = np.empty(len(data))
    nonzero = False
    for n in names:
        nonzero |= (data[n] != 0)

    print('\ttotal number of snps with non-zero annotation:', np.sum(nonzero))

    amb = get_ambiguous(data.loc[nonzero], A1name, A2name)
    if np.sum(amb) > 0:
        print('\tannotation contains strand-ambiguous snps! aborting.')
        print(data.loc[nonzero].SNP[amb])
        exit(1)
    else:
        print('\tno strand-ambiguous snps found')
