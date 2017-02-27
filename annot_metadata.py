from __future__ import print_function, division
import argparse, os, gc, time, resource
import numpy as np
import scipy.stats as st
import pandas as pd
import itertools as it
from pybedtools import BedTool
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import pyutils.iter as pyit
import gprim.annotation as ga; reload(ga)
import gprim.dataset as gd
import pyutils.memo as memo


def mem():
    print('memory usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, 'Mb')


def get_locus_inds(snps, loci, chroms):
    loci.sort_values(by='CHR', inplace=True)
    locusstarts = np.array([])
    locusends = np.array([])
    for c in chroms:
        chrsnps = snps[snps.CHR == c]
        chrloci = loci[loci.CHR == 'chr'+str(c)]
        offset = np.where(snps.CHR.values == c)[0][0]
        locusstarts = np.append(locusstarts,
                offset + np.searchsorted(chrsnps.BP.values, chrloci.start.values))
        locusends = np.append(locusends,
                offset + np.searchsorted(chrsnps.BP.values, chrloci.end.values))
    return zip(locusstarts.astype(int), locusends.astype(int))


def main(args):
    # basic initialization
    annots = [ga.Annotation(annot) for annot in args.sannot_chr]
    target = ga.Annotation(args.corr_to)
    baseline_names = target.names(22)
    print('baseline annotations:', baseline_names)

    # read in baseline annotation
    baseline = pd.concat([target.sannot_df(c) for c in args.chroms], axis=0)
    B = baseline[baseline_names].values

    results = pd.DataFrame()
    t0 = time.time()
    for annot in annots:
        names = annot.names(22) # names of annotations
        print(time.time()-t0, ': reading annot', annot.filestem())
        a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)

        print('creating V')
        print('there are', (~np.isfinite(a[names].values)).sum(), 'nans in the annotation.',
                'This number should be 0.')
        V = a[names].values

        for i, name in enumerate(names):
            print(i, name)

            v = V[:,i]
            X = np.concatenate([v.reshape((-1,1)),B], axis=1)
            corrs = np.corrcoef(X.T)[0,1:]
            result = {'annot':name}
            result.update({
                    n:c for n,c in zip(baseline_names, corrs)})
            result.update({'size':(v != 0).sum()})

            results = results.append(result,
                ignore_index=True)
            print(results.iloc[-1])
            del v

        del V; memo.reset(); gc.collect(); mem()

    print('writing results')
    results.to_csv(args.outfile, sep='\t', index=False)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/processed.a8/8988T/',
                '/groups/price/yakir/data/annot/basset/processed.a8/A549/'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--corr-to', #required=True,
            default='/groups/price/yakir/data/annot/baseline/maf/')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
