from __future__ import print_function, division
import argparse, time, gc
import numpy as np
import scipy.stats as st
import pandas as pd
import pyutils.pretty as pretty
import pyutils.fs as fs
import pyutils.memo as memo
import gprim.annotation as ga
import gprim.dataset as gd

def get_locus_inds(snps, loci, chroms):
    loci.sort_values(by=['CHR','start'], inplace=True)
    loci['startind'] = 0
    loci['endind'] = 0
    for c in chroms:
        chrsnps = snps[snps.CHR == c]
        offset = np.where(snps.CHR.values == c)[0][0]

        chrloci = loci[loci.CHR == 'chr'+str(c)]
        locusstarts = offset + np.searchsorted(chrsnps.BP.values, chrloci.start.values)
        locusends = offset + np.searchsorted(chrsnps.BP.values, chrloci.end.values)
        loci.loc[loci.CHR == 'chr'+str(c), 'startind'] = locusstarts
        loci.loc[loci.CHR == 'chr'+str(c), 'endind'] = locusends
    return loci

def main(args):
    global results
    # basic initialization
    fs.mem()
    mhc = [25684587, 35455756]
    refpanel = gd.Dataset(args.bfile_chr)
    nice_ss_name = args.bhat_chr.split('/')[-2].split('.')[0]

    annots = [ga.Annotation(annot) for annot in args.sannot_chr]

    # read in ldblocks and remove ones that overlap mhc
    loci = pd.read_csv(args.loci, delim_whitespace=True, header=None,
            names=['CHR','start', 'end'])
    mhcblocks = (loci.CHR == 'chr6') & (loci.end > mhc[0]) & (loci.start < mhc[1])
    loci = loci[~mhcblocks]
    print('read in', len(loci), 'loci that dont overlap mhc')

    # read in sumstats and annots, and get to work
    print('reading sumstats'); fs.mem()
    t0 = time.time()
    ss = pd.concat([pd.read_csv(args.bhat_chr+str(c)+'.bhat.gz', sep='\t')
        for c in args.chroms], axis=0)
    print(len(ss), 'snps total'); fs.mem()

    # read in snp coordinates
    print('reading snp info')
    snps = pd.concat([refpanel.bim_df(c) for c in args.chroms], axis=0)
    l2 = pd.concat([pd.read_csv(args.ldscores_full_chr+str(c)+'.l2.ldscore.gz', sep='\t')
        for c in args.chroms], axis=0)
    snps['L2'] = l2.L2
    print(len(snps), 'snps total')
    typed = ss.bhat.notnull().values
    snps = snps[typed]
    ss = ss[typed]
    print(len(snps), 'snps with sumstats'); fs.mem()
    loci = get_locus_inds(snps, loci, args.chroms)

    y = ss[args.use].values
    y_b = ss.bhat.values
    y_h = ss.ghat_hm3.values
    results = pd.DataFrame()
    for anum, annot in enumerate(annots):
        a = pd.concat([annot.sannot_df(c) for c in args.chroms], axis=0)
        names = annot.names(22)
        print(time.time()-t0, ':', anum, names); fs.mem()
        for n in names:
            x = a.loc[typed,n].values
            q = x*y
            q_b = x*y_b
            q_h = x*y_h

            myresults = loci.copy()
            myresults['annot'] = n
            myresults['pheno'] = nice_ss_name

            myresults['maxz'] = np.nan_to_num(np.array([
                np.abs(q[i:j]).sum()/np.linalg.norm(q[i:j])
                for i,j in zip(myresults.startind, myresults.endind)]))
            print('of', len(myresults), np.percentile(myresults.maxz, np.arange(0,101,1)))
            mask = myresults.maxz >= args.low_thresh
            myresults_ = myresults[mask]
            myresults.loc[mask,'agg'] = np.nan_to_num(np.array([
                ((ss[i:j].N*(ss.ahat[i:j]**2)).sum() - (j-i))/snps.L2[i:j].mean()/(j-i)
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'normbhat'] = np.nan_to_num(np.array([
                np.linalg.norm(ss.bhat[i:j])
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'h2g'] = np.nan_to_num(np.array([
                ss.bhat[i:j].dot(ss.ahat[i:j]) - (j-i)/100000.
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'v2'] = np.nan_to_num(np.array([
                np.linalg.norm(x[i:j], ord=2)
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'z'] = np.nan_to_num(np.array([
                q[i:j].sum()/np.linalg.norm(q[i:j])
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'z_b'] = np.nan_to_num(np.array([
                q_b[i:j].sum()/np.linalg.norm(q_b[i:j])
                for i,j in zip(myresults_.startind, myresults_.endind)]))
            myresults.loc[mask,'z_h'] = np.nan_to_num(np.array([
                q_h[i:j].sum()/np.linalg.norm(q_h[i:j])
                for i,j in zip(myresults_.startind, myresults_.endind)]))

            results = results.append(myresults[myresults.maxz >= args.low_thresh],
                    ignore_index=True)
            del x; del q; del q_h; del q_b
        del a; memo.reset(); gc.collect()

    print('writing results')
    print(len(results[results.maxz >= args.high_thresh]))
    print('mean z^2 among high', (results[results.maxz >= args.high_thresh].z**2).mean())
    print('max z^2 among high', (results[results.maxz >= args.high_thresh].z**2).max())
    results.to_csv(args.outfile_stem + '.gwresults', sep='\t', index=False, na_rep='nan')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-stem', #required=True,
            default='/groups/price/yakir/temp',
            help='path to an output file stem')
    parser.add_argument('--bhat-chr', #required=True,
            default='/groups/price/yakir/data/sumstats/processed/UKBB_body_HEIGHTz.'+\
                    'UKBB1K_50.0/',
            help='path to .bhat.gz files, without chr number or extension')
    parser.add_argument('--use', default='bhat')
    parser.add_argument('--low-thresh', type=float, default=6)
    parser.add_argument('--high-thresh', type=float, default=8)
    parser.add_argument('--sannot-chr', nargs='+', #required=True,
            default=['/groups/price/yakir/data/annot/basset/processed.a8/8988T/',
                    '/groups/price/yakir/data/annot/basset/processed.a8/HRT.FET/',
                    '/groups/price/yakir/data/annot/basset/processed.a8/H1-hESC/'],
            help='one or more paths to gzipped annot files, not including ' + \
                    'chromosome number or .sannot.gz extension')
    parser.add_argument('--baseline-sannot-chr', nargs='+',
            default=[])
    parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    parser.add_argument('--ldscores-full-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/allSNPs/'+\
                    '1000G.EUR.QC.',
            help='path to LD scores at a smallish set of SNPs. LD should be computed '+\
                    'to all potentially causal snps. This is used for '+\
                    'heritability estimation')
    parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/LDscore/LDscore.',
            help='path to LD scores at a smallish set of SNPs. LD should be computed '+\
                    'to all potentially causal snps. This is used for '+\
                    'heritability estimation')
    # parser.add_argument('--ldscores-reg-chr',
    #         default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
    #                 'weights.hm3_noMHC.',
    #         help='path to LD scores, where LD is computed to regression SNPs only.')
    # parser.add_argument('--ldscore-percentile', default=None,
    #         help='snps with ld score below this threshold wont be used for regression')
    # parser.add_argument('--svd-stem',
    #         default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/svds_95percent/',
    #         help='path to truncated svds of reference panel, by LD block')
    parser.add_argument('--loci',
            # default='/groups/price/yakir/data/reference/hg19.autosomes.1Mb.bed',
            default='/groups/price/yakir/data/reference/dixon_IMR90.TADs.hg19.bed',
            help='path to UCSC bed file containing one bed interval locus')
    parser.add_argument('--chroms', nargs='+', type=int, default=range(1,23))

    args = parser.parse_args()
    pretty.print_namespace(args)

    main(args)
