from __future__ import print_function, division
import pandas as pd
import numpy as np
import argparse
import gzip
import common; reload(common)
import gprim.dataset as prd
import gprim.annotation as pa

parser = argparse.ArgumentParser()
parser.add_argument('--sumstats', required=True)
parser.add_argument('--sannot-chr', nargs='+', required=True)
parser.add_argument('--outfile-chr', required=True)
parser.add_argument('--Lambda', type=float, default=0.1)
parser.add_argument('--N-thresh', type=float, default=1.0,
        help='this times the 90th percentile N is the sample size threshold')
parser.add_argument('--bfile-chr',
        default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
            '1000G.EUR.QC.',
        help='path to plink bfile of reference panel to use, not including chrom num')
parser.add_argument('--ld-breakpoints',
        default='/groups/price/yakir/data/reference/pickrell_breakpoints.hg19.eur.bed',
        help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                ' breakpoint')
parser.add_argument('--mhc-path',
        default='/groups/price/yakir/data/reference/hg19.MHC.bed',
        help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                ' breakpoint')
parser.add_argument('--chroms', nargs='+',
        default=range(1,23),
        help='For main: which chromosomes to analyze.')
args = parser.parse_args()

sumstats = common.get_sumstats(args.sumstats, args.N_thresh)
refpanel = prd.Dataset(args.bfile_chr)
annots = [pa.Annotation(fname) for fname in args.sannot_chr]

for c in args.chroms:
    data = pd.merge(refpanel.bim_df(c),
            sumstats.rename(columns={'A1':'A1_ss', 'A2':'A2_ss'}),
            how='left', on='SNP')
    data['maf'] = refpanel.frq_df(c).MAF

    # mark the snps we're ultimately interested in
    data['typed'] = ~pd.isnull(data.Z)
    data.typed &= ~common.get_ambiguous(data, 'A1', 'A2')
    print(np.sum(data.typed), 'typed, unambiguous refpanel snps')

    # read the annotations and clean them
    print('reading sannot files')
    A_data, A = common.get_annots([a.sannot_filename(c) for a in annots])
    RA = ['R.'+a for a in A]
    conA = ['cond.'+a for a in A]
    data = pd.merge(data,
            A_data.rename(columns={'A1':'A1_a', 'A2':'A2_a'}).drop(
                ['CHR', 'CM', 'BP'], axis=1),
            how='left', on='SNP')
    # remove strand ambiguous snps from annot
    print('removing strand ambiguous snps')
    common.remove_strand_ambiguity(data, A,
            A1name='A1', A2name='A2')
    # ensure annot allele coding matches refpanel/annotations
    print('matching annot alleles to refpanel alleles')
    toflip = (data['A1_a'] == data['A2']) & (data['A2_a'] == data['A1'])
    data.ix[toflip, A] *= -1
    print('\tflipped annot values at', np.sum(toflip), 'SNPs')
    tokeep = (data['A1_a'] == data['A1']) & (data['A2_a'] == data['A2'])
    toremove = ~(toflip | tokeep) & data.typed
    if np.sum(toremove) > 0:
        print('ERROR')
        exit()
    # correct for MAF
    print('converting to per normalized genotype')
    data[A] *= np.sqrt(2*data.maf*(1-data.maf)).reshape((-1,1)).repeat(
            len(A), axis=1)


    common.convolve(data, A, (refpanel, c), args.ld_breakpoints, args.mhc_path,
            newnames=RA)
    condensed = common.mult_by_Rinv_ldblocks(data[RA].values,
            data.typed.values, (refpanel, c),
            args.ld_breakpoints, args.mhc_path, l=args.Lambda)
    condensed[np.isnan(condensed)] = 0
    data = pd.concat([data, pd.DataFrame(condensed, columns=conA)], axis=1)

    print('writing output')
    with gzip.open(args.outfile_chr+str(c)+'.sannot.gz', 'w') as f:
        data[['CHR','BP','SNP','A1','A2']+conA].to_csv(f,
                sep='\t', index=False)
