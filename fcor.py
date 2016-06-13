from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import gprim.dataset as prd
import gprim.annotation as pa
import common


def main(args):
    print('main')
    refpanel = prd.Dataset(args.bfile_chr)
    annots = [pa.Annotation(fname) for fname in args.sannot_chr]

    print('reading sumstats')
    sumstats = common.get_sumstats(args.sumstats, args.N_thresh)

    for c in args.chroms:
        print('chr', c)
        print('merging sumstats and refpanel')
        data = pd.merge(refpanel.bim_df(c).drop(['A1','A2'], axis=1),
                sumstats.rename(columns={'A1':'A1_ss', 'A2':'A2_ss'}),
                how='left', on='SNP')
        data['TYPED'] = ~pd.isnull(data.Z)

        print('reading sannot files')
        A, A_names = common.get_annots([a.sannot_filename(c) for a in annots])
        if not args.per_norm_genotype:
            print('reading MAF info')
            data['MAF'] = refpanel.frq_df(c).MAF
            A[A_names] *= np.sqrt(2*data.MAF[:,None]*(1-data.MAF[:,None]))
        AO_names = pa.Annotation.names_observed(A_names)
        AOconv_names = pa.Annotation.names_conv(A_names)
        AOw_names = [n + '.w' for n in AO_names]
        AOwconv_names = [n + '.conv' for n in AOw_names]
        if (A.SNP != data.SNP).any():
            raise Exception('ERROR: sannot and refpanel must have same snps in same order')
        data[A_names] = A[A_names]; data[AO_names] = A[A_names]
        data.loc[~data.TYPED, AO_names] = 0 # zero out annotation at unobserved snps
        data[AOw_names] = data[AO_names]
        data.loc[data.TYPED, AOw_names] /= np.sqrt(data.loc[data.TYPED, 'N'].values[:,None])
        data['A1_annot'] = A['A1']; data['A2_annot'] = A['A2']

        print('checking for strand ambiguous snps')
        common.check_for_strand_ambiguity(data, AO_names,
                A1name='A1_annot', A2name='A2_annot')

        print('matching sumstats alleles and annot alleles')
        toflip = data['A1_ss'] == data['A2_annot']
        data.ix[toflip, 'Z'] *= -1
        print('\tflipped', np.sum(toflip), 'SNPs')

        print('convolving annotations')
        common.convolve(data, AO_names + AOw_names, (refpanel, c),
                args.ld_breakpoints, args.mhc_path, fullconv=args.fullconv,
                newnames=AOconv_names+AOwconv_names)

        # create the relevant numpy arrays
        data = data.loc[data.TYPED]
        V = data[AO_names].values
        LV = data[AOw_names].values
        RV = data[AOconv_names].values
        RLV = data[AOwconv_names].values
        alphahat = data['Z'] / np.sqrt(data['N'].values)

        # compute the estimates and write output
        result = pd.DataFrame(
                columns=['VTalphahat',
                    'VTRV',
                    'cov1',
                    'cov2',
                    '|supp(V)|',
                    'names'],
                data=[[
                    V.T.dot(alphahat),
                    V.T.dot(RV),
                    LV.T.dot(RLV),
                    RV.T.dot(RV),
                    np.sum(V != 0, axis=0),
                    ','.join(A_names)
                    ]])
        outfile = '{}{}.res'.format(args.outfile_chr, c)
        fs.makedir_for_file(outfile)
        result.to_csv(outfile, sep='\t', index=False)
        print(result)

def merge(args):
    print('merge')
    ldscores = common.get_ldscores_allchr(args.ldscores_chr, args.chroms)[['SNP','L2']]
    sumstats = common.get_sumstats(args.sumstats, args.N_thresh)
    #TODO: change this to mean N at some point? or the appropriate 1/N1+1/N2+...+1/Nm?
    N = np.mean(sumstats['N'].values)
    print('merging sumstats and ldscores')
    sumstats_merged = pd.merge(sumstats, ldscores, how='inner', on='SNP')
    sigma2g = (np.linalg.norm(sumstats_merged['Z']/np.sqrt(sumstats_merged['N']))**2 - \
            len(sumstats_merged)/N) / np.sum(sumstats_merged['L2'])
    sigma2g = min(max(0,sigma2g), 1/len(sumstats_merged))
    h2g = sigma2g*len(sumstats_merged)
    print('h2g estimated at:', h2g, 'sigma2g:', sigma2g)

    # read in files and merge
    df = pd.concat([
        pd.read_csv('{}{}.res'.format(args.outfile_chr, c),
            sep='\t')
        for c in args.chroms], axis=0).reset_index(drop=True)
    A_names = df.names[0].split(','); df.drop(['names'], axis=1, inplace=True)

    # parse arrays inside dataframe into numpy arrays
    def parse(s):
        s = s.replace('0.]', '0]')
        return np.array(json.loads(s))
    df = df.applymap(parse)

    sums = np.sum(df, axis=0)
    with open('{}all.res'.format(args.outfile_chr), 'w') as outfile:
        print(df); print(df, file=outfile)
        print(np.sum(df, axis=0)); print(np.sum(df, axis=0), file=outfile)
        VTalphahat = np.sum(df.VTalphahat)
        VTRV = np.sum(df.VTRV)
        cov1 = np.sum(df.cov1)
        cov2 = np.sum(df.cov2)

        # compute estimate and covariance matrix
        estimate = np.linalg.solve(VTRV, VTalphahat)
        cov1_ = np.linalg.solve(VTRV, np.linalg.solve(VTRV, cov1).T)
        cov2_ = sigma2g * np.linalg.solve(VTRV, np.linalg.solve(VTRV, cov2).T)
        if args.reg_var:
            cov = cov1_ + cov2_
        else:
            cov = cov1_

        # print output
        output = pd.DataFrame(columns=
                ('NAME','MU_EST','MU_STDERR','MU_Z','MU_P', 'TOP', 'BOTTOM', 'COV1', 'COV2'))
        for i, n in enumerate(A_names):
            std = np.sqrt(cov[i,i])
            output.loc[i] = [n, estimate[i], std,
                    estimate[i]/std,
                    2*stats.norm.sf(abs(estimate[i]/std),0,1),
                    VTalphahat,
                    VTRV,
                    cov1_,
                    cov2_]
        print(output); print(output, file=outfile)

        e = estimate[0]; c1 = cov1_[0,0]; c2 = cov2_[0,0]; s2e = 1-h2g
        variances = [('fX, fb', s2e*c1),
                ('rX, fb', c1),
                ('fX, rb', s2e*c1 + c2),
                ('rX, rb', c1 + c2)]
        for name, var in variances:
            std = np.sqrt(var)
            print('{}: Z={}, 2-sided p={}'.format(
                name, e/std, 2*stats.norm.sf(abs(e/std),0,1)))
            print('{}: Z={}, 2-sided p={}'.format(
                name, e/std, 2*stats.norm.sf(abs(e/std),0,1)), file=outfile)

        covariance = pd.DataFrame(columns=A_names, data=cov)
        print('\nfull covariance matrix:\n{}'.format(cov))
        print('\nfull covariance matrix:\n{}'.format(cov), file=outfile)

    if not args.keep_files:
        print('removing files...')
        for c in args.chroms:
            try:
                os.remove('{}{}.res'.format(args.outfile_chr, c))
                os.remove('{}{}.out'.format(args.outfile_chr, c))
            except OSError:
                pass


def submit(args):
    print('submit')
    # submit parallel jobs
    submit_args = ['--outfile-chr', args.outfile_chr,
            '--sumstats', args.sumstats,
            '--N-thresh', str(args.N_thresh),
            'main',
            '--ld-breakpoints', args.ld_breakpoints,
            '--mhc-path', args.mhc_path,
            '--bfile-chr', args.bfile_chr,
            '--sannot-chr'] + args.sannot_chr + \
                (['-fullconv'] if args.fullconv else []) + \
                (['-per-norm-genotype'] if args.per_norm_genotype else []) + \
            ['--chroms', '$LSB_JOBINDEX']
    outfilename = args.outfile_chr + '%I.out'
    submit_jobname = 'acor{}[{}]'.format(
                args.outfile_chr.replace('/','_'),
                ','.join(map(str, args.chroms)))
    jobid = bsub.submit(['python', '-u', __file__] + submit_args,
            outfilename,
            jobname=submit_jobname,
            time_in_hours=1,
            memory_GB=4,
            debug=args.debug)

    # submit merge job
    merge_args = ['--outfile-chr', args.outfile_chr,
            '--sumstats', args.sumstats,
            '--N-thresh', str(args.N_thresh),
            'merge',
            '--ldscores-chr', args.ldscores_chr] + \
                (['-reg-var'] if args.reg_var else []) + \
                (['-keep-files'] if args.keep_files else []) + \
            ['--chroms'] + map(str, args.chroms)
    outfilename = args.outfile_chr + 'all.out'
    bsub.submit(['python', '-u', __file__] + merge_args,
            outfilename,
            jobname='merge_acor{}'.format(
                args.outfile_chr.replace('/','_')),
            time_in_minutes=20,
            memory_GB=2,
            depends_on=jobid,
            debug=args.debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # define common arguments
    parser.add_argument('--outfile-chr', required=True,
            help='path to an output file stem')
    parser.add_argument('--sumstats', required=True,
            help='path to sumstats.gz file, including extension')
    parser.add_argument('--N-thresh', type=float, default=1.0,
            help='this times the 90th percentile N is the sample size threshold')

    # define arguments for main and submit
    mainsubmit_parser = argparse.ArgumentParser(add_help=False)
    #   required
    mainsubmit_parser.add_argument('--sannot-chr', required=True, nargs='+',
            help='space-delimited list of paths to gzipped annot files, not including ' + \
                    'chromosome number or .annot.gz extension')
    #   optional
    mainsubmit_parser.add_argument('--bfile-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/' + \
                '1000G.EUR.QC.',
            help='path to plink bfile of reference panel to use, not including chrom num')
    mainsubmit_parser.add_argument('--ld-breakpoints',
            default='/groups/price/yakir/data/reference/pickrell_breakpoints.hg19.eur.bed',
            help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                    ' breakpoint')
    mainsubmit_parser.add_argument('--mhc-path',
            default='/groups/price/yakir/data/reference/hg19.MHC.bed',
            help='path to UCSC bed file containing one zero-length bed interval per LD' + \
                    ' breakpoint')
    mainsubmit_parser.add_argument('-fullconv', action='store_true', default=False,
            help='use/generate .fullconv.gz files, which dont take LD blocks into account')
    mainsubmit_parser.add_argument('-convfile', action='store_true', default=False,
            help='use the .conv.gz file corresponding to the annotation supplied, ' + \
                    'rather than generating the .conv.gz file at runtime')
    mainsubmit_parser.add_argument('-per-norm-genotype', action='store_true', default=False,
            help='assume that v is in unites of per normalized genotype rather than per ' +\
                    'allele')
    # mainsubmit_parser.add_argument('--full-ldscores-chr', default='None',
    #         help='ld scores to and at all refpanel snps. If supplied these will be used ' +\
    #                 'to weight the quantity being estimated')

    # define arguments for submit and merge
    submitmerge_parser = argparse.ArgumentParser(add_help=False)
    #   optional
    submitmerge_parser.add_argument('--ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/'+\
                    'weights.hm3_noMHC.',
            help='path to a set of .l2.ldscore.gz files containing a column named L2 with '+\
                    'ld scores at a smallish set of snps. ld should be computed to other '+\
                    'snps in the set only')
    submitmerge_parser.add_argument('-reg-var', action='store_true', default=False,
            help='report a std. error based on a random-beta model')
    submitmerge_parser.add_argument('-keep-files', default=False, action='store_true',
            help='tell merge not to delete per-chromosome intermediate files when finished')

    # create actual subparsers
    sp_main, sp_sub, sp_merge = bsub.add_main_and_submit(parser, main, submit,
            merge_function=merge,
            main_parents=[mainsubmit_parser],
            submit_parents=[mainsubmit_parser, submitmerge_parser],
            merge_parents=[submitmerge_parser])

    # define arguments for main specifically
    sp_main.add_argument('--chroms', nargs='+',
            default=range(1,23),
            help='For main: which chromosomes to analyze.')

    # define arguments for submit specifically
    sp_sub.add_argument('--chroms', nargs='+',
            default=range(1,23),
            help='Which chromosomes to submit.')
    sp_sub.add_argument('-debug', action='store_true', default=False,
            help='do not actually submit the jobs, just print the submission commands')

    # define arguments for merge specifically
    sp_merge.add_argument('--chroms', nargs='+',
            default=range(1,23),
            help='which chromosomes to aggregate')

    bsub.choose_parser_and_run(parser)