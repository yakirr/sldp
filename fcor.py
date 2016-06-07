from __future__ import print_function, division
import argparse
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import pyutils.pretty as pretty
import pyutils.bsub as bsub
import pyutils.fs as fs
import primitives.dataset as prd
import primitives.annotation as pa
import common


def ldscores_sumstats_sigma2g(args):
    ldscores = common.get_ldscores_allchr(args.ldscores_chr, args.chroms)
    ldscores = ldscores[['SNP','L2']]

    sumstats = common.get_sumstats(args.sumstats)
    #TODO: change this to mean N at some point? or the appropriate 1/N1+1/N2+...+1/Nm?
    N = np.mean(sumstats['N'].values)
    print('merging sumstats and ldscores')
    sumstats_merged = pd.merge(sumstats, ldscores, how='inner', on='SNP')
    sigma2g = (np.linalg.norm(sumstats_merged['Z']/np.sqrt(sumstats_merged['N']))**2 - \
            len(sumstats_merged)/N) / np.sum(sumstats_merged['L2'])
    sigma2g = min(max(0,sigma2g), 1/len(sumstats_merged))
    print('h2g estimated at:', sigma2g*len(sumstats_merged), 'sigma2g:', sigma2g)
    return sumstats, sumstats_merged, sigma2g, N

def cor_fe(args):
    print('FIXED EFFECTS')
    sumstats, sumstats_hm3, sigma2g, _ = ldscores_sumstats_sigma2g(args)

    print('loading information by chromosome')
    annot, conv = [], []
    refpanel = prd.Dataset(args.bfile_chr)
    annots = [pa.Annotation(fname) for fname in args.annot_chr]
    convannots = [pa.Annotation(fname) for fname in args.conv_chr]

    # load annot and conv files and merge across chromosomes
    for chrnum in args.chroms:
        print('== chr', chrnum)
        # read data
        myannot, annot_names = common.get_annots(
                [a.sannot_filename(chrnum) for a in annots])
        myconv, conv_names = common.get_convs(
                [a.conv_filename(chrnum, args.fullconv) for a in convannots],
                [a.sannot_filename(chrnum) for a in annots],
                (refpanel, chrnum),
                args.ld_breakpoints,
                args.mhc_path)
        # basic error checks
        if (myannot['SNP']!=myconv['SNP']).any():
            raise Exception(
                    'ERROR: annot and conv do not contain identical snps in the same order')
        if find_conv1_names(myconv.columns) != make_conv1_names(annot_names):
            raise Exception(
                    'ERROR: conv file must contain same columns in same order as annot file')
        # append to previous chromosomes
        annot.append(myannot)
        conv.append(myconv)
    annot = pd.concat(annot, axis=0).reset_index(drop=True)
    conv = pd.concat(conv, axis=0).reset_index(drop=True)

    # merge annot and conv into one dataframe with the right columns
    annot[conv_names] = \
            conv[conv_names]
    print('==done')

    # merge sumstats with the annot, flipping alleles if necessary
    print('merging sumstats and annot file')
    zannotconv = pd.merge(sumstats_hm3, annot, how='inner', on='SNP')
    print(len(zannotconv), 'SNPs have both sumstats and refpanel info')

    print('checking for strand ambiguous snps')
    common.check_for_strand_ambiguity(zannotconv, annot_names)

    print('matching sumstats alleles and annot alleles')
    toflip = zannotconv['A1_x'] == zannotconv['A2_y']
    zannotconv.ix[toflip, 'Z'] *= -1

    # create the relevant numpy arrays
    N = np.min(zannotconv['N'].values) # TODO: try mean instead of min?
    V = zannotconv[annot_names].values
    RV = zannotconv[conv_names].values
    alphahat = zannotconv['Z'] / np.sqrt(zannotconv['N'].values)

    # compute the estimate
    VTRV = V.T.dot(RV)
    estimate = np.linalg.solve(VTRV, V.T.dot(alphahat))
    cov = np.linalg.inv(VTRV) / N
    cov2 = sigma2g * np.linalg.solve(VTRV, np.linalg.solve(VTRV, RV.T.dot(RV)).T)
    if args.reg_var:
        cov += cov2

    # print output
    output = pd.DataFrame(columns=
            ('NAME','MU_EST','MU_STDERR','MU_Z','MU_P', 'TOP', 'BOTTOM', 'COV1', 'COV2'))
    for i, n in enumerate(annot_names):
        std = np.sqrt(cov[i,i])
        output.loc[i] = [n, estimate[i], std,
                estimate[i]/std,
                2*stats.norm.sf(abs(estimate[i]/std),0,1),
                ','.join(map(str, V.T.dot(alphahat).reshape((-1,)))),
                ','.join(map(str, VTRV.reshape((-1,)))),
                ','.join(map(str, cov.reshape((-1,)))),
                ','.join(map(str, cov2.reshape((-1,))))]
    print(output)

    covariance = pd.DataFrame(columns=annot_names, data=cov)
    print('\nfull covariance matrix:\n{}'.format(cov))

    if args.out is not None:
        output.to_csv(args.out+'.results', sep='\t', index=False)
        covariance.to_csv(args.out+'.cov', sep='\t', index=False)

def cor_re(args):
    print('RANDOM EFFECTS')
    _, sumstats, sigma2g, N = ldscores_sumstats_sigma2g(args)

    print('loading information by chromosome')
    annot, conv = [], []
    biascorrection = 0
    refpanel = prd.Dataset(args.bfile_chr)
    annots = [pa.Annotation(fname) for fname in args.annot_chr]
    convannots = [pa.Annotation(fname) for fname in args.conv_chr]

    # load annot and conv files and merge across chromosomes
    for chrnum in args.chroms:
        print('== chr', chrnum)
        # read data
        myannot, annot_names = common.get_annots(
                [a.sannot_filename(chrnum) for a in annots])
        myconv, conv_names = common.get_convs(
                [a.conv_filename(chrnum, args.fullconv) for a in convannots],
                [a.sannot_filename(chrnum) for a in annots],
                (refpanel, chrnum),
                args.ld_breakpoints,
                args.mhc_path)
        myldscores = common.get_ldscores(args.ldscores_chr + chrnum + '.l2.ldscore.gz')
        myldscores_weights = common.get_ldscores(
            args.ldscores_weights_chr + chrnum + '.l2.ldscore.gz')
        # if np.any(myldscores['SNP'] != myldscores_weights['SNP']):
        #     print('the two sets of LD scores dont have the same snps in the same order!')
        #     exit()
        # myldscores['L2_reg'] = myldscores_weights['L2']
        myldscores_weights.rename(columns={'L2': 'L2_reg'}, inplace=True)
        myldscores = pd.merge(myldscores, myldscores_weights[['SNP', 'L2_reg']], how='inner',
            on='SNP')

        # basic error checks
        # if (myannot['SNP']!=myconv['SNP']).any() or (myannot['SNP']!=myldscores['SNP']).any():
        #     raise Exception(
        #             'ERROR: annot, conv, and ldscores do not contain identical snps ' +\
        #                     'in the same order')
        if find_conv1_names(myconv.columns) != make_conv1_names(annot_names):
            raise Exception(
                    'ERROR: conv file must contain same columns in same order as annot file')

        # compute weights
        if args.noweights:
            print('NOT using weights')
            myldscores['Lambda'] = 1
        else:
            print('using weights')
            myldscores['Lambda'] = 1./(
                np.maximum(1, myldscores['L2_reg']) / N + \
                    sigma2g * np.maximum(1, myldscores['L2']) * \
                        np.maximum(1, myldscores['L2_reg']))

        # attach weights to conv
        myconv = pd.merge(myconv, myldscores[['SNP', 'Lambda']], how='left', on='SNP')
        myconv.fillna(0, inplace=True)
        # zero out weights at untyped snps
        myconv = pd.merge(myconv, sumstats[['SNP', 'Z']], how='left', on='SNP')
        myconv.loc[pd.isnull(myconv['Z']), 'Lambda'] = 0
        myconv.drop(['Z'], inplace=True, axis=1)

        # remove low-maf snps from regression if necessary
        print('applying MAF threshold of', args.maf_thresh)
        maf = refpanel.frq_df(chrnum)['MAF'].values
        print('\tremoving', np.sum(maf < args.maf_thresh), 'snps from regression')
        myconv.loc[maf < args.maf_thresh, 'Lambda'] = 0

        # compute bias correction for denominator of regression if necessary
        if args.biascorrect:
            biaschr = common.get_biascorrection(
                    myannot[annot_names].values,
                    myconv[make_conv1_names(annot_names)].values,
                    myconv['Lambda'].values,
                    (refpanel, chrnum),
                    args.ld_breakpoints,
                    args.mhc_path)
            print('\tbias correction for this chr:', biaschr)
            biascorrection += biaschr
        # reconvolve
        myconv[conv_names+'.w'] = myconv['Lambda'].values[:,None] * myconv[conv_names]
        myconv, _ = common.convolve(myconv, conv_names+'.w',
                (refpanel, chrnum), args.ld_breakpoints, args.mhc_path,
                fullconv=args.fullconv)
        # append to previous chromosomes
        annot.append(myannot)
        conv.append(myconv)
    annot = pd.concat(annot, axis=0).reset_index(drop=True)
    conv = pd.concat(conv, axis=0).reset_index(drop=True)

    # merge annot and conv into one dataframe with the right columns
    names = np.concatenate([conv_names,conv_names+'.w',conv_names+'.w.conv1'])
    annot[names] = \
            conv[names]
    print('==done')

    # merge annot with sumstats, flipping alleles if necessary
    print('merging sumstats and annot file')
    zannotconv = pd.merge(sumstats, annot, how='inner', on='SNP')
    print('matching sumstats alleles and annot alleles')
    toflip = zannotconv['A1_x'] == zannotconv['A2_y']
    zannotconv.ix[toflip, 'Z'] *= -1

    # create the relevant numpy arrays
    V = zannotconv[annot_names].values
    RV = zannotconv[conv_names].values
    LambdaRV = zannotconv[[n+'.w' for n in conv_names]].values
    RLambdaRV = zannotconv[[n+'.w.conv1' for n in conv_names]].values
    alphahat = zannotconv['Z'] / np.sqrt(zannotconv['N'].values)

    # compute the estimate
    Sigma = RV.T.dot(LambdaRV) - biascorrection
    estimate = np.linalg.solve(Sigma, LambdaRV.T.dot(alphahat))
    Sigmai = np.linalg.inv(Sigma)
    var1 = Sigmai.dot(LambdaRV.T).dot(RLambdaRV.dot(Sigmai)) / N
    var2 = np.linalg.solve(Sigma, RLambdaRV.T)
    var2 = sigma2g * var2.dot(var2.T)
    cov = var1 + var2

    # print output
    output = pd.DataFrame(columns=
            ('NAME','MU_EST','MU_STDERR','MU_Z','MU_P', 'TOP', 'BOTTOM', 'COV1', 'COV2'))
    for i, n in enumerate(annot_names):
        std = np.sqrt(cov[i,i])
        output.loc[i] = [n, estimate[i], std,
                estimate[i]/std,
                2*stats.norm.sf(abs(estimate[i]/std),0,1),
                ','.join(map(str, LambdaRV.T.dot(alphahat).reshape((-1,)))),
                ','.join(map(str, Sigma.reshape((-1,)))),
                ','.join(map(str, var1.reshape((-1,)))),
                ','.join(map(str, var2.reshape((-1,))))]
    print(output)

    covariance = pd.DataFrame(columns=annot_names, data=cov)
    print('\nfull covariance matrix:\n{}'.format(cov))

    if args.out is not None:
        output.to_csv(args.out+'.results', sep='\t', index=False)
        covariance.to_csv(args.out+'.cov', sep='\t', index=False)

def conv(args):
    print('CONV')
    print('loading information by chromosome')
    annot, conv = pd.DataFrame(), pd.DataFrame()
    refpanel = prd.Dataset(args.bfile_chr)
    annots = [pa.Annotation(fname) for fname in args.annot_chr]
    convannots = [pa.Annotation(fname) for fname in args.conv_chr]

    # load annot and conv files. they will automatically be created if they don't exist
    for chrnum in args.chroms:
        print('== chr', chrnum)
        # read data
        #TODO: implement print-snps
        myconv, conv_names = common.get_convs(
                [a.conv_filename(chrnum, args.fullconv) for a in convannots],
                [a.sannot_filename(chrnum) for a in annots],
                (refpanel, chrnum),
                args.ld_breakpoints,
                args.mhc_path)






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

        print('reading MAF info')
        data['MAF'] = refpanel.frq_df(c).MAF

        print('reading sannot files')
        A, A_names = common.get_annots([a.sannot_filename(c) for a in annots])
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

        print('convolving annotations')
        common.convolve(data, AO_names + AOw_names, (refpanel, c),
                args.ld_breakpoints, args.mhc_path,
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
    df = df.applymap(json.loads).applymap(np.array)
    sums = np.sum(df, axis=0)
    print(df)
    print(np.sum(df, axis=0))
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
    print(output)

    e = estimate[0]; c1 = cov1_[0,0]; c2 = cov2_[0,0]; s2e = 1-h2g
    variances = [('fX, fb', s2e*c1),
            ('rX, fb', c1),
            ('fX, rb', s2e*c1 + c2),
            ('rX, rb', c1 + c2)]
    for name, var in variances:
        std = np.sqrt(var)
        print('{}: Z={}, 2-sided p={}'.format(
            name, e/std, 2*stats.norm.sf(abs(e/std),0,1)))

    covariance = pd.DataFrame(columns=A_names, data=cov)
    print('\nfull covariance matrix:\n{}'.format(cov))

    #TODO: save output and delete intermediate files?


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
            ['--chroms', '$LSB_JOBINDEX']
    outfilename = args.outfile_chr + '%I.out'
    submit_jobname = 'acor{}[{}-{}]'.format(
                args.outfile_chr.replace('/','_'), args.chr_start, args.chr_end)
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
            ['--chroms'] + map(str, range(args.chr_start, args.chr_end+1))
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
    mainsubmit_parser.add_argument('--bfile-chr', required=True,
            help='path to plink bfile of reference panel to use, not including chrom num')
    mainsubmit_parser.add_argument('--sannot-chr', required=True, nargs='+',
            help='space-delimited list of paths to gzipped annot files, not including ' + \
                    'chromosome number or .annot.gz extension')
    #   optional
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
    # mainsubmit_parser.add_argument('--full-ldscores-chr', default='None',
    #         help='ld scores to and at all refpanel snps. If supplied these will be used ' +\
    #                 'to weight the quantity being estimated')

    # define arguments for submit and merge
    submitmerge_parser = argparse.ArgumentParser(add_help=False)
    #   required
    submitmerge_parser.add_argument('--ldscores-chr', required=True,
            help='path to a set of .l2.ldscore.gz files containin a column named L2 with ' + \
                    'ld scores at a smallish set of snps. ld should be computed to other ' + \
                    'snps in the set only')
    submitmerge_parser.add_argument('-reg-var', action='store_true', default=False,
            help='report a std. error based on a random-beta model')

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
    sp_sub.add_argument('--chr-start', type=int,
            default=1,
            help='which chromosome to start submitting at')
    sp_sub.add_argument('--chr-end', type=int,
            default=22,
            help='which chromosome to end submitting at (inclusive)')
    sp_sub.add_argument('-debug', action='store_true', default=False,
            help='do not actually submit the jobs, just print the submission commands')

    # define arguments for merge specifically
    sp_merge.add_argument('--chroms', nargs='+',
            default=range(1,23),
            help='which chromosomes to aggregate')

    bsub.choose_parser_and_run(parser)
