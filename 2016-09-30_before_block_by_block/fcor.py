from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
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
    # #TODO: this should never need to happen
    # print(len(sumstats), 'snps before dropping duplicates')
    # sumstats = sumstats.drop_duplicates(subset=['SNP'])
    # sumstats.reset_index(drop=True, inplace=True)
    print(len(sumstats), 'snps after dropping duplicates')

    for c in args.chroms:
        print('chr', c)
        print('merging sumstats and refpanel')
        data = pd.merge(refpanel.bim_df(c),
                sumstats.rename(columns={'A1':'A1_ss', 'A2':'A2_ss'}),
                how='left', on='SNP')

        data['TYPED'] = ~pd.isnull(data.Z)
        print(np.sum(data.TYPED), 'typed refpanel snps')

        print('reading sannot files')
        A_data, A = common.get_annots([a.sannot_filename(c) for a in annots])
        if args.average_annot:
            print('AVERAGING ANNOTATIONS')
            A_data['average'] = np.mean(A_data[A].values, axis=1)
            A = ['average']

        if args.weight_ld:
            print('reading ld scores for weighting')
            ldscores = common.get_ldscores(args.full_ldscores_chr, c)
            A_data['L2'] = np.maximum(ldscores.L2, 1)
            bigl2 = (A_data.L2 > 50).values
            A_data.loc[bigl2, A] = 0
            print('removed', np.sum(bigl2), 'snps with l2 > 50')
            A_data['L2_real'] = A_data.L2
            A_data['L2'] = 1
        else:
            A_data['L2'] = 1

        # check that A_data and data have the same snps in the same order
        if (A_data.SNP != data.SNP).any():
            raise Exception('ERROR: sannot and refpanel must have same snps in same order')

        # make alleles in A match alleles in reference panel
        toflip = (A_data.A1 == data.A2) & (A_data.A2 == data.A1)
        print('flipping', np.sum(toflip), 'alleles to match annotation to refpanel')
        A_data.ix[toflip, A] *= -1

        # remove strand ambiguous snps from A
        print('removing strand ambiguous snps')
        common.remove_strand_ambiguity(A_data, A,
                A1name='A1', A2name='A2')

        # ensure sumstats allele coding matches refpanel/annotations
        print('matching sumstats alleles to refpanel/annot alleles')
        # TODO: make it so that these lines consider complementary allele coding!!!!!
        toflip = (data['A1_ss'] == A_data['A2']) & (data['A2_ss'] == A_data['A1'])
        tokeep = (data['A1_ss'] == A_data['A1']) & (data['A2_ss'] == A_data['A2'])
        toremove = ~(toflip | tokeep) & data.TYPED
        data.ix[toflip, 'Z'] *= -1
        A_data.loc[toremove, A] = 0
        print('\tflipped', np.sum(toflip), 'typed SNPs')
        print('\tremoved from annotations', np.sum(toremove),
                'SNPs that didnt have the same pair of alleles')

        #remove mhc
        if int(c) == 6:
            mhc = (data.BP > 25500000) & (data.BP < 35500000) #TODO:read this from file
            A_data.loc[mhc, A] = 0 # this line assumes that A_data and data have same
                                    # snps in same order; this is checked above
            print('zeroed out annotation for', len(mhc), 'snps in mhc')

        # adjust by MAF if required
        if not args.per_norm_genotype:
            print('reading MAF info')
            data['MAF'] = refpanel.frq_df(c).MAF
            A_data[A] *= np.sqrt(2*data.MAF*(1-data.MAF)).reshape((-1,1)).repeat(
                    len(A), axis=1)

        # declare names of the annotations we'll need
        AO = pa.Annotation.names_observed(A) # annotation at typed snps only
        AOconv = pa.Annotation.names_conv(A) # convolution of annot at typed snps
        AOl = [n + '.l' for n in AO] # LDscore weighted AO
        AOlconv = [n + '.conv' for n in AOl] # convolution of AOl
        AOln = [n + '.l.n' for n in AO] # LDscore weighted, divided by sqrt(N)
        AOlnconv = [n + '.conv' for n in AOln] # convolution of AOln

        # merge annotation with data and create variations on annotation
        data[A] = A_data[A];
        data['A1_annot'] = A_data['A1']; data['A2_annot'] = A_data['A2']
        # AO
        data[AO] = A_data[A]
        data.loc[~data.TYPED, AO] = 0 # zero out annotation at unobserved snps
        # AOl
        data[AOl] = data[AO]
        data[AOl] = data[AOl].values / A_data.L2.values[:,None]
        # AOln
        data[AOln] = data[AOl]
        data.loc[data.TYPED, AOln] /= np.sqrt(data.loc[data.TYPED, 'N']).reshape(
                (-1,1)).repeat(len(A), axis=1)

        # do the convolution
        print('convolving annotations')
        common.convolve(data, AO + AOl + AOln, (refpanel, c),
                args.ld_breakpoints, args.mhc_path, fullconv=args.fullconv,
                newnames=AOconv + AOlconv + AOlnconv)

        # create the relevant numpy arrays
        data = data.loc[data.TYPED]
        V = data[AO].values
        Vl = data[AOl].values
        Vln = data[AOln].values # = v_m / sqrt(N_m)
        RV = data[AOconv].values
        RVl = data[AOlconv].values
        RVln = data[AOlnconv].values
        alphahat = data['Z'] / np.sqrt(data['N'].values)
        M = np.sum(data.TYPED)

        # compute the estimates and write output
        result = dict()
        result['VlTalphahat'] = Vl.T.dot(alphahat)
        result['VlTRV'] = Vl.T.dot(RV)
        result['VlnTRVln'] = Vln.T.dot(RVln)
        result['RVlTRVl'] = RVl.T.dot(RVl)
        result['sqnormV'] = np.linalg.norm(V, axis=0)**2
        result['supp'] = np.sum(V != 0, axis=0)
        result['names'] = A
        result['M'] = M

        outfile = fs.make_hidden('{}{}.res'.format(args.outfile_chr, c))
        fs.makedir_for_file(outfile)
        pickle.dump(result, open(outfile, 'w'))
        # result.to_csv(outfile, sep='\t', index=False)
        # data.to_csv(outfile + '.data', sep='\t', index=False)
        for key, val in result.items():
            print(key)
            print(val)
            print('===')

def merge(args):
    print('merge')
    ldscores = common.get_ldscores_allchr(args.ldscores_chr, args.chroms)[['SNP','L2']]
    sumstats = common.get_sumstats(args.sumstats, args.N_thresh)
    # TODO: erase the commented lines
    # N = np.mean(sumstats['N'].values)
    print('merging sumstats and ldscores')
    sumstats_merged = pd.merge(sumstats, ldscores, how='inner', on='SNP')
    sample_size_corr = np.sum(1./sumstats_merged.N.values)
    # sigma2g = (np.linalg.norm(sumstats_merged['Z']/np.sqrt(sumstats_merged['N']))**2 - \
    #         len(sumstats_merged)/N) / np.sum(sumstats_merged['L2'])
    sigma2g = (np.linalg.norm(sumstats_merged['Z']/np.sqrt(sumstats_merged['N']))**2 - \
            sample_size_corr) / np.sum(sumstats_merged['L2'])
    sigma2g = min(max(0,sigma2g), 1/len(sumstats_merged))
    h2g = sigma2g*len(sumstats_merged); s2e = 1-h2g
    print('h2g estimated at:', h2g, 'sigma2g:', sigma2g)

    # read in files and merge
    results_chr = {
            c : pickle.load(open(fs.make_hidden('{}{}.res'.format(args.outfile_chr, c))))
            for c in args.chroms}
    results = {}
    results['names'] = results_chr.values()[-1]['names']
    for c in args.chroms:
        del results_chr[c]['names']
    for key in results_chr.values()[-1].keys():
        results[key] = sum([results_chr[c][key] for c in args.chroms])

    # properties unrelated to sumstats
    print('ANNOTATION INFO (chr100 = all chromosomes)')
    properties = {}
    for i, a in enumerate(results['names']):
        properties[a] = pd.DataFrame(columns=
                ('M', '|supp(v)|', '|v|^2', 'vTRv', 'vTalphahat'),
                index=args.chroms + [100])
        for c in args.chroms:
            properties[a].loc[c, 'M'] = results_chr[c]['M']
            properties[a].loc[c, '|supp(v)|'] = results_chr[c]['supp'][i]
            properties[a].loc[c, '|v|^2'] = results_chr[c]['sqnormV'][i]
            properties[a].loc[c, 'vTRv'] = results_chr[c]['VlTRV'][i,i]
            properties[a].loc[c, 'vTalphahat'] = results_chr[c]['VlTalphahat'][i]
        properties[a].loc[100] = np.sum(properties[a], axis=0)
        print(a)
        print(properties[a])
        print()

    # correlation matrix of annotations
    correlations = pd.DataFrame(columns=results['names'], data=results['VlTRV'])
    print('CORRELATION MATRIX')
    print(correlations)
    print()
    correlations.to_csv('{}corr'.format(args.outfile_chr), sep='\t', index=False)

    print('INVERSE OF CORRELATION MATRIX')
    print(np.linalg.inv(correlations))
    print()

    # joint tests
    VTRVinv = np.linalg.inv(results['VlTRV'])
    joint_tests = pd.DataFrame(columns=
            ['name', 'vTalphahat/(VTRV)', 'Z_fX_fb', 'p_fX_fb'] + \
                    list(properties.values()[0].columns))
    for i, a in enumerate(results['names']):
        joint_tests.loc[i, 'name'] = a
        e = joint_tests.loc[i, 'vTalphahat/(VTRV)'] = \
                VTRVinv.dot(results['VlTalphahat'])[i]
        v1 = VTRVinv.dot(results['VlnTRVln'].dot(VTRVinv))[i,i]
        v2 = sigma2g*VTRVinv.dot(results['RVlTRVl'].dot(VTRVinv))[i,i]
        variance = s2e*v1
        joint_tests.loc[i, 'Z_fX_fb'] = e / np.sqrt(variance)
        joint_tests.loc[i, 'p_fX_fb'] = 2*stats.norm.sf(abs(e/np.sqrt(variance)),0,1)
        joint_tests.loc[i, list(properties[a].columns)] = properties[a].loc[100]
    print('JOINT TESTS')
    print(joint_tests)
    print()
    joint_tests.to_csv('{}joint'.format(args.outfile_chr), sep='\t', index=False)

    # separate marginal tests
    marginal_tests = pd.DataFrame(columns=
            ['name', 'gencorr',
                'Z_fX_fb', 'p_fX_fb',
                'Z_fX_rb', 'p_fX_rb',
                'Z_rX_fb', 'p_rX_fb',
                'Z_rX_rb', 'p_rX_rb',
                'vTRv/N', 'sigma2g*vTR2v'] + \
                        list(properties.values()[0].columns))
    for i, a in enumerate(results['names']):
        marginal_tests.loc[i, 'name'] = a
        e = results['VlTalphahat'][i]
        v1 = marginal_tests.loc[i, 'vTRv/N'] = results['VlnTRVln'][i,i]
        v2 = marginal_tests.loc[i, 'sigma2g*vTR2v'] = sigma2g*results['RVlTRVl'][i,i]
        marginal_tests.loc[i, 'gencorr'] = e/np.sqrt(h2g*results['VlTRV'][i,i])
        variances = {'fX_fb' : s2e*v1,
                'rX_fb': v1,
                'fX_rb': s2e*v1 + v2,
                'rX_rb': v1 + v2}
        for t, val in variances.items():
            marginal_tests.loc[i, 'Z_'+t] = e / np.sqrt(variances[t])
            marginal_tests.loc[i, 'p_'+t] = 2*stats.norm.sf(abs(e/np.sqrt(variances[t])),0,1)
        marginal_tests.loc[i, list(properties[a].columns)] = properties[a].loc[100]
    print('MARGINAL TESTS')
    print(marginal_tests)
    print()
    marginal_tests.to_csv('{}marginal'.format(args.outfile_chr), sep='\t', index=False)


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
            '--full-ldscores-chr', args.full_ldscores_chr,
            '--sannot-chr'] + args.sannot_chr + \
                (['-fullconv'] if args.fullconv else []) + \
                (['-per-norm-genotype'] if args.per_norm_genotype else []) + \
                (['-average-annot'] if args.average_annot else []) + \
                (['-weight-ld'] if args.weight_ld else []) + \
            ['--chroms', '$LSB_JOBINDEX']
    outfilename = fs.make_hidden(args.outfile_chr + '%I.out')
    fs.makedir_for_file(outfilename)
    submit_jobname = 'acor{}[{}]'.format(
                args.outfile_chr.replace('/','_'),
                ','.join(map(str, args.chroms)))
    jobid = bsub.submit(['python', '-u', __file__] + submit_args,
            outfilename,
            jobname=submit_jobname,
            time_in_hours=5,
            memory_GB=16,
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
            time_in_minutes=60,
            memory_GB=8,
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
            help='assume that v is in units of per normalized genotype rather than per ' +\
                    'allele')
    mainsubmit_parser.add_argument('-average-annot', action='store_true', default=False,
            help='average all annotations before doing anything')
    mainsubmit_parser.add_argument('--full-ldscores-chr',
            default='/groups/price/ldsc/reference_files/1000G_EUR_Phase3/allSNPs/'+\
                    '1000G.EUR.QC.',
            help='ld scores to and at all refpanel snps. We assume these are the same snps '+\
                    'in the same order as the reference panel. These ld scores will be '+\
                    'used to weight the quantity being estimated if weight-ld flag is used.')
    mainsubmit_parser.add_argument('-weight-ld', action='store_true', default=False,
            help='weight the quantity being estimated by the --full-ldscores-chr data.')

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
