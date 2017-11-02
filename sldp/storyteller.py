from __future__ import print_function, division
import numpy as np
import pandas as pd
import gprim.annotation as ga
import gprim.dataset as gd

# find loci with genome-wide significant SNPs that are consistent with the global signal
def write(args, name, z):
    print('STORYTELLING for ', name, 'z=', z)
    annot = [ga.Annotation(annot) for annot in args.sannot_chr
            if name in ga.Annotation(annot).names(22, RV=True)][0]

    backgroundannots = [ga.Annotation(annot) for annot in args.background_sannot_chr]
    background_names = sum([a.names(22, True) for a in backgroundannots], [])
    background_names = [n for n in background_names if '.R' in n]
    print('focal annotation columns:', annot.names(22, True))
    print('background annotations:', background_names)
